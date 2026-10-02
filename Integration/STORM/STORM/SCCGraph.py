from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOK: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
QUEEN: Tuple[Tuple[int, int], ...] = ROOK + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def lattice_adj(bx, by, neigh: str = "rook") -> sp.csr_matrix:
    """
    Exact adjacency between occupied tiles of an integer lattice.

    neigh='rook'  -> 4 edge-sharing neighbours (the paper's effective graph)
    neigh='queen' -> 8 neighbours (adds the diagonals)

    Missing tiles (off-tissue, dropped by QC) simply have no edge; nothing is
    interpolated. Binary, symmetric, zero diagonal.
    """
    if neigh == "rook":
        offsets = ROOK
    elif neigh == "queen":
        offsets = QUEEN
    else:
        raise ValueError(f"neigh must be 'rook' or 'queen', got {neigh!r}")

    bx = np.asarray(bx, dtype=np.int64).ravel()
    by = np.asarray(by, dtype=np.int64).ravel()
    n = bx.shape[0]
    if by.shape[0] != n:
        raise ValueError(f"bx/by length mismatch {n} vs {by.shape[0]}")
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    # shift to >= 1 so that a -1 offset never goes negative, then encode
    # (bx, by) -> injective integer key
    bx = bx - bx.min() + 1
    by = by - by.min() + 1
    W = int(by.max()) + 2
    keys = bx * W + by
    if np.unique(keys).shape[0] != n:
        raise ValueError("duplicate lattice positions: obs are not unique "
                         "tiles, so a lattice graph is undefined")
    order = np.argsort(keys)
    skeys = keys[order]

    rows, cols = [], []
    for dx, dy in offsets:
        q = (bx + dx) * W + (by + dy)
        pos = np.searchsorted(skeys, q)
        pos = np.minimum(pos, n - 1)
        hit = skeys[pos] == q
        rows.append(np.nonzero(hit)[0])
        cols.append(order[pos[hit]])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    A = sp.csr_matrix((np.ones(r.shape[0], dtype=np.float32), (r, c)),
                      shape=(n, n))
    A = A.maximum(A.T).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def expression_knn_adj(A, use_rep: str = "X_pca", n_neighbors: int = 10,
                       seed: int = 0, key_added: str = "scc_expr"
                       ) -> sp.csr_matrix:
    """
    Binarised, symmetrised scanpy/UMAP kNN connectivities on obsm[use_rep].
    This is what Stereopy's tl.neighbors (method='umap') produces before it
    binarises in tl.spatial_neighbors.
    """
    import scanpy as sc
    if use_rep not in A.obsm:
        raise KeyError(f"obsm[{use_rep!r}] missing; run per_sample_pca first")
    sc.pp.neighbors(A, n_neighbors=n_neighbors, use_rep=use_rep,
                    random_state=seed, key_added=key_added)
    C = A.obsp[f"{key_added}_connectivities"].tocsr().astype(np.float32)
    C.data[:] = 1.0
    C = C.maximum(C.T).tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()
    return C


def scc_union(expr_adj: sp.spmatrix, spatial_adj: sp.spmatrix) -> sp.csr_matrix:
    """Stereopy's rule: binarise, add, binarise. Zero diagonal."""
    if expr_adj.shape != spatial_adj.shape:
        raise ValueError(f"shape mismatch {expr_adj.shape} vs "
                         f"{spatial_adj.shape}")
    E = expr_adj.tocsr().astype(np.float32).copy()
    E.data[:] = 1.0
    S = spatial_adj.tocsr().astype(np.float32).copy()
    S.data[:] = 1.0
    U = (E + S).tocsr()
    U.data[:] = 1.0
    U.setdiag(0.0)
    U.eliminate_zeros()
    return U


def per_sample_pca(A, n_pcs: int = 50, scale_max: Optional[float] = 10.0,
                   seed: int = 0, key: str = "X_pca") -> Dict[str, float]:
    """
    Stereopy-default embedding for the expression kNN, computed on a COPY so
    A.X (which STORM fits on) is never densified or centred in place:

        scale(zero_center=True, max_value=scale_max) -> pca(n_pcs, arpack)

    A.X must already be the log-normalised HVG matrix. Writes obsm[key].
    """
    import anndata as ad
    import scanpy as sc
    n_comps = int(min(n_pcs, A.n_obs - 1, A.n_vars - 1))
    if n_comps < 2:
        raise ValueError(f"too few obs/vars for PCA: {A.shape}")
    T = ad.AnnData(X=A.X.copy())
    sc.pp.scale(T, zero_center=True, max_value=scale_max)
    sc.tl.pca(T, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    A.obsm[key] = np.asarray(T.obsm["X_pca"], dtype=np.float32)
    evr = np.asarray(T.uns["pca"]["variance_ratio"], dtype=float)
    del T
    return dict(n_comps=n_comps, evr_total=float(evr.sum()),
                evr_pc1=float(evr[0]))


def degree_stats(A: sp.spmatrix) -> Dict[str, float]:
    d = np.asarray(A.sum(1)).ravel()
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                min=float(d.min()) if d.size else 0.0,
                max=float(d.max()) if d.size else 0.0,
                frac_isolated=float((d == 0).mean()) if d.size else 0.0)


def lattice_from_adata(A, bin_size: Optional[int] = None,
                       neigh: str = "rook") -> sp.csr_matrix:
    """
    Lattice adjacency from an AnnData: obs['bx'/'by'] when present, otherwise
    recovered from obsm['spatial'] as round((coord - bin/2) / bin), which is
    the inverse of the convention in build_bin110.py (tile centre = bx*bin +
    bin//2). bin_size falls back to uns['bin_size'].
    """
    if "bx" in A.obs and "by" in A.obs:
        return lattice_adj(A.obs["bx"].to_numpy(), A.obs["by"].to_numpy(), neigh)
    if "spatial" not in A.obsm:
        raise KeyError("neither obs['bx','by'] nor obsm['spatial'] present")
    b = bin_size or int(A.uns.get("bin_size", 0))
    if not b:
        raise ValueError("bin_size unknown: pass it or set uns['bin_size']")
    xy = np.asarray(A.obsm["spatial"], dtype=np.float64)
    bx = np.round((xy[:, 0] - b / 2.0) / b).astype(np.int64)
    by = np.round((xy[:, 1] - b / 2.0) / b).astype(np.int64)
    return lattice_adj(bx, by, neigh)


def embedding_lattice_graph(Z, section, x, y, bin_size: int = 110, k: int = 10,
                            neigh: str = "rook"):
    """
    The clustering graph for STORM embeddings: EXACT kNN (k neighbours, kd-tree,
    Euclidean) on the embedding Z, binarised and symmetrised, unioned with the
    rook tile lattice within each section. This is the graph Gong et al.'s
    Stereopy SCC clusters on, applied to the STORM coordinates instead of PCA.

    Exact kNN rather than scanpy's approximate index: on a 336k-bin run the
    approximate graph differed between nodes/thread counts enough to move
    Leiden at res 0.5 from 5 to 7 domains. A kd-tree in R=10-20 dims is cheap
    and bit-reproducible.

    Returns (adjacency csr, dict of mean degrees).
    """
    from sklearn.neighbors import NearestNeighbors
    Z = np.ascontiguousarray(np.asarray(Z, dtype=np.float32))
    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(Z)
    _, idx = nn.kneighbors(Z)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    E = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    E = E.maximum(E.T)
    sec_idx = pd.factorize(pd.Series(section).astype(str))[0]
    bx = np.round((np.asarray(x, float) - bin_size / 2.0) / bin_size).astype(np.int64)
    by = np.round((np.asarray(y, float) - bin_size / 2.0) / bin_size).astype(np.int64)
    bx = bx - bx.min(); by = by - by.min()
    S = lattice_adj(bx + sec_idx * (bx.max() + 3), by, neigh=neigh)
    U = scc_union(E, S)
    deg = dict(expr=float(np.asarray(E.sum(1)).mean()),
               lattice=float(np.asarray(S.sum(1)).mean()),
               union=float(np.asarray(U.sum(1)).mean()), k=k, neigh=neigh,
               exact_knn=True)
    return U, deg
