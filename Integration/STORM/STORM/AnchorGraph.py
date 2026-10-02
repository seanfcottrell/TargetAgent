#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree, distance

try:
    import ot
    _HAS_POT = True
except ImportError:
    _HAS_POT = False


# ===========================================================================
# config
# ===========================================================================

@dataclass
class AnchorGraphConfig:
    # --- degree targets (these ARE the edge weights) ---
    k_within: float = 10.0          # mean within-slice spatial degree
    k_cross: float = 5.0            # mean cross-slice degree -> lambda_cross=0.5

    # --- FGW ---
    alpha: float = 0.5              # weight on the Gromov (structure) term
    epsilon: float = 0.02           # entropic reg, on the NORMALISED cost
    max_outer: int = 20             # MM iterations of the linearisation
    max_inner: int = 200            # unbalanced Sinkhorn iterations
    tol: float = 1e-7

    # --- tau derivation ---
    tau_method: str = "otsu"        # otsu | gmm | fixed
    tau_fixed_within: float = 20.0  # only if tau_method == "fixed"
    tau_fixed_cross: float = 2.0
    tau_within_floor: float = 5.0   # tau_within is tight: never below this
    tau_scale_floor: float = 0.25   # floor as a fraction of median row-min cost
    tau_sweep: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)  # plateau check

    # --- coupling resolution ---
    coupling_level: str = "niche"   # niche | spot
    n_niches_per_sample: int = 1000
    niche_k_spatial: int = 6
    lift_local_k: int = 5
    max_dense_gb: float = 16.0

    # --- feature space for the linear cost ---
    emb_key: str = "X_pca_anchor"

    # --- cross-condition topology ---
    # "anchor_only": only the anchor pair carries cross-condition edges.
    # "compose"    : leaves also get 2-hop composed cross-condition edges.
    # Default anchor_only; let the spectral pre-flight decide.
    cross_mode: str = "anchor_only"

    normalize_laplacian: bool = True
    seed: int = 0
    verbose: bool = True


# ===========================================================================
# 1. anchor selection  (TECHNICAL criteria only)
# ===========================================================================


# ===========================================================================
# 2. shared feature space  (anchors fit, leaves projected)
# ===========================================================================


# ===========================================================================
# 3. within-slice radius solved per slice for a target mean degree
# ===========================================================================

def solve_radius_for_degree(coords: np.ndarray, target_k: float,
                            tol: float = 0.05, max_iter: int = 40
                            ) -> Tuple[float, float]:
    """
    Bisect the radius so the mean within-slice degree hits target_k.

    Deterministic root-find, not a judgement -- spot density differs by
    platform and by tissue region, so a single global radius yields wildly
    different neighbourhood sizes across slices.

    Returns (radius, realised_mean_degree).
    """
    n = coords.shape[0]
    if n <= 1:
        return 0.0, 0.0
    kdt = cKDTree(coords)
    k_probe = int(min(n - 1, max(8, np.ceil(target_k) * 3)))
    d, _ = kdt.query(coords, k=k_probe + 1)
    lo, hi = 0.0, float(np.percentile(d[:, -1], 99)) * 1.5

    def mean_deg(r):
        return float(np.mean(kdt.query_ball_point(coords, r,
                                                  return_length=True) - 1))

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m = mean_deg(mid)
        if abs(m - target_k) <= tol * target_k:
            return mid, m
        if m < target_k:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    return r, mean_deg(r)


def _radius_adj(coords: np.ndarray, radius: float) -> sp.csr_matrix:
    n = coords.shape[0]
    if n <= 1:
        return sp.csr_matrix((n, n))
    kdt = cKDTree(coords)
    D = kdt.sparse_distance_matrix(kdt, max_distance=radius,
                                   output_type="coo_matrix")
    keep = D.row != D.col
    A = sp.coo_matrix((np.ones(int(keep.sum())), (D.row[keep], D.col[keep])),
                      shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


# ===========================================================================
# 4. tau derivation from the row-wise linear-cost distribution
# ===========================================================================

def _otsu_threshold(x: np.ndarray, nbins: int = 128) -> Tuple[float, float]:
    """
    Otsu on a 1-D sample. Returns (threshold, separability in [0,1]).

    nbins ADAPTS to sample size. A fixed 128 bins over a few hundred niches
    leaves most bins empty, and Otsu on a sparse histogram picks thresholds in
    empty regions -- which produced tau ~1e-3 against a cost whose median
    best-match is 1.0 by construction, i.e. a coupling that transports nothing.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float(np.median(x)) if x.size else 0.0, 0.0
    nbins = int(min(nbins, max(8, x.size // 8)))
    hist, edges = np.histogram(x, bins=nbins)
    p = hist.astype(float) / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = w0 * w1
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_b = (mu_t * w0 - mu) ** 2 / np.where(denom > 0, denom, np.nan)
    # A degenerate distribution (all mass in one bin, or zero variance) makes
    # every between-class variance NaN. That is not an error -- it means no
    # threshold exists, i.e. tau is UNIDENTIFIED for this pair. Report it as
    # separability 0 and let the caller flag it, rather than throwing.
    total_var = float(np.var(x))
    if not np.isfinite(sigma_b).any() or total_var <= 0:
        return float(np.median(x)), 0.0
    i = int(np.nanargmax(sigma_b))
    sep = float(np.nanmax(sigma_b) / total_var)
    return float(centers[i]), min(max(sep, 0.0), 1.0)


def derive_tau(M: np.ndarray, kind: str, cfg: AnchorGraphConfig
               ) -> Tuple[float, Dict[str, Any]]:
    """
    tau from the distribution of each row's BEST-MATCH feature cost.

    Rationale: the linear term has interpretable units (expression
    dissimilarity), so "a spot drops mass when nothing in the other slice
    looks similar enough" is a defensible statement. The Gromov term's units
    are squared metric discrepancy and are not something to threshold
    biologically. Using the linear term only errs toward MATCHING, which is
    the conservative direction -- the failure we most want to avoid is
    fabricating unmatched-ness that is not there.

    Row-wise (one value per spot), not over all n^2 entries: the latter is
    dominated by the mass of irrelevant pairs and will not separate.

    Per-pair, not global: pairs genuinely differ in correspondence quality,
    and per-pair thresholding is what lets the within/cross tau asymmetry
    EMERGE from the data instead of being asserted.
    """
    if cfg.tau_method == "fixed":
        tau = (cfg.tau_fixed_within if kind == "within" else cfg.tau_fixed_cross)
        return float(tau), dict(method="fixed", identified=True, separability=None)

    row_min = M.min(axis=1)
    thr, sep = _otsu_threshold(row_min)

    # tau is a KL weight against a cost; the threshold sits where transport
    # stops being cheap, so tau ~ that cost level. Costs are normalised to
    # median(row_min) = 1 upstream, so thr is already dimensionless.
    # M is normalised so median(row_min) == 1, so a usable tau is O(1). A
    # threshold far below that means Otsu landed in an empty region of the
    # histogram, not that mass genuinely should not move -- clamp to a
    # fraction of the typical best-match cost rather than to an absolute
    # epsilon, and let `identified` carry the doubt.
    tau = float(max(thr, cfg.tau_scale_floor))
    if kind == "within":
        tau = max(tau, cfg.tau_within_floor)

    identified = sep >= 0.15   # unimodal -> no threshold exists
    return tau, dict(method=cfg.tau_method, identified=bool(identified),
                     separability=float(sep), raw_threshold=float(thr))


# ===========================================================================
# 5. unbalanced FGW
# ===========================================================================

def _check_dense(ni: int, nj: int, cfg: AnchorGraphConfig, what: str) -> None:
    gb = (ni * nj + 2 * ni * ni + 2 * nj * nj) * 8 / 1e9
    if gb > cfg.max_dense_gb:
        raise MemoryError(
            f"{what}: dense unbalanced FGW for a {ni} x {nj} block needs "
            f"~{gb:.0f} GB, over max_dense_gb={cfg.max_dense_gb}. "
            "Use cfg.coupling_level='niche', subsample, or raise the cap."
        )


def unbalanced_fgw(M: np.ndarray, C1: np.ndarray, C2: np.ndarray,
                   a: np.ndarray, b: np.ndarray,
                   tau_a: float, tau_b: float,
                   cfg: AnchorGraphConfig) -> Tuple[np.ndarray, Dict]:
    """
    Unbalanced fused Gromov-Wasserstein by majorisation-minimisation.

        min_G  (1-alpha)<M,G> + alpha * sum_ijkl (C1_ik - C2_jl)^2 G_ij G_kl
               + eps H(G) + tau_a KL(G1 || a) + tau_b KL(G'1 || b)

    Each outer step linearises the quadratic term at the current G and solves
    the resulting unbalanced Sinkhorn problem. Because the marginal penalty is
    a plain KL against a and b, unmatched mass is directly readable as
    a - G1 -- which is the whole point. POT's fused_unbalanced_gromov_
    wasserstein optimises the FUGW lower bound with a quadratic marginal
    penalty instead, from which per-spot unmatched mass is not recoverable
    in the same way.

    For square loss the gradient of the quadratic term at G is

        2 * [ (C1^2 @ g_r)_i + (C2^2 @ g_c)_j - 2 (C1 @ G @ C2)_ij ]

    with g_r = G1, g_c = G'1 -- recomputed each iteration, so it stays
    correct as the marginals drift away from a and b.
    """
    if not _HAS_POT:
        raise ImportError("POT required: pip install pot")

    C1sq, C2sq = C1 ** 2, C2 ** 2
    G = np.outer(a, b) / max(a.sum() * b.sum(), 1e-30) * min(a.sum(), b.sum())
    prev = None
    n_outer = 0
    for n_outer in range(1, cfg.max_outer + 1):
        g_r = G.sum(axis=1)
        g_c = G.sum(axis=0)
        quad = (C1sq @ g_r)[:, None] + (C2sq @ g_c)[None, :] - 2.0 * (C1 @ G @ C2)
        Gc = (1.0 - cfg.alpha) * M + cfg.alpha * 2.0 * quad
        Gc = Gc - Gc.min()                      # keep the kernel well scaled
        Gc = Gc / (np.median(Gc[Gc > 0]) or 1.0)

        G = ot.unbalanced.sinkhorn_unbalanced(
            a, b, Gc, reg=cfg.epsilon, reg_m=[tau_a, tau_b],
            numItermax=cfg.max_inner, stopThr=cfg.tol,
        )
        G = np.asarray(G)
        if not np.isfinite(G).all():
            raise FloatingPointError(
                "unbalanced FGW produced non-finite mass; epsilon is likely "
                "too small for this cost scale."
            )
        if prev is not None:
            delta = np.abs(G - prev).sum() / max(G.sum(), 1e-30)
            if delta < cfg.tol * 100:
                break
        prev = G.copy()

    info = dict(outer_iters=int(n_outer),
                converged=bool(n_outer < cfg.max_outer))
    return G, info


# ===========================================================================
# 6. per-pair coupling + diagnostics
# ===========================================================================


def spatial_coherence(unmatched: np.ndarray, coords: np.ndarray,
                      k: int = 8) -> float:
    """
    Moran's-I-like coherence of unmatched mass over the spatial kNN graph.

    The acceptance test that needs no prior at all: unmatched mass should be
    CONTIGUOUS (a lesion, a fold, a fragment). Salt-and-pepper unmatched mass
    means tau is too low and mass is being dropped on noise, whatever the
    derivation said.
    """
    n = coords.shape[0]
    if n < k + 1:
        return float("nan")
    kdt = cKDTree(coords)
    _, idx = kdt.query(coords, k=k + 1)
    idx = idx[:, 1:]
    x = unmatched - unmatched.mean()
    denom = float((x ** 2).sum())
    if denom <= 0:
        return float("nan")
    num = float((x[:, None] * x[idx]).sum())
    return num / (denom * k)


# ===========================================================================
# 7. niche aggregation (spot-level OT does not scale)
# ===========================================================================

def build_niches(coords: np.ndarray, emb: np.ndarray, n_niches: int,
                 k_spatial: int = 6, seed: int = 0) -> np.ndarray:
    """Spatially-constrained agglomeration into contiguous niches."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    n = coords.shape[0]
    n_niches = int(min(max(n_niches, 1), n))
    if n_niches >= n:
        return np.arange(n)
    conn = kneighbors_graph(coords, n_neighbors=min(k_spatial, n - 1),
                            include_self=False)
    Z = np.hstack([emb, coords / (np.abs(coords).max() or 1.0)])
    return AgglomerativeClustering(n_clusters=n_niches, connectivity=conn,
                                   linkage="ward").fit_predict(Z)


# ===========================================================================
# 8. pooled top-k and degree normalisation  (this is lambda_cross)
# ===========================================================================

def pooled_topk(blocks: Dict[int, sp.csr_matrix], k: int) -> Dict[int, sp.csr_matrix]:
    """
    Budget cross-sample degree GLOBALLY per spot, not per partner slice.

    k-per-partner would give a spot in a chain 2k cross edges and a spot in a
    star 1k, so the effective lambda_cross would vary by topology and by
    sample. Pooling all candidate partners across every coupled slice and
    taking the top k overall keeps mean cross degree at k regardless -- and
    lets the OT decide the allocation: a spot with one strong counterpart and
    nothing elsewhere puts all k in one slice, which is correct.
    """
    if not blocks:
        return {}
    keys = sorted(blocks)
    n_rows = blocks[keys[0]].shape[0]
    cat = sp.hstack([blocks[j] for j in keys], format="csr")
    widths = [blocks[j].shape[1] for j in keys]
    bounds = np.cumsum([0] + widths)

    cat.sort_indices()
    R, C, V = [], [], []
    for r in range(n_rows):
        s0, s1 = cat.indptr[r], cat.indptr[r + 1]
        if s1 <= s0:
            continue
        d, c = cat.data[s0:s1], cat.indices[s0:s1]
        if d.size > k:
            sel = np.argpartition(-d, k - 1)[:k]
            d, c = d[sel], c[sel]
        R.append(np.full(d.size, r)); C.append(c); V.append(d)
    if not R:
        return {j: sp.csr_matrix(blocks[j].shape) for j in keys}
    R, C, V = np.concatenate(R), np.concatenate(C), np.concatenate(V)

    out = {}
    for bi, j in enumerate(keys):
        m = (C >= bounds[bi]) & (C < bounds[bi + 1])
        out[j] = sp.coo_matrix((V[m], (R[m], C[m] - bounds[bi])),
                               shape=blocks[j].shape).tocsr()
    return out


def rescale_to_degree(A: sp.csr_matrix, target_row_sum: float) -> sp.csr_matrix:
    """
    Rescale a coupling block so its mean row-sum equals target_row_sum.

    THIS is the step that makes cross-slice edges matter at all. Gamma entries
    are O(1/n^2); a radius graph's are O(1). Multiplying Gamma by a weight and
    dropping it into the same Laplacian -- which is what
    `spec.weight * P` does in GraphSchema.build_Ls -- leaves cross-slice edges
    contributing essentially nothing, and any subsequent weight sweep starts
    from an origin so far off that you conclude integration strength does not
    matter.
    """
    if A.nnz == 0:
        return A
    cur = float(np.asarray(A.sum(axis=1)).ravel().mean())
    if cur <= 0:
        return A
    return (A * (target_row_sum / cur)).tocsr()


def balance_cross_degrees(blocks: Dict[Tuple[int, int], sp.csr_matrix],
                          ns: Sequence[int], target: float,
                          iters: int = 12) -> Dict[Tuple[int, int], sp.csr_matrix]:
    """
    Make every slice's mean cross degree equal `target`, symmetrically.

    Needed because the anchor topology is a star: an anchor is the target of
    every leaf's coupling, so it accumulates in-edges at a rate proportional
    to its partner count while a leaf keeps exactly one budget. Left alone,
    the effective lambda_cross would be ~3x higher at the anchors than at the
    leaves -- i.e. the anchors would be smoothed much harder than the tissue
    they are meant to be a reference for.

    Scaling block (i,j) by sqrt(f_i f_j) keeps the block symmetric while
    moving both endpoints toward the target; a few passes converge.
    """
    B = {k: v.copy() for k, v in blocks.items()}
    for _ in range(iters):
        deg = np.zeros(len(ns))
        for (i, j), A in B.items():
            deg[i] += float(np.asarray(A.sum(axis=1)).ravel().sum()) / ns[i]
            deg[j] += float(np.asarray(A.sum(axis=0)).ravel().sum()) / ns[j]
        f = np.where(deg > 0, target / np.maximum(deg, 1e-30), 1.0)
        if np.allclose(f, 1.0, rtol=1e-3):
            break
        for (i, j) in B:
            B[(i, j)] = (B[(i, j)] * float(np.sqrt(f[i] * f[j]))).tocsr()
    return B


# ===========================================================================
# 9. assembly + pre-flight
# ===========================================================================

def _coords_of(ad) -> Optional[np.ndarray]:
    if "spatial" in ad.obsm:
        return np.asarray(ad.obsm["spatial"])[:, :2].astype(float)
    if {"x", "y"}.issubset(ad.obs.columns):
        return np.c_[ad.obs["x"].to_numpy(), ad.obs["y"].to_numpy()].astype(float)
    return None


def _sym_normalize(L: sp.spmatrix) -> sp.csr_matrix:
    d = np.asarray(L.diagonal(), float)
    dinv = np.zeros_like(d)
    nz = d > 0
    dinv[nz] = 1.0 / np.sqrt(d[nz])
    D = sp.diags(dinv)
    return (D @ L @ D).tocsr()

