from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wasserstein_distance
from concurrent.futures import ProcessPoolExecutor

from PSL import PSL


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def _norm(names, case):
    if case == "upper":
        return [str(g).upper() for g in names]
    if case == "lower":
        return [str(g).lower() for g in names]
    return [str(g) for g in names]


def _rank(x):
    order = x.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x))
    return r


def build_gcn(genes, ppi_path, expr_df, logfc,
              score_threshold=700, case="upper",
              corr="pearson", eps=1e-2, drop_isolates=True,
              weight_transform=None):
    """
    Parameters
    ----------
    genes           : ordered iterable of gene names.
    ppi_path        : CSV with columns gene1, gene2, combined_score (STRING-style).
    expr_df         : DataFrame, index = gene, columns = the cells of interest.
                      Rows are expression vectors over which co-expression is taken.
    logfc           : dict or Series, gene -> signed LogFC (the sheaf charge).
    score_threshold : keep STRING edges with combined_score >= this.
    case            : {"upper","lower","keep"} name normalization applied to ALL
                      three sources (PPI, expr index, logfc keys).
    corr            : {"pearson","spearman"} co-expression measure.
    eps             : weight floor. Also the top-radius margin. Sets the maximum
                      restriction magnitude (1/eps); tune deliberately.
    drop_isolates   : remove edgeless nodes (their leave-one-out score is 0).
    weight_transform: None (default) keeps weight = 1 - |rho| as measured.
                      "rank" replaces each weight by the QUANTILE RANK of its
                      |rho| among all edges, rescaled to [eps, 1]. Use it when
                      the measured |rho| are compressed into a narrow band -- as
                      single-nucleus co-expression is, where dropout attenuates
                      every correlation toward 0 and 1 - |rho| lands in
                      [0.99, 1.0] for nearly every edge. A filtration over that
                      band carries no scale information and the sheaf degenerates
                      to a plain weighted graph Laplacian. The rank transform
                      preserves the ORDERING of co-expression strength, which is
                      the part the attenuation leaves intact, and restores the
                      dynamic range the restriction maps (1/F) need.

    Returns
    -------
    nx.Graph with named nodes carrying 'logFCs' and edges carrying 'weight'.
    """
    gmatch = _norm(genes, case)

    # Expression + charges keyed by the SAME normalized names used to match PPI.
    expr, lf = {}, {}
    for g_raw, g in zip(genes, gmatch):
        if g_raw in expr_df.index:
            expr[g] = np.asarray(expr_df.loc[g_raw], dtype=float)
        v = logfc.get(g_raw) if hasattr(logfc, "get") else None
        lf[g] = 0.0 if v is None else float(v)

    if not any(abs(v) > 0 for v in lf.values()):
        warnings.warn(
            "All charges are 0 after name-matching -- check that logfc keys share "
            "the same case as `genes`. A zero-charge sheaf gives degenerate Laplacians."
        )

    G = nx.Graph()
    for g in gmatch:                       # every gene is a node, in given order
        G.add_node(g, logFCs=lf.get(g, 0.0))

    df = pd.read_csv(ppi_path)
    df["_g1"] = _norm(df["gene1"], case)
    df["_g2"] = _norm(df["gene2"], case)
    have = set(expr)                       # in gene set AND has expression
    keep = (
        (df["combined_score"] >= score_threshold)
        & df["_g1"].isin(have)
        & df["_g2"].isin(have)
        & (df["_g1"] != df["_g2"])
    )

    for a, b in (df.loc[keep, ["_g1", "_g2"]]
                   .drop_duplicates()
                   .itertuples(index=False, name=None)):
        xa, xb = expr[a], expr[b]
        if corr == "spearman":
            xa, xb = _rank(xa), _rank(xb)
        rho = np.corrcoef(xa, xb)[0, 1]
        rho = 0.0 if np.isnan(rho) else rho            # flat gene -> no coupling
        w = min(max(1.0 - abs(rho), eps), 1.0)         # strong coexpr -> small dist
        G.add_edge(a, b, weight=w)

    if weight_transform == "rank":
        es = list(G.edges())
        if es:
            rho = np.array([1.0 - G[u][v]["weight"] for u, v in es], dtype=float)
            order = rho.argsort()
            r = np.empty(len(rho), dtype=float)
            r[order] = np.arange(len(rho), dtype=float)
            r /= max(len(rho) - 1, 1)          # 0 = weakest |rho|, 1 = strongest
            for (u, v), rr in zip(es, r):
                G[u][v]["weight"] = float(min(max(1.0 - rr * (1.0 - eps), eps), 1.0))
    elif weight_transform is not None:
        raise ValueError(f"unknown weight_transform {weight_transform!r}")

    if drop_isolates:
        G.remove_nodes_from(list(nx.isolates(G)))
    return G


def make_radii(G, n_scales=20, eps=1e-2):
    """Quantile grid over the actual edge-weight distribution, so scales land
    where edges actually are. Top radius nudged past max weight so the strict
    `>=` break in build_simplicial_pair still includes the weakest edge."""
    w = np.array([d["weight"] for *_, d in G.edges(data=True)], dtype=float)
    if w.size == 0:
        raise ValueError("Graph has no edges; nothing to filter over.")
    radii = np.quantile(w, np.linspace(0.0, 1.0, n_scales))
    radii = np.unique(radii)               # guard against duplicate quantiles
    radii[-1] += eps
    return radii


# --------------------------------------------------------------------------- #
# Node remapping (carries node + edge attributes, unlike the original)
# --------------------------------------------------------------------------- #
def map_graph_nodes(G):
    """Relabel nodes to 0..n-1 while preserving node attrs (logFCs) and edge
    attrs (weight). Charge alignment downstream depends on this."""
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    H = nx.Graph()
    for node, idx in node_mapping.items():
        H.add_node(idx, **G.nodes[node])
    for u, v, data in G.edges(data=True):
        H.add_edge(node_mapping[u], node_mapping[v], **data)
    return H, node_mapping


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #
def _spectra(G, radii):
    """Run the plain sheaf Laplacian and return (l0_eigs, l1_eigs): lists of
    real eigenvalue arrays, one per scale. Charges are read AFTER remapping,
    in mapped order, so they can never desync from the adjacency."""
    adj, _ = map_graph_nodes(G)
    n = adj.number_of_nodes()
    charges = np.array([adj.nodes[i].get("logFCs", 0.0) for i in range(n)],
                       dtype=float)

    psl = PSL(adj, charges, radii)
    psl.build_filtration()
    psl.build_simplicial_pair()
    psl.build_matrices()
    l0s, l1s = psl.psl_0(), psl.psl_1()

    e0 = [np.linalg.eigvalsh(L) if L.size else np.empty(0) for L in l0s]
    e1 = [np.linalg.eigvalsh(L) if L.size else np.empty(0) for L in l1s]
    return e0, e1


def _w1(a, b):
    """W1 between two eigenvalue spectra. Empty spectra (e.g. no triangles yet,
    or a block that vanished on removal) are represented by a single 0, so the
    distance measures spectral mass appearing/disappearing rather than crashing.
    Natively handles the n vs n-1 eigenvalue-count mismatch from node removal."""
    a = a if a.size else np.zeros(1)
    b = b if b.size else np.zeros(1)
    return float(wasserstein_distance(a, b))


# --------------------------------------------------------------------------- #
# Parallel leave-one-out scoring
# --------------------------------------------------------------------------- #
_W = {}   # per-process cache populated by the pool initializer


def _init_worker(G_full, radii, base_e0, base_e1):
    _W["G"] = G_full
    _W["radii"] = radii
    _W["e0"] = base_e0
    _W["e1"] = base_e1


def _score_gene(gene):
    G = _W["G"].copy()
    G.remove_node(gene)
    e0, e1 = _spectra(G, _W["radii"])
    b0, b1 = _W["e0"], _W["e1"]
    # per scale: shift in L0 spectrum + shift in L1 spectrum
    return [_w1(b0[s], e0[s]) + _w1(b1[s], e1[s]) for s in range(len(b0))]


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def rank_genes(scores, gene_names):
    """Aggregate each gene's significance-vs-scale curve into one number (mean
    across scales = area under the curve) and return a descending total order."""
    scores = np.asarray(scores, dtype=float)          # (num_genes, num_scales)
    agg = scores.mean(axis=1)
    order = np.argsort(-agg)
    return [(gene_names[i], float(agg[i])) for i in order]


def intersect_top_genes(scores, gene_names, top_n):
    """Diagnostic (not the primary ranker): genes in the top_n at EVERY scale --
    i.e. consistently central across all resolutions. Can be small or empty."""
    scores = np.asarray(scores, dtype=float)
    per_scale = [set(np.argsort(-scores[:, s])[:top_n]) for s in range(scores.shape[1])]
    idx = sorted(set.intersection(*per_scale)) if per_scale else []
    return [gene_names[i] for i in idx]


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def run(genes, ppi_path, expr_df, logfc,
        score_threshold=700, case="upper", corr="pearson",
        eps=1e-2, n_scales=20, max_workers=None, drop_isolates=True,
        weight_transform=None):
    """
    Build the network, compute the baseline sheaf spectra once, then score every
    node by its leave-one-out spectral shift and rank.

    Returns
    -------
    dict with:
        graph    : the constructed nx.Graph (named nodes)
        radii    : the filtration grid actually used
        genes    : node order used for scoring (aligns with `scores` rows)
        scores   : (num_genes, num_scales) array of per-scale W1 shifts
        ranking  : list of (gene, mean_shift), descending
        stable   : intersect diagnostic (top-10 at every scale)

    NOTE: on spawn platforms (Windows/macOS) call this under
    `if __name__ == "__main__":`.
    """
    G = build_gcn(genes, ppi_path, expr_df, logfc,
                  score_threshold=score_threshold, case=case,
                  corr=corr, eps=eps, drop_isolates=drop_isolates,
                  weight_transform=weight_transform)
    if G.number_of_edges() == 0:
        raise ValueError("Constructed graph has no edges -- check thresholds / "
                         "name matching / expression coverage.")

    radii = make_radii(G, n_scales=n_scales, eps=eps)
    gene_order = list(G.nodes())

    # Baseline computed ONCE; every gene is compared against this single reference.
    base_e0, base_e1 = _spectra(G, radii)

    try:
        from tqdm import tqdm
    except ImportError:                     # tqdm optional
        def tqdm(x, **k):
            return x

    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_init_worker,
                             initargs=(G, radii, base_e0, base_e1)) as ex:
        scores = list(tqdm(
            ex.map(_score_gene, gene_order),
            total=len(gene_order),
            desc="LogFC sheaf Laplacian spectral shift",
            unit="gene",
        ))

    scores = np.asarray(scores, dtype=float)
    ranking = rank_genes(scores, gene_order)
    stable = intersect_top_genes(scores, gene_order, top_n=min(10, len(gene_order)))

    return {
        "graph": G,
        "radii": radii,
        "genes": gene_order,
        "scores": scores,
        "ranking": ranking,
        "stable": stable,
    }
