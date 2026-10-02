#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree, distance

from AnchorGraph import (
    AnchorGraphConfig, derive_tau, unbalanced_fgw, spatial_coherence,
    solve_radius_for_degree, _radius_adj, build_niches, pooled_topk,
    rescale_to_degree, balance_cross_degrees, _sym_normalize, _coords_of,
    _check_dense,
)
from SCCGraph import (
    lattice_from_adata, expression_knn_adj, scc_union, degree_stats,
)


# ===========================================================================
# config + edge type
# ===========================================================================

@dataclass
class ChainConfig(AnchorGraphConfig):
    stage_key: str = "stage"
    lot_key: str = "chip_lot"
    section_key: str = "section"
    stage_order: Tuple[str, ...] = ("NA", "moderate", "severe")
    # obs column with per-bin nuclear counts. Used ONLY as a diagnostic.
    density_key: str = "n_nuclei"
    require_lot_matched_bridges: bool = True

    # --- within-sample graph -------------------------------------------
    # "scc"    : Stereopy's spatially-constrained-clustering graph -- the
    #            binarised UNION of an expression kNN (expr_k neighbours on
    #            obsm[expr_key], per sample) and the tile lattice (rook = 4
    #            neighbours, which is what squidpy's grid mode with n_neighs=6
    #            reduces to on a square lattice). This is the graph Gong et al.
    #            used to call cortical layers on bin110; here it is the
    #            within-sample block, and the OT couplings supply the
    #            cross-sample block.
    # "radius" : the original radius graph, bisected to mean degree k_within.
    within_graph: str = "scc"
    expr_key: str = "X_pca"
    expr_k: int = 10
    spatial_neigh: str = "rook"
    # Cross degree as a FRACTION of the realised mean within degree. The SCC
    # union has degree ~15-20 rather than k_within=10, so a fixed k_cross=5
    # would silently halve the effective lambda_cross. None -> use k_cross.
    lambda_cross: Optional[float] = 0.5


@dataclass
class Edge:
    src: str
    dst: str
    kind: str                  # 'within' | 'cross'
    lot_matched: bool = True
    note: str = ""


# ===========================================================================
# 1. topology derivation
# ===========================================================================

def sample_table(adatas: Sequence, cfg: ChainConfig) -> pd.DataFrame:
    rows = []
    for k, A in enumerate(adatas):
        o = A.obs
        c = np.asarray(o["n_counts"] if "n_counts" in o
                       else A.X.sum(1)).ravel()
        g = np.asarray(o["n_genes"] if "n_genes" in o
                       else (A.X > 0).sum(1)).ravel()
        rows.append(dict(
            idx=k,
            section=str(o[cfg.section_key].iloc[0]),
            stage=str(o[cfg.stage_key].iloc[0]),
            lot=str(o[cfg.lot_key].iloc[0]),
            n_bins=int(A.n_obs),
            med_counts=float(np.median(c)),
            med_genes=float(np.median(g)),
        ))
    df = pd.DataFrame(rows)
    # technical quality only -- never coupling behaviour, never pathology
    df["quality"] = 0.0
    for st, sub in df.groupby("stage"):
        q = (sub.med_counts.rank(pct=True) + sub.med_genes.rank(pct=True)
             + sub.n_bins.rank(pct=True)) / 3.0
        df.loc[sub.index, "quality"] = q
    return df


def derive_chain(df: pd.DataFrame, cfg: ChainConfig
                 ) -> Tuple[List[Edge], Dict[str, List[str]], List[str]]:
    """
    Build the edge list from stages and lots.

    For each ADJACENT stage pair, pick the bridge from the lot shared by both
    stages, breaking ties on technical quality. Those endpoints become hubs.
    Remaining slices attach to a hub of their own stage, preferring a same-lot
    hub. Stages that end up with two hubs get a within-stage edge joining them.

    Returns (edges, hubs_per_stage, warnings).
    """
    stages = [s for s in cfg.stage_order if s in set(df.stage)]
    if len(stages) < 2:
        raise ValueError(f"need >=2 stages present, got {stages}")

    warns: List[str] = []
    hubs: Dict[str, List[str]] = {s: [] for s in stages}
    edges: List[Edge] = []

    # --- cross-stage bridges, one per adjacent pair ---------------------
    for a, b in zip(stages[:-1], stages[1:]):
        A, B = df[df.stage == a], df[df.stage == b]
        shared = sorted(set(A.lot) & set(B.lot))
        if shared:
            best, score = None, -np.inf
            for lot in shared:
                for _, ra in A[A.lot == lot].iterrows():
                    for _, rb in B[B.lot == lot].iterrows():
                        s = ra.quality + rb.quality
                        if s > score:
                            score, best = s, (ra.section, rb.section, lot)
            src, dst, lot = best
            edges.append(Edge(src, dst, "cross", True, f"lot {lot}"))
        else:
            # no shared lot: fall back to quality, and say so loudly
            ra = A.loc[A.quality.idxmax()]
            rb = B.loc[B.quality.idxmax()]
            src, dst = ra.section, rb.section
            edges.append(Edge(src, dst, "cross", False,
                              f"lots {ra.lot}/{rb.lot} DIFFER"))
            warns.append(
                f"{a}<->{b} bridge {src}<->{dst} spans lots "
                f"{ra.lot}/{rb.lot}: batch is confounded with stage on the "
                f"edge that carries this contrast")
            if cfg.require_lot_matched_bridges:
                warns.append(
                    f"  (require_lot_matched_bridges=True but no lot spans "
                    f"{a} and {b}; proceeding with the confound recorded)")
        for st, sec in ((a, src), (b, dst)):
            if sec not in hubs[st]:
                hubs[st].append(sec)

    # --- join multi-hub stages ------------------------------------------
    for st in stages:
        for h1, h2 in zip(hubs[st][:-1], hubs[st][1:]):
            l1 = df.loc[df.section == h1, "lot"].iloc[0]
            l2 = df.loc[df.section == h2, "lot"].iloc[0]
            edges.append(Edge(h1, h2, "within", l1 == l2,
                              f"joins {st} hubs"))

    # --- leaves ---------------------------------------------------------
    hub_set = {h for hs in hubs.values() for h in hs}
    for _, r in df.iterrows():
        if r.section in hub_set:
            continue
        cand = df[df.section.isin(hubs[r.stage])]
        same = cand[cand.lot == r.lot]
        pick = (same.loc[same.quality.idxmax()] if len(same)
                else cand.loc[cand.quality.idxmax()])
        edges.append(Edge(r.section, pick.section, "within",
                          r.lot == pick.lot,
                          "" if r.lot == pick.lot else
                          f"lots {r.lot}/{pick.lot} differ (within-stage: "
                          f"nuisance, not a confound)"))
    return edges, hubs, warns


# ===========================================================================
# 2. niches from POOLED COUNTS
# ===========================================================================

def niche_pooled(A, cfg: ChainConfig, seed: int = 0):
    """
    Spatially contiguous niches whose profile is POOLED RAW COUNTS.

    The agglomeration itself needs some embedding to define similarity; a
    rough per-sample PCA is fine there because it only has to group
    neighbouring bins. What matters is that the niche PROFILE afterwards is a
    sum of counts, not a mean of embeddings -- summing ~N bins multiplies
    effective depth by N, whereas averaging embeddings cannot recover
    information the PCA already discarded.
    """
    from sklearn.decomposition import PCA
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    coords = np.asarray(A.obsm["spatial"], float)

    cnt = np.asarray(X.sum(1)).ravel()
    Z = X.multiply(1e4 / np.maximum(cnt, 1)[:, None]).tocsr()
    Z.data = np.log1p(Z.data)
    npc = int(min(30, min(Z.shape) - 1))
    rough = PCA(n_components=npc, random_state=seed).fit_transform(
        np.asarray(Z.todense()) if Z.shape[1] < 4000 else
        _svd_dense(Z, npc, seed))

    lab = build_niches(coords, rough, cfg.n_niches_per_sample,
                       cfg.niche_k_spatial, seed=seed)
    uniq = np.unique(lab)
    members = [np.where(lab == u)[0] for u in uniq]

    P = sp.vstack([sp.csr_matrix(X[m].sum(0)) for m in members]).tocsr()
    C = np.vstack([coords[m].mean(0) for m in members])
    mass = np.array([m.size for m in members], float)

    dens = None
    if cfg.density_key in A.obs.columns:
        d = np.asarray(A.obs[cfg.density_key], float)
        dens = np.array([d[m].mean() for m in members])
    return dict(labels=lab, members=members, counts=P, coords=C,
                mass=mass, density=dens)


def _svd_dense(Z, npc, seed):
    from sklearn.decomposition import TruncatedSVD
    return TruncatedSVD(n_components=npc, random_state=seed).fit_transform(Z)


def embed_niches(niches: Dict[str, dict], hub_sections: Sequence[str],
                 n_pcs: int = 30, n_hvg: int = 2000, seed: int = 0):
    """
    One feature space for every cost, fit on the HUB niches and applied to all.

    Fitting on hubs rather than on everything concatenated keeps the space
    fixed regardless of which leaves are present, so per-pair tau thresholds
    stay comparable across edges.
    """
    from sklearn.decomposition import PCA
    secs = list(niches)
    Pall = sp.vstack([niches[s]["counts"] for s in secs]).tocsr()

    # HVG on pooled niche profiles, by dispersion of log-normalised values
    tot = np.asarray(Pall.sum(1)).ravel()
    N = Pall.multiply(1e4 / np.maximum(tot, 1)[:, None]).tocsr()
    N.data = np.log1p(N.data)
    mu = np.asarray(N.mean(0)).ravel()
    m2 = np.asarray(N.multiply(N).mean(0)).ravel()
    var = np.maximum(m2 - mu ** 2, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(mu > 0, var / mu, 0.0)
    keep = np.argsort(-disp)[:min(n_hvg, Pall.shape[1])]

    hub_rows, off, spans = [], 0, {}
    for s in secs:
        n = niches[s]["counts"].shape[0]
        spans[s] = (off, off + n)
        if s in hub_sections:
            hub_rows.append(np.arange(off, off + n))
        off += n
    hub_rows = np.concatenate(hub_rows)

    D = np.asarray(N[:, keep].todense(), dtype=np.float32)
    mu_, sd_ = D[hub_rows].mean(0), np.maximum(D[hub_rows].std(0), 1e-8)
    D = (D - mu_) / sd_
    pca = PCA(n_components=min(n_pcs, D.shape[1] - 1),
              random_state=seed).fit(D[hub_rows])
    E = pca.transform(D)
    for s in secs:
        a, b = spans[s]
        niches[s]["emb"] = E[a:b]
    return float(pca.explained_variance_ratio_.sum()), len(keep)


# ===========================================================================
# 3. couple one edge
# ===========================================================================

def couple_edge(ni: dict, nj: dict, kind: str, cfg: ChainConfig) -> dict:
    Ei, Ej = ni["emb"], nj["emb"]
    _check_dense(len(Ei), len(Ej), cfg, f"{kind} edge")

    M = distance.cdist(Ei, Ej)
    scale = float(np.median(M.min(axis=1))) or 1.0
    M = M / scale
    C1 = distance.cdist(ni["coords"], ni["coords"]); C1 /= (C1.max() or 1)
    C2 = distance.cdist(nj["coords"], nj["coords"]); C2 /= (C2.max() or 1)
    a = ni["mass"] / ni["mass"].sum()
    b = nj["mass"] / nj["mass"].sum()

    tau, tinfo = derive_tau(M, kind, cfg)
    G, fit = unbalanced_fgw(M, C1, C2, a, b, tau, tau, cfg)

    g_r, g_c = G.sum(1), G.sum(0)
    um_i = np.clip(1 - g_r / np.maximum(a, 1e-30), 0, 1)
    um_j = np.clip(1 - g_c / np.maximum(b, 1e-30), 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = G / np.maximum(g_r[:, None], 1e-30)
        H = -np.nansum(np.where(R > 0, R * np.log(R), 0.0), axis=1)
    conc = H / np.log(max(len(Ej), 2))

    # density is NOT in the cost; check post hoc whether matched partners
    # nonetheless agree on cellularity. Disagreement = expression-only
    # matching is pairing tissue of very different composition.
    dens_r = np.nan
    if ni.get("density") is not None and nj.get("density") is not None:
        w = np.maximum(g_r, 1e-30)
        partner = (G @ nj["density"]) / w
        ok = (g_r > 1e-8) & np.isfinite(partner)
        if ok.sum() > 8:
            dens_r = float(np.corrcoef(ni["density"][ok], partner[ok])[0, 1])

    return dict(G=G, tau=float(tau), tau_info=tinfo, fit=fit,
                unmatched_i=um_i, unmatched_j=um_j, concentration=conc,
                mean_unmatched_i=float(um_i.mean()),
                mean_unmatched_j=float(um_j.mean()),
                mean_concentration=float(np.nanmean(conc)),
                density_partner_r=dens_r)


def gated_lift_counts(G: np.ndarray, mem_i, mem_j, emb_i_bins, emb_j_bins,
                      local_k: int) -> sp.csr_matrix:
    """Lift niche coupling to bin edges; empty niche rows contribute nothing."""
    rows, cols, vals = [], [], []
    nz = np.argwhere(G > 0)
    for a_, b_ in nz:
        ii, jj = mem_i[a_], mem_j[b_]
        if ii.size == 0 or jj.size == 0:
            continue
        k = int(min(local_k, jj.size))
        D = distance.cdist(emb_i_bins[ii], emb_j_bins[jj])
        sel = (np.argpartition(D, k - 1, axis=1)[:, :k] if k < jj.size
               else np.tile(np.arange(jj.size), (ii.size, 1)))
        r = np.repeat(np.arange(ii.size), k)
        c = sel.ravel()
        rows.append(ii[r]); cols.append(jj[c])
        vals.append(float(G[a_, b_]) / (1.0 + D[r, c]))
    if not rows:
        return sp.csr_matrix((len(emb_i_bins), len(emb_j_bins)))
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(emb_i_bins), len(emb_j_bins))).tocsr()
    A.sum_duplicates()
    return A


# ===========================================================================
# 4. assemble
# ===========================================================================

def stage_preflight(L: sp.csr_matrix, stages: Sequence[str],
                    ns: Sequence[int], order: Sequence[str],
                    min_bridge_ratio: float = 0.05) -> Dict[str, Any]:
    """
    Is the chain connected WELL ENOUGH, before any fitting?

    NOTE what this deliberately does NOT test. The anchor (two-condition)
    version clustered the leading eigenvectors and flagged a partition that
    recovered the condition labels. For a CHAIN that test is guaranteed to
    fire and therefore says nothing: the stage groups are joined by exactly
    one bridge each, so cutting k-1 edges separates them, and spectral
    clustering will always find that cut. It is the topology working as
    designed, not integration failing.

    What matters instead is whether each bridge carries enough weight to be a
    real connection. Reported per bridge as the ratio of its total edge weight
    to the mean total weight of a within-stage edge. A bridge far below that
    is a bottleneck the tensor fit cannot compensate for, because the shared
    gene mode is what ties stages together and the graph only smooths -- but a
    near-zero bridge means the smoothing does not reach across the stage
    boundary at all.
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.csgraph import connected_components

    A = sp.diags(L.diagonal()) - L
    ncomp, lbl_c = connected_components(A, directed=False)

    per = np.concatenate([[s] * n for s, n in zip(stages, ns)])
    uq = [s for s in order if s in set(stages)]

    fied = float("nan")
    if L.shape[0] > 3:
        try:
            vals = eigsh(L.astype(float), k=min(4, L.shape[0] - 2),
                         sigma=-1e-6, which="LM", return_eigenvectors=False)
            vals = np.sort(vals)
            fied = float(vals[1]) if len(vals) > 1 else float("nan")
        except Exception:
            pass

    # weight crossing each adjacent-stage boundary, vs weight within stages
    bridges = {}
    for a, b in zip(uq[:-1], uq[1:]):
        ma, mb = (per == a), (per == b)
        w_ab = float(A[ma][:, mb].sum())
        w_in = float(A[ma][:, ma].sum() + A[mb][:, mb].sum())
        bridges[f"{a}->{b}"] = dict(
            cross_weight=w_ab, within_weight=w_in,
            ratio=float(w_ab / max(w_in, 1e-12)))

    weak = [k for k, v in bridges.items() if v["ratio"] < min_bridge_ratio]
    return dict(n_components=int(ncomp), fiedler=fied,
                bridges=bridges, weak_bridges=weak,
                component_sizes=np.bincount(lbl_c).tolist())


def build_chain_graph(adatas: Sequence, cfg: Optional[ChainConfig] = None,
                      edges: Optional[List[Edge]] = None,
                      run_preflight: bool = True
                      ) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    cfg = cfg or ChainConfig()
    df = sample_table(adatas, cfg)
    sec2idx = dict(zip(df.section, df.idx))
    ns = [A.n_obs for A in adatas]
    offs = np.cumsum([0] + ns[:-1])
    N = int(sum(ns))
    rep: Dict[str, Any] = {"config": asdict(cfg),
                           "samples": df.to_dict("records")}

    # --- topology -------------------------------------------------------
    if edges is None:
        edges, hubs, warns = derive_chain(df, cfg)
    else:
        hubs, warns = {}, []
    rep["edges"] = [asdict(e) for e in edges]
    rep["hubs"] = hubs
    rep["warnings"] = warns
    for w in warns:
        print(f"[chain] WARN {w}")
    for st, hs in hubs.items():
        if len(hs) > 1:
            print(f"[chain] stage '{st}' has {len(hs)} hubs {hs}: each bridge "
                  f"needs its own lot-matched endpoint")

    # --- diagonal blocks ------------------------------------------------
    R_, C_, V_ = [], [], []

    def place(r0, c0, A):
        A = A.tocoo()
        if A.nnz:
            R_.append(A.row + r0); C_.append(A.col + c0); V_.append(A.data)

    radii = []
    coords = [_coords_of(A) for A in adatas]
    within_deg = []
    for k, A in enumerate(adatas):
        if cfg.within_graph == "scc":
            E = expression_knn_adj(A, use_rep=cfg.expr_key,
                                   n_neighbors=cfg.expr_k, seed=cfg.seed)
            S = lattice_from_adata(A, neigh=cfg.spatial_neigh)
            W = scc_union(E, S)
            dW = degree_stats(W)
            radii.append(dict(section=df.section[k], graph="scc",
                              mean_k_expr=degree_stats(E)["mean"],
                              mean_k_spatial=degree_stats(S)["mean"],
                              mean_k=dW["mean"],
                              frac_isolated=dW["frac_isolated"]))
        elif cfg.within_graph == "radius":
            r, m = solve_radius_for_degree(coords[k], cfg.k_within)
            W = _radius_adj(coords[k], r)
            radii.append(dict(section=df.section[k], graph="radius",
                              radius=float(r), mean_k=float(m)))
        else:
            raise ValueError(f"within_graph must be 'scc' or 'radius', "
                             f"got {cfg.within_graph!r}")
        within_deg.append(float(radii[-1]["mean_k"]))
        place(offs[k], offs[k], W)
    rep["within_slice"] = radii
    # The cross degree IS lambda_cross (see rescale_to_degree). Tie it to the
    # realised within degree so changing the within graph does not silently
    # change how hard cross-sample smoothing acts relative to within-sample.
    mean_within = float(np.mean(within_deg))
    k_cross = (float(cfg.lambda_cross) * mean_within
               if cfg.lambda_cross is not None else float(cfg.k_cross))
    rep["mean_within_degree"] = mean_within
    rep["k_cross_effective"] = k_cross
    print(f"[chain] within graph '{cfg.within_graph}': mean degree "
          f"{mean_within:.1f} -> cross degree target {k_cross:.2f} "
          f"(lambda_cross={cfg.lambda_cross})")

    # --- niches (pooled counts) + one shared embedding -------------------
    niches = {}
    for k, A in enumerate(adatas):
        niches[df.section[k]] = niche_pooled(A, cfg, seed=cfg.seed)
    hub_secs = sorted({h for hs in hubs.values() for h in hs}) or list(niches)
    # seed MUST be keyword: the third positional parameter is n_pcs, so
    # passing cfg.seed (0) here built PCA(n_components=0) -- a zero-width niche
    # embedding, hence an all-zero feature cost M, hence tau pinned at its floor
    # with separability exactly 0 and ~0% unmatched mass on every edge. The
    # coupling was then driven by the Gromov (spatial) term alone.
    evr, n_hvg = embed_niches(niches, hub_secs, seed=cfg.seed)
    rep["embedding"] = dict(evr=evr, n_hvg=n_hvg, fit_on=hub_secs)

    # --- couple every edge ----------------------------------------------
    bin_emb = {}
    for k, A in enumerate(adatas):
        nb = niches[df.section[k]]
        bin_emb[df.section[k]] = nb["emb"][nb["labels"]]

    raw, erep = {}, []
    coup_arrays = []          # the transport plans themselves, saved by the runner
    # per-bin unmatched mass, max over every edge a slice takes part in.
    # Leaves only have within-stage edges (tau floored tight), so high
    # unmatched mass -- and therefore triage's "stage_specific" class -- can
    # in practice only arise on hub sections. storm_chain_run.py pops this
    # from the report, saves it as unmatched_per_bin.npy and feeds triage.
    um_bin = {df.section[k]: np.zeros(ns[k], dtype=np.float32)
              for k in range(len(adatas))}
    n_edges_sec = {s: 0 for s in um_bin}
    for e in edges:
        i, j = int(sec2idx[e.src]), int(sec2idx[e.dst])
        ni, nj = niches[e.src], niches[e.dst]
        r = couple_edge(ni, nj, e.kind, cfg)
        A = gated_lift_counts(r["G"], ni["members"], nj["members"],
                              bin_emb[e.src], bin_emb[e.dst], cfg.lift_local_k)
        raw[(i, j)] = A
        um = r["unmatched_i"][ni["labels"]]
        um_j = r["unmatched_j"][nj["labels"]]
        um_bin[e.src] = np.maximum(um_bin[e.src], um.astype(np.float32))
        um_bin[e.dst] = np.maximum(um_bin[e.dst], um_j.astype(np.float32))
        n_edges_sec[e.src] += 1
        n_edges_sec[e.dst] += 1
        coup_arrays.append(dict(
            src=e.src, dst=e.dst, kind=e.kind, tau=float(r["tau"]),
            G=np.asarray(r["G"], dtype=np.float32),
            unmatched_i=np.asarray(r["unmatched_i"], dtype=np.float32),
            unmatched_j=np.asarray(r["unmatched_j"], dtype=np.float32),
            concentration=np.asarray(r["concentration"], dtype=np.float32)))
        erep.append(dict(
            src=e.src, dst=e.dst, kind=e.kind, lot_matched=e.lot_matched,
            note=e.note, tau=r["tau"],
            tau_identified=r["tau_info"]["identified"],
            tau_separability=r["tau_info"].get("separability"),
            converged=r["fit"]["converged"],
            unmatched_src=r["mean_unmatched_i"],
            unmatched_dst=r["mean_unmatched_j"],
            concentration=r["mean_concentration"],
            unmatched_spatial_coherence=spatial_coherence(um, coords[i]),
            density_partner_r=r["density_partner_r"]))
        print(f"[chain] {e.kind:>6s} {e.src}->{e.dst} "
              f"{'lot-ok' if e.lot_matched else 'LOT-DIFF'} "
              f"tau={r['tau']:.3g} unmatched={r['mean_unmatched_i']:.1%} "
              f"conc={r['mean_concentration']:.2f} "
              f"dens_r={r['density_partner_r']:+.2f}"
              f"{'' if r['tau_info']['identified'] else '  TAU UNIDENTIFIED'}")
    rep["couplings"] = erep
    # Raw objects for the transport agent: niche partition of every section
    # (labels, centroids, mass, density, shared embedding, pooled counts) and
    # every coupling G. Popped from the report by storm_chain_run.py and saved
    # as npz under niches/ and couplings/; with them, domain-to-domain transport
    # tables and post-hoc couplings of ANY pair can be computed without
    # rebuilding the graph (transport_tables.py, couple_pairs.py).
    rep["_niches"] = {s_: dict(labels=n["labels"], coords=n["coords"],
                               mass=n["mass"], density=n.get("density"),
                               emb=n["emb"], counts=n["counts"])
                      for s_, n in niches.items()}
    rep["_couplings"] = coup_arrays
    rep["unmatched_per_bin"] = um_bin
    rep["unmatched_summary"] = {
        s: dict(mean=float(v.mean()), q90=float(np.quantile(v, 0.9)),
                frac_gt_0_5=float((v > 0.5).mean()), n_edges=n_edges_sec[s])
        for s, v in um_bin.items()}

    # --- pooled top-k, symmetrise, balance ------------------------------
    per_src: Dict[int, Dict[int, sp.csr_matrix]] = {}
    for (i, j), A in raw.items():
        per_src.setdefault(i, {})[j] = A
        per_src.setdefault(j, {})[i] = A.T.tocsr()
    kept = {}
    for i, blocks in per_src.items():
        for j, A in pooled_topk(blocks, max(1, int(round(k_cross)))).items():
            kept[(i, j)] = A

    sym = {}
    for (i, j) in raw:
        A = kept.get((i, j), raw[(i, j)])
        B = kept.get((j, i))
        if B is not None:
            A = A.maximum(B.T)
        sym[(i, j)] = rescale_to_degree(A, k_cross)
    sym = balance_cross_degrees(sym, ns, k_cross)
    for (i, j), A in sym.items():
        place(offs[i], offs[j], A)
        place(offs[j], offs[i], A.T)

    # --- Laplacian ------------------------------------------------------
    A_all = sp.coo_matrix(
        (np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))),
        shape=(N, N)).tocsr()
    A_all = A_all.maximum(A_all.T)
    A_all.setdiag(0.0); A_all.eliminate_zeros()
    d = np.asarray(A_all.sum(1)).ravel()
    L = (sp.diags(d) - A_all).tocsr()
    if cfg.normalize_laplacian:
        L = _sym_normalize(L)

    dw, dc = [], []
    for k in range(len(adatas)):
        s = slice(int(offs[k]), int(offs[k]) + ns[k])
        blk = A_all[s, :]
        w = np.asarray(blk[:, s].sum(1)).ravel()
        t = np.asarray(blk.sum(1)).ravel()
        dw.append(float(w.mean())); dc.append(float((t - w).mean()))
    rep["realised_degrees"] = dict(
        within=dw, cross=dc,
        effective_lambda_cross=[float(c / w) if w > 0 else None
                                for c, w in zip(dc, dw)],
        hub_leaf_ratio=float(max(dc) / max(min(dc), 1e-9)))

    if run_preflight:
        rep["preflight"] = stage_preflight(L, list(df.stage), ns,
                                           cfg.stage_order)
        pf = rep["preflight"]
        print(f"[chain] preflight: components={pf['n_components']} "
              f"fiedler={pf['fiedler']:.2e}")
        for k, v in pf["bridges"].items():
            print(f"[chain]   bridge {k:<22s} cross/within weight = "
                  f"{v['ratio']:.4f}")
        if pf["weak_bridges"]:
            print(f"[chain] WARN weak bridges {pf['weak_bridges']}: smoothing "
                  f"barely reaches across that stage boundary")
        if pf["n_components"] > 1:
            print("[chain] HALT: graph disconnected; a slice is fit alone")
    print(f"[chain] degrees within={np.mean(dw):.1f} cross={np.mean(dc):.1f} "
          f"hub/leaf={rep['realised_degrees']['hub_leaf_ratio']:.1f}x")
    return L, rep
