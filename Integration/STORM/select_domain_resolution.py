#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "STORM"))               # the STORM package: bare imports, as it uses itself
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root FIRST: `Utils` must not resolve to STORM/Utils
from Utils.domain_utils import read_meta, spatial_coherence, stage_purity  # noqa: E402
from storm_chain_run import testable, triage                    # noqa: E402  AFTER Utils: it puts STORM/ first on sys.path


def search_target(counts: dict, target: int, evaluate, steps: int) -> list:
    """Bisect for a resolution giving exactly `target` domains.

    counts maps every resolution already evaluated to its domain count and is
    updated in place; evaluate(res) clusters at res and returns the count. The
    bracket is the largest resolution giving fewer than `target` and the
    smallest larger resolution giving more. Returns the resolutions added.
    """
    added = []
    if target in counts.values():
        return added
    below = [r for r, n in counts.items() if n < target]
    if not below:
        return added
    lo = max(below)
    higher = [r for r, n in counts.items() if n > target and r > lo]
    if not higher:
        return added
    hi = min(higher)
    for _ in range(steps):
        mid = round((lo + hi) / 2, 6)
        if mid in counts:
            break
        n = evaluate(mid)
        counts[mid] = n
        added.append(mid)
        if n == target:
            break
        lo, hi = (mid, hi) if n < target else (lo, mid)
    return added


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--scc", default=None, help="SCC baseline run dir")
    p.add_argument("--scc-clusters", default="bin_clusters_res0.5.csv")
    p.add_argument("--resolutions", nargs="+", type=float,
                   default=[0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5])
    p.add_argument("--fixed", type=float, default=None,
                   help="commit this resolution without selecting")
    p.add_argument("--rep", choices=["auto", "QHD", "QH"], default="auto")
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--cluster-graph", choices=["knn", "scc"], default="scc",
                   help="knn = expression kNN on the embedding (STORM's default "
                        "clustering); scc = kNN(--expr-k) UNION the tile lattice, "
                        "i.e. the graph the SCC baseline clusters on, applied to "
                        "the STORM embedding")
    p.add_argument("--expr-k", type=int, default=10,
                   help="kNN neighbours when --cluster-graph scc")
    p.add_argument("--bin-size", type=int, default=110)
    p.add_argument("--target-n", type=int, default=None,
                   help="exact number of domains wanted (the data card's prior); "
                        "overrides --target-min/--target-max")
    p.add_argument("--refine-steps", type=int, default=10,
                   help="bisection steps when no grid resolution gives --target-n")
    p.add_argument("--target-min", type=int, default=7)
    p.add_argument("--target-max", type=int, default=12)
    p.add_argument("--max-untestable", type=float, default=0.10)
    p.add_argument("--max-section", type=float, default=0.5,
                   help="candidates must have no domain (>=1%% of bins) with a larger single-section share")
    p.add_argument("--min-donors", type=int, default=2)
    p.add_argument("--min-bins", type=int, default=50)
    p.add_argument("--coherence-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write-all", action="store_true",
                   help="also write bin_clusters<sfx>_res<r>.csv for EVERY swept resolution")
    a = p.parse_args()

    import scanpy as sc
    import anndata as ad_mod
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    run = a.run
    fname = {"auto": "shared_coords.npy", "QHD": "shared_coords_QHD.npy",
             "QH": "shared_coords_QH.npy"}[a.rep]
    Z = np.load(os.path.join(run, fname))
    meta = read_meta(run)
    if len(meta) != Z.shape[0]:
        sys.exit(f"row mismatch: Z {Z.shape[0]} vs meta {len(meta)}")
    um_path = os.path.join(run, "unmatched_per_bin.npy")
    um = np.load(um_path) if os.path.isfile(um_path) else np.zeros(len(meta))
    stages = sorted(set(meta.stage))
    stage_order = ([s for s in ("NA", "moderate", "severe") if s in stages]
                   + [s for s in stages if s not in ("NA", "moderate", "severe")])

    # SCC labels joined onto this run's tiles (section, x, y)
    scc_lab = None
    if a.scc and not a.fixed or (a.scc and a.fixed):
        S = read_meta(a.scc, a.scc_clusters)
        key = lambda M: (M.section.astype(str) + "|" +
                         np.round(M.x.astype(float)).astype(int).astype(str) + "|" +
                         np.round(M.y.astype(float)).astype(int).astype(str))
        s_map = pd.Series(S.cl.to_numpy(), index=key(S))
        scc_lab = s_map.reindex(key(meta)).to_numpy()
        hit = pd.notna(scc_lab)
        print(f"SCC labels joined for {hit.mean():.1%} of tiles", flush=True)
        if hit.mean() < 0.99:
            print("WARNING: SCC join < 99%; ARI computed on the joined tiles only")

    A = ad_mod.AnnData(np.ascontiguousarray(Z))
    A.obs = meta.reset_index(drop=True)
    adjacency = None
    if a.cluster_graph == "scc":
        # exact kNN(--expr-k) on the embedding U rook lattice, from SCCGraph
        # (bit-reproducible; scanpy's approximate kNN moved Leiden at res 0.5
        # between 5 and 7 domains across nodes)
        from SCCGraph import embedding_lattice_graph
        adjacency, gdeg = embedding_lattice_graph(
            Z, meta.section, meta.x, meta.y, bin_size=a.bin_size, k=a.expr_k)
        print(f"SCC clustering graph (exact kNN): expr mean degree "
              f"{gdeg['expr']:.1f}, lattice {gdeg['lattice']:.1f}, union "
              f"{gdeg['union']:.1f}", flush=True)
    else:
        print(f"{fname} {Z.shape}; building approximate kNN(k={a.n_neighbors}) ...",
              flush=True)
        sc.pp.neighbors(A, n_neighbors=a.n_neighbors, use_rep="X",
                        random_state=a.seed)

    todo = sorted(set(a.resolutions + ([a.fixed] if a.fixed is not None else [])))
    rows, parts = [], {}

    def evaluate(res):
        key_ = f"leiden_{res:g}"
        sc.tl.leiden(A, resolution=float(res), key_added=key_, random_state=a.seed,
                     flavor="igraph", n_iterations=2, directed=False,
                     adjacency=adjacency)
        lab = A.obs[key_].astype(str).to_numpy()
        parts[res] = lab
        m = meta.assign(cl=lab)
        bad = sum(len(sub) for cl, sub in m.groupby("cl")
                  if not testable(sub, a.min_donors, a.min_bins, stages))
        pur = stage_purity(m)
        big = pur[pur.n_bins >= 0.01 * len(m)]
        coh = float(np.nanmean(spatial_coherence(m, k=a.coherence_k)[0])) \
            if {"x", "y"}.issubset(m.columns) else float("nan")
        r = dict(resolution=res, n_clusters=int(len(set(lab))),
                 untestable_frac=bad / len(m),
                 median_stage_purity=float(pur.stage_purity.median()),
                 max_section_big=float(big.max_section.max()) if len(big) else np.nan,
                 coherence=coh,
                 largest_frac=float(pd.Series(lab).value_counts(normalize=True).iloc[0]))
        if scc_lab is not None:
            hit = pd.notna(scc_lab)
            r["ari_vs_scc"] = float(adjusted_rand_score(scc_lab[hit], lab[hit]))
            r["nmi_vs_scc"] = float(normalized_mutual_info_score(scc_lab[hit], lab[hit]))
        rows.append(r)
        print("  " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.items()), flush=True)
        return r["n_clusters"]

    for res in todo:
        evaluate(res)

    refined = []
    if a.target_n is not None and a.fixed is None:
        counts = {r["resolution"]: r["n_clusters"] for r in rows}
        refined = search_target(counts, a.target_n, evaluate, a.refine_steps)
        for r_ in refined:
            print(f"  refined: resolution {r_:g} -> {counts[r_]} domains", flush=True)
    tab = pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)
    sfx0 = ("" if a.rep == "auto" else f"_{a.rep}") + ("_sccg" if a.cluster_graph == "scc" else "")
    tab.to_csv(os.path.join(run, f"resolution_selection{sfx0}.csv"), index=False)
    if a.write_all:
        for res, lab in parts.items():
            pd.DataFrame(dict(cluster=lab)).to_csv(
                os.path.join(run, f"bin_clusters{sfx0}_res{res:g}.csv"), index=False)
        print(f"wrote bin_clusters{sfx0}_res<r>.csv for {len(parts)} resolutions")

    if a.fixed is not None:
        chosen, rule = float(a.fixed), "fixed"
    elif a.target_n is not None:
        hit = tab[tab.n_clusters == a.target_n]
        if len(hit):
            chosen = float(hit.sort_values("coherence", ascending=False).iloc[0].resolution)
            rule = (f"exactly {a.target_n} domains (the data card's prior); highest coherence "
                    f"among {len(hit)} resolution(s) giving {a.target_n}")
        else:
            t2 = tab.assign(d=(tab.n_clusters - a.target_n).abs())
            chosen = float(t2.sort_values(["d", "coherence"], ascending=[True, False]).iloc[0].resolution)
            rule = f"fallback: no resolution gave {a.target_n} domains; closest count"
    else:
        cand = tab[(tab.n_clusters >= a.target_min) & (tab.n_clusters <= a.target_max)
                   & (tab.untestable_frac <= a.max_untestable)
                   & (tab.max_section_big.fillna(0) <= a.max_section)]
        score = "ari_vs_scc" if (a.scc and "ari_vs_scc" in tab.columns) else "coherence"
        if len(cand):
            chosen, rule = float(cand.sort_values(score, ascending=False).iloc[0].resolution), f"max {score} within target"
        else:
            t2 = tab.assign(d=(tab.n_clusters - a.target_min).abs())
            chosen = float(t2.sort_values(["d", score], ascending=[True, False]).iloc[0].resolution)
            rule = "fallback: n_clusters closest to target_min"
    row = tab[tab.resolution == chosen].iloc[0].to_dict()
    lab = parts[chosen]
    tri = triage(meta, lab, um if len(um) == len(meta) else np.zeros(len(meta)),
                 stage_order, a.min_donors, a.min_bins)
    rs = f"{chosen:g}"
    # explicit --rep QH/QHD gets its own file names so the two representations
    # of one fit can be selected and compared without overwriting each other
    sfx = ("" if a.rep == "auto" else f"_{a.rep}") + ("_sccg" if a.cluster_graph == "scc" else "")
    pd.DataFrame(dict(cluster=lab)).to_csv(os.path.join(run, f"bin_clusters{sfx}_res{rs}.csv"), index=False)
    pd.DataFrame(dict(cluster=lab)).to_csv(os.path.join(run, f"bin_clusters{sfx}_chosen.csv"), index=False)
    tri.to_csv(os.path.join(run, f"cluster_triage{sfx}_res{rs}.csv"), index=False)
    info = dict(resolution=chosen, resolution_str=rs, rule=rule, rep=fname,
                cluster_graph=a.cluster_graph,
                n_neighbors=a.n_neighbors,
                target=(a.target_n if a.target_n is not None else [a.target_min, a.target_max]),
                refined_resolutions=refined,
                max_untestable=a.max_untestable, metrics=row,
                triage=tri.classification.value_counts().to_dict())
    json.dump(info, open(os.path.join(run, f"chosen_resolution{sfx}.json"), "w"), indent=2, default=str)
    print(f"\nchosen resolution {rs} ({rule}): {row}")
    print(tri.head(15).to_string(index=False))
    print(f"wrote bin_clusters{sfx}_res{rs}.csv, bin_clusters{sfx}_chosen.csv, "
          f"cluster_triage{sfx}_res{rs}.csv, resolution_selection{sfx}.csv, "
          f"chosen_resolution{sfx}.json")


if __name__ == "__main__":
    main()
