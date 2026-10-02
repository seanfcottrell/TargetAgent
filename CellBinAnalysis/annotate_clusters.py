#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URLS = {
    "Adult_Human_PrefrontalCortex":
        "https://celltypist.cog.sanger.ac.uk/models/Human_PFC_Ma/v1/"
        "Adult_Human_PrefrontalCortex.pkl",
    "Adult_Human_MTG":
        "https://celltypist.cog.sanger.ac.uk/models/Human_MTG_Allen/v1/"
        "Adult_Human_MTG.pkl",
}
ALLEN_URLS = {
    "trimmed_means.csv": "https://idk-etl-prod-download-bucket.s3.amazonaws.com/"
                         "aibs_human_ctx_smart-seq/trimmed_means.csv",
    "metadata.csv": "https://idk-etl-prod-download-bucket.s3.amazonaws.com/"
                    "aibs_human_ctx_smart-seq/metadata.csv",
}
COARSE_ORDER = ["Ex", "Inh", "Ast", "Oli", "Opc", "Mic", "End", "Other", "Unk"]


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dst: str) -> None:
    import urllib.request
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"downloading {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, dst + ".part")
    os.replace(dst + ".part", dst)


class CoarseMap:
    def __init__(self, path: str):
        df = pd.read_csv(path)
        self.rules = [(re.compile(p), c) for p, c in zip(df.pattern, df.coarse)]

    def __call__(self, label: str):
        for rx, c in self.rules:
            if rx.search(str(label)):
                return c
        return None

    def validate(self, labels, what: str) -> dict:
        m = {str(l): self(l) for l in labels}
        bad = sorted(k for k, v in m.items() if v is None)
        if bad:
            raise SystemExit(f"coarse_map.csv does not cover {len(bad)} {what} "
                             f"label(s): {bad}\nAdd a pattern and rerun.")
        return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="output dir of celltype_nuclei.py")
    p.add_argument("--indir", required=True, help="dir of *_cellbin.h5ad")
    p.add_argument("--models", nargs="*",
                   default=["Adult_Human_MTG", "Adult_Human_PrefrontalCortex"],
                   help="first model is the primary caller")
    p.add_argument("--model-dir", default=os.path.join(_HERE, "..", "reference",
                                                       "celltypist"))
    p.add_argument("--allen-dir", default=os.path.join(_HERE, "..", "reference",
                                                       "allen_ctx_smartseq"))
    p.add_argument("--no-allen", action="store_true")
    p.add_argument("--coarse-map", required=True,
                   help="regex -> coarse label map from the data card (run_all.py passes <run>/inputs/coarse_map.csv)")
    p.add_argument("--min-share", type=float, default=0.5,
                   help="pseudo-cell vote share the primary model needs to decide")
    p.add_argument("--pseudo-n", type=int, default=50,
                   help="nuclei per pseudo-cell (summed raw counts)")
    p.add_argument("--max-pseudo", type=int, default=60,
                   help="max pseudo-cells per (section, cluster)")
    p.add_argument("--per-cell", action="store_true",
                   help="also run per-nucleus prediction (slow, noisy; record only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plots", action="store_true")
    a = p.parse_args()

    import scanpy as sc
    import anndata as ad
    import celltypist
    t0 = time.time()
    run = a.run
    cells = pd.read_csv(os.path.join(run, "cells.csv.gz"), keep_default_na=False,
                        dtype={"section": str, "stage": str, "cell_id": str,
                               "leiden": str})
    for c in ("cell_id", "section", "leiden"):
        if c not in cells.columns:
            raise SystemExit(f"cells.csv.gz lacks column {c}; rerun "
                             f"celltype_nuclei.py")
    types = pd.read_csv(os.path.join(run, "cluster_types.csv"), index_col=0,
                        dtype={"cluster": str})
    types.index = types.index.astype(str)
    hvg = [l.strip() for l in open(os.path.join(run, "hvg.txt")) if l.strip()]
    prof = pd.read_csv(os.path.join(run, "cluster_profiles.csv.gz"), index_col=0)
    prof.columns = prof.columns.astype(str)
    cmap = CoarseMap(a.coarse_map)
    rec = dict(started=time.ctime(), args=vars(a), models={}, allen={})
    print(f"{len(cells):,} QC-passed nuclei, {cells.leiden.nunique()} clusters, "
          f"{len(hvg)} HVGs", flush=True)

    # ---- models: fetch if absent, validate vocabulary -------------------
    models = {}
    for name in a.models:
        path = os.path.join(a.model_dir, f"{name}.pkl")
        if not os.path.isfile(path):
            if name not in MODEL_URLS:
                raise SystemExit(f"unknown model {name} and no file at {path}")
            _fetch(MODEL_URLS[name], path)
        m = celltypist.models.Model.load(path)
        models[name] = m
        coarse_of = cmap.validate(m.cell_types, f"celltypist:{name}")
        rec["models"][name] = dict(path=path, sha1=_sha1(path),
                                   n_types=len(m.cell_types),
                                   coarse_counts=pd.Series(coarse_of)
                                   .value_counts().to_dict())
        print(f"model {name}: {len(m.cell_types)} types -> "
              f"{rec['models'][name]['coarse_counts']}", flush=True)

    # ---- pseudo-bulk pseudo-cells per (section, cluster), shared genes ----
    import anndata as ad_mod
    import scipy.sparse as sp
    rng = np.random.default_rng(a.seed)
    sec_files = {}
    for sec in cells.section.unique():
        f = sorted(x for x in os.listdir(a.indir)
                   if x.startswith(f"{sec}_") and x.endswith(".h5ad"))
        if not f:
            raise SystemExit(f"no h5ad for section {sec} in {a.indir}")
        sec_files[sec] = os.path.join(a.indir, f[0])
    shared = None
    for f in sec_files.values():
        v = set(ad_mod.read_h5ad(f, backed="r").var_names)
        shared = v if shared is None else shared & v
    shared = np.array(sorted(shared))
    pb_rows, pb_meta = [], []
    percell = []
    for sec, sub in cells.groupby("section", observed=True):
        S = ad_mod.read_h5ad(sec_files[sec])
        S.obs_names = S.obs_names.astype(str)
        S = S[sub.cell_id.to_numpy(), shared].copy()      # rows in `sub` order
        X = sp.csr_matrix(S.X)
        for cl, idx in sub.groupby("leiden", observed=True).indices.items():
            perm = rng.permutation(idx)
            n_pc = int(min(a.max_pseudo, max(1, len(perm) // a.pseudo_n)))
            for chunk in np.array_split(perm, n_pc):
                pb_rows.append(np.asarray(X[chunk].sum(0)).ravel())
                pb_meta.append(dict(section=sec, cluster=str(cl),
                                    n_nuclei=int(len(chunk))))
        if a.per_cell:
            sc.pp.normalize_total(S, target_sum=1e4)
            sc.pp.log1p(S)
            d = pd.DataFrame(dict(cell_id=sub.cell_id.to_numpy(), section=sec,
                                  leiden=sub.leiden.to_numpy()))
            for name, m in models.items():
                res = celltypist.annotate(S, model=m, majority_voting=False)
                d[f"label_{name}"] = res.predicted_labels["predicted_labels"].to_numpy()
                d[f"conf_{name}"] = res.probability_matrix.max(1).to_numpy()
            percell.append(d)
        print(f"  {sec}: {S.n_obs:,} nuclei -> "
              f"{sum(1 for m in pb_meta if m['section'] == sec)} pseudo-cells "
              f"({time.time()-t0:.0f}s)", flush=True)
        del S, X
    PB = ad_mod.AnnData(X=sp.csr_matrix(np.vstack(pb_rows).astype(np.float32)),
                        obs=pd.DataFrame(pb_meta))
    PB.var_names = shared
    PB.obs["genes_detected"] = np.asarray((PB.X > 0).sum(1)).ravel()
    sc.pp.normalize_total(PB, target_sum=1e4)
    sc.pp.log1p(PB)
    print(f"{PB.n_obs} pseudo-cells over {len(shared):,} shared genes; median "
          f"genes detected per pseudo-cell {int(PB.obs.genes_detected.median())}",
          flush=True)
    pb = PB.obs.copy()
    for name, m in models.items():
        res = celltypist.annotate(PB, model=m, majority_voting=False)
        pb[f"label_{name}"] = res.predicted_labels["predicted_labels"].astype(str).to_numpy()
        pb[f"conf_{name}"] = res.probability_matrix.max(1).to_numpy().astype(np.float32)
        pb[f"coarse_{name}"] = pb[f"label_{name}"].map(cmap)
    pb.to_csv(os.path.join(run, "celltypist_pseudocells.csv.gz"), index=False)
    if percell:
        pd.concat(percell).to_csv(os.path.join(run, "celltypist_percell.csv.gz"),
                                  index=False)

    # ---- cluster-level majority per model (pseudo-cell votes) -------------
    for name in models:
        ct = pd.crosstab(pb.cluster, pb[f"coarse_{name}"], normalize="index")
        fine = pd.crosstab(pb.cluster, pb[f"label_{name}"], normalize="index")
        types[f"type_{name}"] = ct.idxmax(1).reindex(types.index)
        types[f"share_{name}"] = ct.max(1).reindex(types.index)
        types[f"fine_{name}"] = fine.idxmax(1).reindex(types.index)
        types[f"fine_share_{name}"] = fine.max(1).reindex(types.index)
        types[f"conf_{name}"] = (pb.groupby("cluster")[f"conf_{name}"].mean()
                                 .reindex(types.index))
    types["n_pseudocells"] = pb.groupby("cluster").size().reindex(types.index)

    # ---- Allen profile correlation ---------------------------------------
    if not a.no_allen:
        for fn, url in ALLEN_URLS.items():
            path = os.path.join(a.allen_dir, fn)
            if not os.path.isfile(path):
                _fetch(url, path)
        tm = pd.read_csv(os.path.join(a.allen_dir, "trimmed_means.csv"),
                         index_col=0)
        md = pd.read_csv(os.path.join(a.allen_dir, "metadata.csv"),
                         usecols=["cluster_label", "subclass_label",
                                  "class_label"], low_memory=False)
        sub_of = md.groupby("cluster_label").subclass_label.agg(
            lambda s: s.value_counts().index[0])
        subclasses = sorted(set(sub_of.values))
        coarse_sub = cmap.validate(subclasses, "allen subclass")
        sh = [g for g in hvg if g in tm.index and g in prof.index]
        X = prof.loc[sh]                                        # our clusters
        Y = tm.loc[sh, [c for c in tm.columns if c in sub_of.index]]
        Y = np.log1p(Y) if Y.to_numpy().max() > 50 else Y      # tolerate CPM
        # specificity profiles: centre every gene across clusters on each side,
        # keep genes that vary on both sides, then Pearson between clusters
        Xc = X.sub(X.mean(1), axis=0)
        Yc = Y.sub(Y.mean(1), axis=0)
        ok = (Xc.std(1) > 0) & (Yc.std(1) > 0)
        Xc, Yc = Xc[ok], Yc[ok]
        Xz = (Xc - Xc.mean(0)) / Xc.std(0)
        Yz = (Yc - Yc.mean(0)) / Yc.std(0)
        R = pd.DataFrame(Xz.T.to_numpy() @ Yz.to_numpy() / (ok.sum() - 1),
                         index=X.columns, columns=Y.columns)
        R.to_csv(os.path.join(run, "allen_correlation.csv"))
        shared_allen = int(ok.sum())
        best = R.idxmax(1)
        # margin: best rho minus best rho among clusters of a DIFFERENT subclass
        margin = {}
        for cl in R.index:
            bsub = sub_of[best[cl]]
            other = [c for c in R.columns if sub_of[c] != bsub]
            margin[cl] = float(R.loc[cl].max() - R.loc[cl, other].max())
        types["allen_cluster"] = best.reindex(types.index)
        types["allen_subclass"] = types.allen_cluster.map(sub_of)
        types["type_allen"] = types.allen_subclass.map(coarse_sub)
        types["rho_allen"] = R.max(1).reindex(types.index)
        types["margin_allen"] = pd.Series(margin).reindex(types.index)
        rec["allen"] = dict(n_shared_hvg=shared_allen, n_ref_clusters=int(R.shape[1]),
                            method="pearson on gene-centred (specificity) profiles",
                            sha1={fn: _sha1(os.path.join(a.allen_dir, fn))
                                  for fn in ALLEN_URLS})
        print(f"allen: {shared_allen} shared HVGs x {R.shape[1]} reference "
              f"clusters (centred-profile correlation)", flush=True)

    # ---- decision + agreement --------------------------------------------
    primary = a.models[0]
    # Recompute the marker call here from the saved profiles, so the vote does
    # not depend on whichever threshold the upstream run happened to use.
    from celltype_nuclei import score_clusters
    S_mk = score_clusters(prof)
    types["type_marker"] = S_mk.type_marker.reindex(types.index).fillna("Unk")
    types["margin"] = S_mk.margin.reindex(types.index)
    # Voting callers: the primary model, Allen, markers. Secondary models are
    # recorded but do not vote (the PFC model's systematic 'End' bias would
    # otherwise outvote a correct call).
    voters = [f"type_{primary}"] + (["type_allen"] if not a.no_allen else []) \
        + ["type_marker"]
    callers = [f"type_{n}" for n in models] + (["type_allen"] if not a.no_allen
                                               else []) + ["type_marker"]
    votes = types[voters].replace({"Unk": np.nan, "Other": np.nan})

    def decide(row):
        v = votes.loc[row.name].dropna()
        vc = v.value_counts()
        if len(vc) and vc.iloc[0] >= 2:                 # two callers agree
            return vc.index[0], f"consensus:{int(vc.iloc[0])}/{len(voters)}"
        if row[f"share_{primary}"] >= a.min_share and \
                pd.notna(row[f"type_{primary}"]) and \
                row[f"type_{primary}"] not in ("Unk", "Other"):
            return row[f"type_{primary}"], f"primary:{primary}"
        if row["type_marker"] != "Unk":
            return row["type_marker"], "marker"
        return "Unk", "none"

    dec = types.apply(decide, axis=1, result_type="expand")
    types["type_final"], types["decided_by"] = dec[0], dec[1]
    types["agree"] = votes.apply(lambda r: r.dropna().nunique() == 1
                                 and r.notna().sum() >= 2, axis=1)
    types.to_csv(os.path.join(run, "cluster_types.csv"))

    show = ["n_cells", "median_n_genes", "max_section"] + callers + \
           [f"share_{primary}", "type_final", "decided_by", "agree"]
    show = [c for c in show if c in types.columns]
    print("\ncluster annotation:")
    print(types[show].round(3).to_string())
    n_agree = int(types.agree.sum())
    w = types.n_cells / types.n_cells.sum()
    print(f"\nagree: {n_agree}/{len(types)} clusters, "
          f"{float(w[types.agree].sum()):.1%} of nuclei; "
          f"type_final Unk: {float(w[types.type_final == 'Unk'].sum()):.1%} of nuclei")

    # ---- write back: cells, composition, record --------------------------
    cells["type_final"] = cells.leiden.map(types.type_final).fillna("Unk")
    cells = cells[[c for c in cells.columns if not c.startswith("ct_")]]
    cells.to_csv(os.path.join(run, "cells.csv.gz"), index=False)
    comp_path = os.path.join(run, "composition_by_section.csv")
    base = (pd.read_csv(comp_path, index_col=0) if os.path.isfile(comp_path)
            else pd.DataFrame(index=sorted(cells.section.unique())))
    base = base[[c for c in base.columns if not c.startswith("frac_")
                 or c == "frac_kept"]]
    comp = pd.crosstab(cells.section, cells.type_final, normalize="index")
    comp = comp.reindex(columns=[c for c in COARSE_ORDER if c in comp.columns])
    comp.columns = [f"frac_{c}" for c in comp.columns]
    comp = base.join(comp)
    comp.to_csv(comp_path)
    print("\ncomposition per section (type_final; read against frac_kept):")
    print(comp.round(3).to_string())

    rp = os.path.join(run, "run_record.json")
    R0 = json.load(open(rp)) if os.path.isfile(rp) else {}
    rec.update(n_agree=n_agree, n_clusters=int(len(types)),
               frac_nuclei_agree=float(w[types.agree].sum()),
               frac_nuclei_unk=float(w[types.type_final == "Unk"].sum()),
               celltypist_version=celltypist.__version__,
               elapsed_s=round(time.time() - t0, 1))
    R0["annotate"] = rec
    json.dump(R0, open(rp, "w"), indent=2, default=str)

    if a.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tps = [t for t in COARSE_ORDER if t in set(cells.type_final)]
        cm = plt.get_cmap("tab10", max(len(tps), 3))
        col = {t: cm(i) for i, t in enumerate(tps)}
        if {"umap1", "umap2"}.issubset(cells.columns):
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(cells.umap1, cells.umap2, s=0.5, linewidths=0,
                       c=[col[t] for t in cells.type_final], rasterized=True)
            for cl, sub in cells.groupby("leiden"):
                ax.text(sub.umap1.median(), sub.umap2.median(), cl, fontsize=6,
                        ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("type_final (cluster ids at medians)", fontsize=9)
            ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=6,
                                          color=col[t], label=t) for t in tps],
                      fontsize=7, frameon=False, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(run, "umap_type_final.png"), dpi=150)
            plt.close(fig)
        if {"x", "y"}.issubset(cells.columns):        # spatial map of type_final
            secs = sorted(cells.section.unique()); ncol = 4
            nrow = int(np.ceil(len(secs) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow),
                                     squeeze=False)
            for ax, sec in zip(axes.ravel(), secs):
                sub = cells[cells.section == sec]
                ax.scatter(sub.x, sub.y, s=0.3, linewidths=0,
                           c=[col[t] for t in sub.type_final], rasterized=True)
                ax.set_title(f"{sec} ({sub.stage.iloc[0]}) n={len(sub):,}", fontsize=9)
                ax.set_aspect("equal"); ax.invert_yaxis()
                ax.set_xticks([]); ax.set_yticks([])
            for ax in axes.ravel()[len(secs):]:
                ax.axis("off")
            fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=6,
                                           color=col[t], label=t) for t in tps],
                       loc="lower center", ncol=len(tps), frameon=False)
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            fig.savefig(os.path.join(run, "spatial_type_final.png"), dpi=170)
            plt.close(fig)
        ct = pd.crosstab(pb.cluster, pb[f"coarse_{primary}"], normalize="index")
        ct = ct.reindex(columns=[c for c in COARSE_ORDER if c in ct.columns])
        ct = ct.loc[types.sort_values("type_final").index]
        fig, ax = plt.subplots(figsize=(1.5 + 0.5 * ct.shape[1],
                                        1.5 + 0.22 * ct.shape[0]))
        im = ax.imshow(ct.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, fontsize=8)
        ax.set_yticks(range(ct.shape[0]))
        ax.set_yticklabels([f"{c}  [{types.loc[c, 'type_final']}]"
                            for c in ct.index], fontsize=6)
        ax.set_xlabel(f"celltypist {primary} (coarse)"); ax.set_ylabel("leiden")
        fig.colorbar(im, ax=ax, fraction=0.03, label="share of pseudo-cells")
        fig.tight_layout()
        fig.savefig(os.path.join(run, "heatmap_cluster_x_celltypist.png"), dpi=150)
        plt.close(fig)

    print(f"\ndone in {rec['elapsed_s']}s -> {run}")


if __name__ == "__main__":
    main()
