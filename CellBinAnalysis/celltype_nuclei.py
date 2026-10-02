#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

DEFAULT_MARKER_PANEL = {
    "Ex":  ["SNAP25", "SLC17A7", "SATB2", "CAMK2A"],
    "Inh": ["GAD1", "GAD2", "SLC32A1"],
    "Ast": ["AQP4", "GFAP", "SLC1A2", "ALDH1L1"],
    "Oli": ["MBP", "PLP1", "MOBP", "ST18"],
    "Opc": ["LHFPL3", "PDGFRA", "VCAN", "PTPRZ1"],
    "Mic": ["CD74", "CX3CR1", "P2RY12", "CSF1R", "PTPRC"],
    "End": ["CLDN5", "FLT1", "PECAM1", "VWF", "PDGFRB"],
}


def _load_marker_panel() -> dict:
    path = os.environ.get("WALKTHROUGH_MARKERS")
    if not path:
        return {k: list(v) for k, v in DEFAULT_MARKER_PANEL.items()}
    with open(path) as fh:
        panel = json.load(fh)
    return {str(k): [str(g) for g in v] for k, v in panel.items()}


MARKER_PANEL = _load_marker_panel()
MARKER_SOURCE = os.environ.get("WALKTHROUGH_MARKERS") or "built-in default"
ALL_MARKERS = [g for gs in MARKER_PANEL.values() for g in gs]


def _file_fingerprint(path: str) -> dict:
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return dict(path=path, size=st.st_size, mtime=time.ctime(st.st_mtime),
                sha1_head_tail=h.hexdigest()[:16])


def _versions() -> dict:
    out = {}
    for m in ("scanpy", "anndata", "numpy", "pandas", "sklearn", "harmonypy",
              "umap", "igraph", "leidenalg"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def score_clusters(prof: pd.DataFrame, panel=MARKER_PANEL,
                   min_score: float = 0.5, min_margin: float = 0.5
                   ) -> pd.DataFrame:
    """
    Marker-set scores per cluster from mean log-normalised profiles: z-score
    each gene ACROSS clusters, average within a marker set. A cluster gets a
    type when its best score clears min_score and beats the runner-up by
    min_margin; otherwise 'Unk' (reported, never dropped).
    """
    rows = {}
    for t, genes in panel.items():
        g = [x for x in genes if x in prof.index]
        if not g:
            rows[f"score_{t}"] = pd.Series(np.nan, index=prof.columns)
            continue
        sub = prof.loc[g]
        z = (sub.sub(sub.mean(1), axis=0)
                .div(sub.std(1).replace(0, np.nan), axis=0))
        rows[f"score_{t}"] = z.mean(0)
    S = pd.DataFrame(rows)
    cols = list(S.columns)
    arr = S.to_numpy(dtype=float)
    arr = np.where(np.isnan(arr), -np.inf, arr)
    order = np.argsort(-arr, axis=1)
    best = arr[np.arange(len(S)), order[:, 0]]
    second = arr[np.arange(len(S)), order[:, 1]] if len(cols) > 1 else -np.inf
    label = np.array([cols[i].replace("score_", "") for i in order[:, 0]])
    ok = (best >= min_score) & ((best - second) >= min_margin)
    S["type_marker"] = np.where(ok, label, "Unk")
    S["margin"] = best - second
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True, help="dir of *_cellbin.h5ad")
    p.add_argument("--out", required=True, help="output DIRECTORY")
    p.add_argument("--min-genes", type=int, default=150,
                   help="paper: nuclei with <150 detected genes removed")
    p.add_argument("--min-counts", type=int, default=0)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--hvg-batch-key", default="section",
                   help="'none' for pooled dispersion (the literal paper "
                        "statement); default ranks HVGs per section first")
    p.add_argument("--scale-max", type=float, default=10.0)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="scanpy default; Stereopy's is 10")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="scanpy default, as in the paper")
    p.add_argument("--sweep", nargs="*", type=float,
                   default=[0.3, 0.5, 0.8, 1.0, 1.5])
    p.add_argument("--no-harmony", action="store_true")
    p.add_argument("--no-umap", action="store_true")
    p.add_argument("--sections", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--save-h5ad", action="store_true",
                   help="write typed.h5ad (HVG-scaled matrix, ~3 GB)")
    a = p.parse_args()

    import scanpy as sc
    import anndata as ad
    sc.settings.verbosity = 1
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    rec = dict(args=vars(a), started=time.ctime(), versions=_versions(),
               marker_panel=MARKER_PANEL, marker_source=MARKER_SOURCE)
    print(f"marker panel ({MARKER_SOURCE}): {', '.join(MARKER_PANEL)}", flush=True)

    # ---- load + QC -----------------------------------------------------
    files = sorted(glob.glob(os.path.join(a.indir, "*.h5ad")))
    if a.sections:
        want = set(a.sections)
        files = [f for f in files if os.path.basename(f).split("_")[0] in want]
    if not files:
        raise SystemExit(f"no h5ads in {a.indir}")
    rec["inputs"] = [_file_fingerprint(f) for f in files]
    parts = {}
    for f in files:
        sec = os.path.basename(f).split("_")[0]
        S = ad.read_h5ad(f)
        S.obs["cell_id"] = S.obs_names.astype(str)
        parts[sec] = S
    A = ad.concat(parts, label="section", index_unique="-", merge="same")
    A.obs_names_make_unique()
    del parts
    tot = np.asarray(A.X.sum(1)).ravel()
    ng = np.asarray((A.X > 0).sum(1)).ravel()
    A.obs["n_counts"], A.obs["n_genes"] = tot, ng
    before = A.obs.groupby("section", observed=True).agg(
        n_all=("n_genes", "size"), med_genes_all=("n_genes", "median"),
        med_counts_all=("n_counts", "median"))

    keep = (ng >= a.min_genes) & (tot >= a.min_counts)
    A = A[keep].copy()
    after = A.obs.groupby("section", observed=True).agg(
        n_kept=("n_genes", "size"), med_genes_kept=("n_genes", "median"),
        med_counts_kept=("n_counts", "median"))
    qc = before.join(after).fillna(0)
    qc["frac_kept"] = qc.n_kept / qc.n_all
    print(f"QC: n_genes >= {a.min_genes}, n_counts >= {a.min_counts}: kept "
          f"{A.n_obs:,} of {int(keep.size):,} cells "
          f"({A.n_obs / keep.size:.1%}); median genes "
          f"{np.median(A.obs.n_genes):.0f}, shared genes {A.n_vars:,}")
    print(qc.round(3).to_string(), flush=True)
    rec["qc_by_section"] = qc.reset_index().to_dict("records")
    rec["n_all"], rec["n_kept"] = int(keep.size), int(A.n_obs)
    all_genes = A.var_names.to_numpy().copy()

    # ---- normalise -> HVG -> markers -> scale -> PCA --------------------
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    bkey = (a.hvg_batch_key if a.hvg_batch_key.lower() != "none"
            and A.obs[a.hvg_batch_key].nunique() > 1 else None)
    sc.pp.highly_variable_genes(A, flavor="seurat", n_top_genes=a.n_top_genes,
                                batch_key=bkey)
    present = [g for g in ALL_MARKERS if g in A.var_names]
    missing = sorted(set(ALL_MARKERS) - set(present))
    if missing:
        print(f"markers absent from the shared gene space: {missing}")
    M = A[:, present].X
    A.obsm["markers"] = pd.DataFrame(
        M.toarray() if hasattr(M, "toarray") else np.asarray(M),
        index=A.obs_names, columns=present).astype(np.float32)
    A = A[:, A.var.highly_variable].copy()
    hvg = list(A.var_names)
    print(f"HVGs: {len(hvg)} (flavor=seurat, batch_key={bkey})", flush=True)
    sc.pp.scale(A, max_value=a.scale_max)
    sc.tl.pca(A, n_comps=a.n_pcs, svd_solver="arpack", random_state=a.seed)
    logdepth = np.log1p(A.obs["n_counts"].to_numpy())
    r = [float(abs(np.corrcoef(A.obsm["X_pca"][:, i], logdepth)[0, 1]))
         for i in range(min(5, a.n_pcs))]
    print("|corr(PC, log total counts)| PC1-5: "
          + " ".join(f"{v:.2f}" for v in r), flush=True)
    rec["pc_depth_corr"] = r
    rec["pca_evr_total"] = float(np.sum(A.uns["pca"]["variance_ratio"]))

    # ---- Harmony (harmonypy called directly; see celltype_cellbin.py) ---
    rep = "X_pca"
    if not a.no_harmony and bkey is not None:
        import harmonypy
        ho = harmonypy.run_harmony(A.obsm["X_pca"].astype(np.float64), A.obs,
                                   ["section"], random_state=a.seed)
        Z = np.asarray(ho.Z_corr)
        if Z.shape[0] != A.n_obs and Z.shape[-1] == A.n_obs:
            Z = Z.T
        if Z.shape != (A.n_obs, a.n_pcs):
            raise RuntimeError(f"harmony returned {Z.shape}, expected "
                               f"({A.n_obs}, {a.n_pcs})")
        A.obsm["X_pca_harmony"] = Z.astype(np.float32)
        rep = "X_pca_harmony"
        print(f"harmony done -> {Z.shape} ({time.time()-t0:.0f}s)", flush=True)
    rec["embedding"] = rep

    # ---- neighbours + Leiden (+ sweep on the same graph) + UMAP ---------
    sc.pp.neighbors(A, n_neighbors=a.n_neighbors, use_rep=rep,
                    random_state=a.seed)
    from scipy.sparse.csgraph import connected_components
    ncomp, comp = connected_components(A.obsp["connectivities"], directed=False)
    rec["knn_components"] = int(ncomp)
    rec["knn_largest_component_frac"] = float(np.bincount(comp).max() / len(comp))
    print(f"kNN graph: {ncomp} connected components (Leiden floor)")

    gmed = float(np.median(A.obs.n_counts))
    sweep_rows = []
    todo = sorted(set([float(x) for x in a.sweep] + [float(a.resolution)]))
    for res in todo:
        key = "leiden" if res == float(a.resolution) else f"leiden_{res:g}"
        sc.tl.leiden(A, resolution=res, key_added=key, random_state=a.seed,
                     flavor="igraph", n_iterations=2, directed=False)
        lab = A.obs[key].astype(str)
        comp_sec = pd.crosstab(lab, A.obs.section, normalize="index")
        med = A.obs.groupby(lab, observed=True).n_counts.median()
        sizes = lab.value_counts(normalize=True)
        big = sizes[sizes >= 0.01].index
        sweep_rows.append(dict(
            resolution=res, n_clusters=int(lab.nunique()),
            max_section_all=float(comp_sec.max(1).max()),
            max_section_big=float(comp_sec.loc[big].max(1).max()),
            depth_split=float((med < gmed / 2).mean()),
            largest_frac=float(sizes.iloc[0]),
            n_small_lt100=int((lab.value_counts() < 100).sum())))
        print(f"  res={res:<5g} k={sweep_rows[-1]['n_clusters']:<4d} "
              f"max_section(big)={sweep_rows[-1]['max_section_big']:.2f} "
              f"depth_split={sweep_rows[-1]['depth_split']:.2f} "
              f"largest={sweep_rows[-1]['largest_frac']:.1%}", flush=True)
        if key != "leiden":
            del A.obs[key]
    diag = pd.DataFrame(sweep_rows)
    diag.to_csv(os.path.join(a.out, "diagnostics.csv"), index=False)
    n_cl = int(A.obs.leiden.nunique())
    rec["resolution"], rec["n_clusters"] = float(a.resolution), n_cl
    rec["n_neighbors"], rec["leiden_flavor"] = a.n_neighbors, "igraph"
    rec["sweep"] = sweep_rows
    print(f"\n{n_cl} clusters at resolution {a.resolution}", flush=True)

    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(A.n_obs, size=min(20000, A.n_obs), replace=False)
    lab_i = A.obs.leiden.to_numpy()[idx]
    rec["silhouette_20k"] = (float(silhouette_score(A.obsm[rep][idx], lab_i))
                             if len(set(lab_i)) > 1 else None)
    print(f"silhouette (20k subsample, {rep}): {rec['silhouette_20k']}")

    if not a.no_umap:
        sc.tl.umap(A, random_state=a.seed)
        print(f"umap done ({time.time()-t0:.0f}s)", flush=True)

    # ---- early write: embedding + labels ---------------------------------
    keep_cols = [c for c in ("cell_id", "section", "stage", "x", "y", "area",
                             "n_counts", "n_genes", "leiden")
                 if c in A.obs.columns]
    cells = A.obs[keep_cols].copy()
    if "X_umap" in A.obsm:
        cells["umap1"] = A.obsm["X_umap"][:, 0]
        cells["umap2"] = A.obsm["X_umap"][:, 1]
    cells.to_csv(os.path.join(a.out, "cells.csv.gz"), index=False)
    np.save(os.path.join(a.out, "X_pca.npy"), A.obsm[rep].astype(np.float32))
    with open(os.path.join(a.out, "hvg.txt"), "w") as f:
        f.write("\n".join(hvg) + "\n")
    rec["elapsed_s_embedding"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "run_record.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print("wrote cells.csv.gz, X_pca.npy, hvg.txt, run_record.json "
          "(recluster_cells.py can now resweep from these)", flush=True)

    # ---- cluster profiles over ALL shared genes, streamed ---------------
    lab_by_sec = {s: pd.Series(d.leiden.astype(str).to_numpy(),
                               index=d.cell_id.astype(str))
                  for s, d in cells.groupby("section", observed=True)}
    cls = sorted(cells.leiden.astype(str).unique(), key=lambda s: int(s))
    ci = {c: i for i, c in enumerate(cls)}
    acc = np.zeros((len(all_genes), len(cls)), dtype=np.float64)
    cnt = np.zeros(len(cls), dtype=np.int64)
    frac_pos = np.zeros((len(present), len(cls)), dtype=np.float64)
    seen = 0
    for f in files:
        sec = os.path.basename(f).split("_")[0]
        if sec not in lab_by_sec:
            continue
        S = ad.read_h5ad(f)
        S = S[:, all_genes].copy()
        lab = lab_by_sec[sec]
        m = np.isin(S.obs_names.astype(str), lab.index.to_numpy())
        S = S[m].copy()
        sub = lab.loc[S.obs_names.astype(str)].to_numpy()
        seen += S.n_obs
        sc.pp.normalize_total(S, target_sum=1e4)
        sc.pp.log1p(S)
        gi = [list(all_genes).index(g) for g in present]
        for c in set(sub):
            mm = sub == c
            acc[:, ci[c]] += np.asarray(S.X[mm].sum(0)).ravel()
            frac_pos[:, ci[c]] += np.asarray((S.X[mm][:, gi] > 0).sum(0)).ravel()
            cnt[ci[c]] += int(mm.sum())
        del S
    if seen != len(cells):
        raise RuntimeError(f"profile pass saw {seen} cells, expected {len(cells)}")
    prof = pd.DataFrame(acc / np.maximum(cnt, 1), index=all_genes, columns=cls)
    prof.to_csv(os.path.join(a.out, "cluster_profiles.csv.gz"))
    fpos = pd.DataFrame(frac_pos / np.maximum(cnt, 1), index=present, columns=cls)

    # ---- marker-based coarse labels per cluster --------------------------
    S = score_clusters(prof)
    comp_sec = pd.crosstab(cells.leiden.astype(str), cells.section,
                           normalize="index")
    types = pd.DataFrame(dict(
        cluster=cls,
        n_cells=[int(cnt[ci[c]]) for c in cls],
        frac_cells=[float(cnt[ci[c]] / cnt.sum()) for c in cls],
        median_n_genes=[float(cells.n_genes[cells.leiden.astype(str) == c].median())
                        for c in cls],
        max_section=[float(comp_sec.loc[c].max()) for c in cls],
        top_section=[str(comp_sec.loc[c].idxmax()) for c in cls],
    )).set_index("cluster").join(S)
    for t, genes in MARKER_PANEL.items():
        g0 = genes[0]
        if g0 in fpos.index:
            types[f"frac_{g0}"] = fpos.loc[g0].values
    types.to_csv(os.path.join(a.out, "cluster_types.csv"))
    cells["type_marker"] = cells.leiden.astype(str).map(types.type_marker)
    cells.to_csv(os.path.join(a.out, "cells.csv.gz"), index=False)
    print("\ncluster -> marker type:")
    print(types[["n_cells", "median_n_genes", "max_section", "type_marker",
                 "margin"]].to_string())
    rec["type_marker_coverage"] = float((cells.type_marker != "Unk").mean())
    rec["max_section_big_clusters"] = float(
        types[types.frac_cells >= 0.01].max_section.max())

    # ---- composition per section -----------------------------------------
    comp = pd.crosstab(cells.section, cells.type_marker, normalize="index")
    comp.columns = [f"frac_{c}" for c in comp.columns]
    comp = qc[["n_all", "n_kept", "frac_kept", "med_genes_kept"]].join(comp)
    comp.to_csv(os.path.join(a.out, "composition_by_section.csv"))
    print("\ncomposition of KEPT cells per section (read against frac_kept):")
    print(comp.round(3).to_string())

    # ---- plots -----------------------------------------------------------
    if a.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        A.obs["type_marker"] = cells.type_marker.to_numpy()
        A.obs["log_n_genes"] = np.log10(A.obs.n_genes.astype(float) + 1)
        if "X_umap" in A.obsm:
            for col in ("leiden", "type_marker", "section", "stage",
                        "log_n_genes"):
                fig = sc.pl.umap(A, color=col, size=2, alpha=0.6,
                                 legend_loc="on data" if col in
                                 ("leiden", "type_marker") else "right margin",
                                 legend_fontsize=6, return_fig=True,
                                 show=False)
                fig.savefig(os.path.join(a.out, f"umap_{col}.png"), dpi=150,
                            bbox_inches="tight")
                plt.close(fig)
        Mk = ad.AnnData(X=A.obsm["markers"].to_numpy(),
                        obs=A.obs[["leiden"]].copy())
        Mk.var_names = list(A.obsm["markers"].columns)
        panel = {t: [g for g in gs if g in Mk.var_names]
                 for t, gs in MARKER_PANEL.items()}
        dp = sc.pl.dotplot(Mk, var_names=panel, groupby="leiden",
                           standard_scale="var", return_fig=True, show=False)
        dp.savefig(os.path.join(a.out, "dotplot_markers.png"), dpi=150,
                   bbox_inches="tight")
        plt.close("all")
        if {"x", "y"}.issubset(cells.columns):
            secs = sorted(cells.section.unique())
            tps = sorted(cells.type_marker.unique())
            cmap = plt.get_cmap("tab10", max(len(tps), 3))
            cidx = {t: i for i, t in enumerate(tps)}
            ncol = 4
            nrow = int(np.ceil(len(secs) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow),
                                     squeeze=False)
            for ax, sec in zip(axes.ravel(), secs):
                sub = cells[cells.section == sec]
                ax.scatter(sub.x, sub.y, s=0.3, linewidths=0,
                           c=[cmap(cidx[t]) for t in sub.type_marker])
                ax.set_title(f"{sec} ({sub.stage.iloc[0]}) n={len(sub):,} "
                             f"kept={qc.loc[sec, 'frac_kept']:.0%}", fontsize=9)
                ax.set_aspect("equal"); ax.invert_yaxis()
                ax.set_xticks([]); ax.set_yticks([])
            for ax in axes.ravel()[len(secs):]:
                ax.axis("off")
            handles = [plt.Line2D([], [], marker="o", ls="", ms=6,
                                  color=cmap(cidx[t]), label=t) for t in tps]
            fig.legend(handles=handles, loc="lower center", ncol=len(tps),
                       frameon=False)
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            fig.savefig(os.path.join(a.out, "spatial_types.png"), dpi=170)
            plt.close(fig)
        fc = [c for c in comp.columns if c.startswith("frac_") and c != "frac_kept"]
        fig, ax = plt.subplots(figsize=(10, 4))
        comp[fc].plot(kind="bar", stacked=True, ax=ax, width=0.8,
                      colormap="tab10")
        for i, (sec, row) in enumerate(comp.iterrows()):
            ax.text(i, 1.01, f"{row.frac_kept:.0%}", ha="center", fontsize=7)
        ax.set_ylabel("fraction of kept nuclei"); ax.set_ylim(0, 1.08)
        ax.set_title("composition per section (label above bar = fraction of "
                     "segmented nuclei passing QC)", fontsize=9)
        ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "composition_bars.png"), dpi=150)
        plt.close(fig)

    if a.save_h5ad:
        A.write_h5ad(os.path.join(a.out, "typed.h5ad"), compression="gzip")

    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["outputs"] = sorted(os.listdir(a.out))
    with open(os.path.join(a.out, "run_record.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"\ndone in {rec['elapsed_s']}s -> {a.out}")
    for k in sorted(os.listdir(a.out)):
        print(f"  {k:28s} {os.path.getsize(os.path.join(a.out, k))/1e6:8.1f} MB")


if __name__ == "__main__":
    main()
