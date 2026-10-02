#!/usr/bin/env python
"""
Cell-type composition of spatial domains: join QC-passed nuclei to the tile they
fall in, count types per (domain, section), and contrast case against control
per domain.

    python compose_domains.py --cells <run>/celltypes_qc/cells.csv.gz --storm <fit> \
        --storm-clusters bin_clusters_QH_sccg_res<r>.csv --out <run>/composition/res<r>

A nucleus at (x, y) lives in tile (x // bin, y // bin), the rule prep_bins.py
uses for nuclear density. Nuclei whose tile is absent from the fit are counted
as 'no_tile' and reported, never silently dropped. Only QC-passed nuclei are
typed, so every output carries frac_kept per section and a composition shift
must be read against it. The contrast is a descriptive Welch t-test on logit
fractions with the section as the unit: a screen, not an inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_utils import read_meta      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True,
                   help="cells.csv.gz from celltype_nuclei.py / annotate_clusters.py")
    p.add_argument("--type-col", default=None,
                   help="default: type_final if present, else type_marker")
    p.add_argument("--storm", required=True, help="domain run dir")
    p.add_argument("--storm-clusters", default="bin_clusters.csv")
    p.add_argument("--bin-size", type=int, default=110)
    p.add_argument("--min-cells", type=int, default=20,
                   help="(domain, section) cells needed to enter the test")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)

    cells = pd.read_csv(a.cells, keep_default_na=False,
                        dtype={"section": str, "stage": str})
    tcol = a.type_col or ("type_final" if "type_final" in cells.columns
                          else "type_marker")
    if tcol not in cells.columns:
        raise SystemExit(f"{a.cells} has no column {tcol}")
    cells[tcol] = cells[tcol].replace("", "Unk")
    b = a.bin_size
    cells["bx"] = (cells.x.astype(float) // b).astype(np.int64)
    cells["by"] = (cells.y.astype(float) // b).astype(np.int64)

    meta = read_meta(a.storm, a.storm_clusters)
    meta["bx"] = np.round((meta.x.astype(float) - b / 2.0) / b).astype(np.int64)
    meta["by"] = np.round((meta.y.astype(float) - b / 2.0) / b).astype(np.int64)
    tiles = meta[["section", "bx", "by", "cl", "stage"]].rename(
        columns={"cl": "domain", "stage": "stage_bin"})
    if tiles.duplicated(["section", "bx", "by"]).any():
        raise SystemExit("bin_meta has duplicate tiles; wrong --bin-size?")

    J = cells.merge(tiles, on=["section", "bx", "by"], how="left")
    J["domain"] = J.domain.fillna("no_tile")
    hit = (J.domain != "no_tile").mean()
    print(f"{len(J):,} typed nuclei; {hit:.1%} fall in a tile carried by the "
          f"run ({(J.domain == 'no_tile').sum():,} in no_tile)")
    J.to_csv(os.path.join(a.out, "cells_with_domain.csv.gz"), index=False)

    # frac_kept caveat, from the cell-typing run directory if available
    frac_kept = None
    comp_path = os.path.join(os.path.dirname(a.cells), "composition_by_section.csv")
    if os.path.isfile(comp_path):
        cs = pd.read_csv(comp_path, index_col=0)
        if "frac_kept" in cs.columns:
            frac_kept = cs.frac_kept

    # ---- counts per (domain, section, type) --------------------------------
    stage_of = J.groupby("section").stage.first()
    grp = (stage_of != "NA").map({True: "AD", False: "CTRL"})
    cnt = pd.crosstab([J.domain, J.section], J[tcol])
    n = cnt.sum(1)
    frac = cnt.div(n, axis=0)
    long = frac.stack().rename("frac").reset_index()
    long.columns = ["domain", "section", "type", "frac"]
    long["n_cells"] = long.set_index(["domain", "section"]).index.map(n)
    long["group"] = long.section.map(grp)
    long["stage"] = long.section.map(stage_of)
    if frac_kept is not None:
        long["frac_kept_section"] = long.section.map(frac_kept)
    long.to_csv(os.path.join(a.out, "composition_long.csv"), index=False)

    # ---- AD vs CTRL per (domain, type), sample as the unit -----------------
    rows = []
    eps = 1e-3
    for (dom, typ), sub in long.groupby(["domain", "type"]):
        sub = sub[sub.n_cells >= a.min_cells]
        x = sub[sub.group == "AD"].frac.to_numpy()
        y = sub[sub.group == "CTRL"].frac.to_numpy()
        r = dict(domain=dom, type=typ, n_ad=len(x), n_ctrl=len(y),
                 mean_ad=float(x.mean()) if len(x) else np.nan,
                 mean_ctrl=float(y.mean()) if len(y) else np.nan)
        if len(x) >= 2 and len(y) >= 2:
            lx = np.log((x + eps) / (1 - x + eps))
            ly = np.log((y + eps) / (1 - y + eps))
            t, pv = stats.ttest_ind(lx, ly, equal_var=False)
            r.update(diff=r["mean_ad"] - r["mean_ctrl"], t=float(t), p=float(pv))
        rows.append(r)
    res = pd.DataFrame(rows).sort_values(["domain", "type"])
    res.to_csv(os.path.join(a.out, "composition_by_domain.csv"), index=False)
    print("\ncase vs control per domain (descriptive):")
    show = res.dropna(subset=["p"]).sort_values("p").head(25)
    print(show.round(3).to_string(index=False))

    # ---- plot: mean composition per domain, CTRL vs AD ---------------------
    doms = [d for d in sorted(long.domain.unique(), key=lambda s: (len(s), s))
            if d != "no_tile"]
    types = sorted(long.type.unique())
    cmap = plt.get_cmap("tab10", max(len(types), 3))
    fig, axes = plt.subplots(1, 2, figsize=(1.0 + 0.6 * len(doms) * 2, 4.5),
                             sharey=True)
    for ax, g in zip(axes, ("CTRL", "AD")):
        M = (long[(long.group == g) & (long.n_cells >= a.min_cells)]
             .groupby(["domain", "type"]).frac.mean().unstack()
             .reindex(index=doms, columns=types).fillna(0))
        bottom = np.zeros(len(doms))
        for i, t in enumerate(types):
            ax.bar(range(len(doms)), M[t].to_numpy(), bottom=bottom,
                   color=cmap(i), label=t, width=0.8)
            bottom += M[t].to_numpy()
        ax.set_xticks(range(len(doms))); ax.set_xticklabels(doms, fontsize=7)
        ax.set_title(f"{g}: mean over sections", fontsize=9)
        ax.set_xlabel("domain")
    axes[0].set_ylabel("fraction of typed nuclei")
    axes[1].legend(fontsize=7, frameon=False, ncol=1, loc="upper right")
    cav = ("" if frac_kept is None else
           f"  |  QC pass rate: CTRL {frac_kept[grp[grp == 'CTRL'].index].mean():.0%}, "
           f"AD {frac_kept[grp[grp == 'AD'].index].mean():.0%}")
    fig.suptitle(f"composition per domain ({tcol}){cav}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "composition_by_domain.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(a.out, "run_record.json"), "w") as f:
        json.dump(dict(args=vars(a), type_col=tcol, n_cells=int(len(J)),
                       frac_in_tile=float(hit), n_domains=len(doms),
                       frac_kept_by_section=(frac_kept.to_dict()
                                             if frac_kept is not None else None)),
                  f, indent=2, default=str)
    print(f"\nwrote {a.out}/: composition_long.csv composition_by_domain.csv "
          f"composition_by_domain.png cells_with_domain.csv.gz")


if __name__ == "__main__":
    main()
