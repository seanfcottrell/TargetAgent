#!/usr/bin/env python
"""
Plot spatial domains on tissue, one panel per section, and write the per-domain
spatial-coherence table the packets and the report read.

    python plot_domains.py --run <fit> --clusters bin_clusters_QH_sccg_res<r>.csv --out <png>

Coherence is the fraction of each bin's k spatial neighbours sharing its label,
averaged per domain. It is descriptive: the fit is already spatially
regularised, so high coherence partly reflects that prior; low coherence is the
informative direction. A domain id means the same thing in every panel, because
the partition is joint over all sections.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a storm_out/<tag> directory")
    p.add_argument("--clusters", default="bin_clusters.csv",
                   help="cluster file inside --run (or an absolute path)")
    p.add_argument("--out", default=None, help="default: <run>/domains.png")
    p.add_argument("--k", type=int, default=8,
                   help="spatial neighbours for the coherence score")
    p.add_argument("--point-size", type=float, default=1.0)
    p.add_argument("--max-cols", type=int, default=4)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from domain_utils import read_meta, spatial_coherence

    run = a.run
    meta = read_meta(run, a.clusters)
    if "x" not in meta.columns:
        raise SystemExit(
            "bin_meta.csv has no x/y. This run predates the coordinate fix in "
            "storm_chain_run.py -- rerun, or join coordinates from the prepped "
            "h5ads in the same row order.")
    stem = os.path.splitext(os.path.basename(a.clusters))[0]
    suffix = "" if stem == "bin_clusters" else stem.replace("bin_clusters", "")

    doms = sorted(meta.cl.unique())
    cmap = plt.get_cmap("tab20", max(len(doms), 3))
    cidx = {d: i for i, d in enumerate(doms)}
    print(f"{len(doms)} domains, {len(meta):,} bins, "
          f"{meta.section.nunique()} sections")

    # ---- spatial coherence, per section then pooled per domain ----------
    coh, tbl = spatial_coherence(meta, k=a.k)
    print(f"\nspatial coherence (fraction of {a.k} nearest neighbours sharing "
          f"the label)\noverall mean = {np.nanmean(coh):.3f}\n")
    print(tbl.to_string())
    tbl.to_csv(os.path.join(run, f"domain_coherence{suffix}.csv"))

    # ---- panels ---------------------------------------------------------
    secs = sorted(meta.section.unique())
    ncol = min(a.max_cols, len(secs))
    nrow = int(np.ceil(len(secs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow),
                             squeeze=False)
    for ax, sec in zip(axes.ravel(), secs):
        sub = meta[meta.section == sec]
        ax.scatter(sub.x, sub.y, s=a.point_size,
                   c=[cmap(cidx[c]) for c in sub.cl], linewidths=0)
        ax.set_title(f"{sec}  ({sub.stage.iloc[0]})  n={len(sub):,}",
                     fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()          # image convention, so pial edge reads up
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(secs):]:
        ax.axis("off")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5,
                          color=cmap(cidx[d]), label=d) for d in doms]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(doms), 12),
               fontsize=7, frameon=False)
    fig.suptitle(f"{os.path.basename(run.rstrip('/'))}  [{stem}]", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    out = a.out or os.path.join(run, f"domains{suffix}.png")
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}\n      "
          f"{os.path.join(run, f'domain_coherence{suffix}.csv')}")


if __name__ == "__main__":
    main()
