#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root
from Utils.domain_utils import read_meta   # noqa: E402


def domain_matrix(labels, dom_codes, n_niches, n_dom):
    """niche x domain fraction matrix from per-bin niche labels + domain codes."""
    M = np.zeros((n_niches, n_dom))
    np.add.at(M, (labels, dom_codes), 1.0)
    return M / np.maximum(M.sum(1, keepdims=True), 1)


def one_direction(G, um_src, conc_src, lab_src, lab_dst, dom_src, dom_dst,
                  mass_src, n_dom):
    a = mass_src / mass_src.sum()
    Ds = domain_matrix(lab_src, dom_src, G.shape[0], n_dom)
    Dd = domain_matrix(lab_dst, dom_dst, G.shape[1], n_dom)
    T = Ds.T @ G @ Dd                                  # D x D mass
    a_d = Ds.T @ a                                     # source mass per domain
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(a_d[:, None] > 0, T / a_d[:, None], np.nan)
    unmatched = 1.0 - np.nansum(frac, 1)
    g_r = G.sum(1)
    R = G / np.maximum(g_r[:, None], 1e-30)
    P = R @ Dd                                         # niche x partner-domain
    Pb = P[lab_src]                                    # per bin
    return frac, unmatched, a_d, Pb, um_src[lab_src], conc_src[lab_src]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--clusters", default="bin_clusters.csv")
    p.add_argument("--couplings", default=None, help="default <run>/couplings")
    p.add_argument("--niches", default=None, help="default <run>/niches")
    p.add_argument("--out", default=None, help="default <run>/transport")
    p.add_argument("--min-domain-bins", type=int, default=100,
                   help="source domains with fewer bins in a section are NaN")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = a.run
    cdir = a.couplings or os.path.join(run, "couplings")
    ndir = a.niches or os.path.join(run, "niches")
    out = a.out or os.path.join(run, "transport")
    os.makedirs(out, exist_ok=True)
    meta = read_meta(run, a.clusters)
    doms = sorted(meta.cl.unique(), key=lambda s: (len(s), s))
    dcode = {d: i for i, d in enumerate(doms)}
    meta["dcode"] = meta.cl.map(dcode).to_numpy()
    sec_rows = {s: np.where(meta.section.to_numpy() == s)[0]
                for s in meta.section.unique()}
    niche = {}
    for f in glob.glob(os.path.join(ndir, "*.npz")):
        s = os.path.basename(f)[:-4]
        if s.endswith("_counts"):
            continue
        z = np.load(f)
        niche[s] = dict(labels=z["labels"], mass=z["mass"])
        if len(z["labels"]) != len(sec_rows.get(s, [])):
            sys.exit(f"niche labels for {s} ({len(z['labels'])}) do not match "
                     f"bins in bin_meta ({len(sec_rows.get(s, []))})")
    files = sorted(glob.glob(os.path.join(cdir, "*.npz")))
    if not files:
        sys.exit(f"no couplings in {cdir}")
    print(f"{len(files)} couplings, {len(doms)} domains, {len(niche)} sections",
          flush=True)

    long, summ = [], []
    for f in files:
        z = np.load(f)
        src, dst = os.path.basename(f)[:-4].split("__")
        kind = str(z["kind"]) if "kind" in z else "?"
        G = z["G"].astype(np.float64)
        for direction, (S, D, Gd, um, cc) in {
                "fwd": (src, dst, G, z["unmatched_i"], z["concentration"]),
                "rev": (dst, src, G.T, z["unmatched_j"],
                        np.full(G.shape[1], np.nan, np.float32))}.items():
            if S not in niche or D not in niche:
                continue
            ms, md = meta.iloc[sec_rows[S]], meta.iloc[sec_rows[D]]
            frac, unm, a_d, Pb, umb, ccb = one_direction(
                Gd, um, cc, niche[S]["labels"], niche[D]["labels"],
                ms.dcode.to_numpy(), md.dcode.to_numpy(), niche[S]["mass"],
                len(doms))
            n_src = np.bincount(ms.dcode, minlength=len(doms))
            for i, d in enumerate(doms):
                if n_src[i] < a.min_domain_bins:
                    continue
                for j, d2 in enumerate(doms):
                    long.append(dict(edge=f"{src}__{dst}", direction=direction,
                                     kind=kind, src=S, dst=D, src_domain=d,
                                     dst_domain=d2, frac=float(frac[i, j]),
                                     src_n_bins=int(n_src[i])))
                row = frac[i]
                summ.append(dict(edge=f"{src}__{dst}", direction=direction,
                                 kind=kind, src=S, dst=D, src_domain=d,
                                 src_n_bins=int(n_src[i]),
                                 unmatched=float(unm[i]),
                                 self_map=float(row[i]),
                                 top_partner=doms[int(np.nanargmax(row))],
                                 top_partner_frac=float(np.nanmax(row)),
                                 entropy=float(-np.nansum(np.where(row > 0, row * np.log(np.where(row > 0, row, 1.0)), 0.0))),
                                 tau=float(z["tau"])))
            top = np.nanargmax(Pb, 1)
            pd.DataFrame(dict(section=S, x=ms.x.to_numpy(), y=ms.y.to_numpy(),
                              domain=ms.cl.to_numpy(),
                              partner=D, partner_top_domain=[doms[t] for t in top],
                              partner_top_share=Pb[np.arange(len(top)), top],
                              unmatched=umb, concentration=ccb)).to_csv(
                os.path.join(out, f"bin_partner_{S}__{D}.csv.gz"), index=False)
        # heatmaps, both directions
        fig, axes = plt.subplots(1, 2, figsize=(2.2 + 0.55 * len(doms) * 2, 1.8 + 0.5 * len(doms)))
        for ax, direction in zip(axes, ("fwd", "rev")):
            sub = [r for r in long if r["edge"] == f"{src}__{dst}" and r["direction"] == direction]
            if not sub:
                ax.axis("off"); continue
            M = pd.DataFrame(sub).pivot(index="src_domain", columns="dst_domain", values="frac").reindex(index=doms, columns=doms)
            im = ax.imshow(M.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(doms))); ax.set_xticklabels(doms, fontsize=7)
            ax.set_yticks(range(len(doms))); ax.set_yticklabels(doms, fontsize=7)
            S_, D_ = (src, dst) if direction == "fwd" else (dst, src)
            ax.set_title(f"{S_} -> {D_} ({kind})  rows: source domain", fontsize=8)
            ax.set_xlabel("partner domain"); ax.set_ylabel("source domain")
            for i in range(len(doms)):
                for j in range(len(doms)):
                    v = M.iat[i, j]
                    if pd.notna(v) and v >= 0.15:
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w")
        fig.colorbar(im, ax=axes, fraction=0.02, label="share of source-domain mass")
        fig.savefig(os.path.join(out, f"heatmap_{src}__{dst}.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  {src}->{dst} ({kind}) tau={float(z['tau']):.3g}", flush=True)

    L = pd.DataFrame(long); Sm = pd.DataFrame(summ)
    # domain ids are strings; keep them so 'NA'-style labels and leading zeros survive
    L["src_domain"] = L.src_domain.astype(str); L["dst_domain"] = L.dst_domain.astype(str)
    Sm["src_domain"] = Sm.src_domain.astype(str); Sm["top_partner"] = Sm.top_partner.astype(str)
    L.to_csv(os.path.join(out, "transport_tables.csv"), index=False)
    Sm.to_csv(os.path.join(out, "domain_transport_summary.csv"), index=False)
    print("\nper source domain, mean over edges of the same kind:")
    print(Sm.groupby(["kind", "src_domain"])[["unmatched", "self_map", "top_partner_frac"]]
            .mean().round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
