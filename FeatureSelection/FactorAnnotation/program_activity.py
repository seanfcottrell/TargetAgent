#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root
from CellBinAnalysis.celltype_nuclei import MARKER_PANEL       # noqa: E402
from Utils.domain_utils import read_meta                       # noqa: E402

STAGE_ORD = {"NA": 0, "moderate": 1, "severe": 2}


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.empty(n)
    r[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(r, 0, 1)


def contrast(long, value, min_n):
    """Welch AD vs CTRL + stage trend per (domain, programme), sample as unit."""
    from scipy import stats
    rows = []
    for (d, r), sub in long[long.n_bins >= min_n].groupby(["domain", "program"]):
        x = sub[sub.group == "AD"][value].to_numpy()
        y = sub[sub.group == "CTRL"][value].to_numpy()
        row = dict(domain=d, program=r, n_ad=len(x), n_ctrl=len(y),
                   mean_ad=x.mean() if len(x) else np.nan,
                   mean_ctrl=y.mean() if len(y) else np.nan)
        if len(x) >= 2 and len(y) >= 2:
            sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) or np.nan
            t, p = stats.ttest_ind(x, y, equal_var=False)
            row.update(diff=x.mean() - y.mean(), cohen_d=(x.mean() - y.mean()) / sp,
                       t=float(t), p=float(p))
            rho, pr = stats.spearmanr(sub.stage_ord, sub[value])
            row.update(stage_rho=float(rho), stage_p=float(pr))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "p" in out:
        ok = out.p.notna()
        out.loc[ok, "fdr"] = bh(out.loc[ok, "p"])
        ok2 = out.stage_p.notna()
        out.loc[ok2, "stage_fdr"] = bh(out.loc[ok2, "stage_p"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--clusters", required=True, help="cluster file inside --run")
    p.add_argument("--out", required=True)
    p.add_argument("--min-bins", type=int, default=50,
                   help="(domain, sample) needs this many bins to enter the tests")
    p.add_argument("--top-genes", type=int, default=25)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    run = a.run

    meta = read_meta(run, a.clusters)
    QH = np.load(os.path.join(run, "shared_coords_QH.npy"))
    QHD = np.load(os.path.join(run, "shared_coords_QHD.npy"))
    w = np.load(os.path.join(run, "w.npy"))
    B = np.load(os.path.join(run, "B_gene_loadings.npy"))
    genes = pd.read_csv(os.path.join(run, "genes.csv")).iloc[:, 0].astype(str).to_numpy()
    R = QH.shape[1]
    progs = [f"F{r}" for r in range(R)]
    secs = list(dict.fromkeys(meta.section))
    assert w.shape == (len(secs), R) and B.shape == (len(genes), R)
    stage_of = meta.groupby("section").stage.first()
    group_of = (stage_of != "NA").map({True: "AD", False: "CTRL"})
    print(f"{len(meta):,} bins, {meta.cl.nunique()} domains, R={R}, "
          f"{len(secs)} samples", flush=True)

    # ---- programme annotation: top genes + marker-set enrichment ----------
    top = {}
    for r in range(R):
        o = np.argsort(-B[:, r])[:a.top_genes]
        top[progs[r]] = [f"{genes[i]} ({B[i, r]:.3f})" for i in o]
    pd.DataFrame(top).to_csv(os.path.join(a.out, "program_top_genes.csv"), index=False)
    gi = {g: i for i, g in enumerate(genes)}
    enr = {}
    for t, gs in MARKER_PANEL.items():
        idx = [gi[g] for g in gs if g in gi]
        enr[t] = (B[idx].mean(0) / B.mean(0)) if idx else np.full(R, np.nan)
    enr = pd.DataFrame(enr, index=progs)
    types = list(MARKER_PANEL)
    enr["annotation"] = [types[int(np.nanargmax(row))] if np.nanmax(row) > 2 else "mixed"
                         for row in enr[types].to_numpy()]
    enr["top5"] = [", ".join(g.split(" ")[0] for g in top[pg][:5]) for pg in progs]
    enr.to_csv(os.path.join(a.out, "program_annotation.csv"))
    print("\nprogramme annotation (marker enrichment = mean loading of the set / "
          "mean loading of all genes):")
    print(enr.round(2).to_string())

    # ---- per (domain, sample) means ---------------------------------------
    rows = []
    sec_idx = {s: i for i, s in enumerate(secs)}
    mu = {s: QH[np.where(meta.section.to_numpy() == s)[0]].mean(0) for s in secs}
    pd.DataFrame(np.vstack([mu[s] for s in secs]), index=secs, columns=progs
                 ).to_csv(os.path.join(a.out, "sample_mean_usage.csv"))
    for (d, s), idx in meta.groupby(["cl", "section"]).indices.items():
        for r in range(R):
            u = float(QH[idx, r].mean())
            uc = u - float(mu[s][r])
            wk = float(w[sec_idx[s], r])
            rows.append(dict(domain=d, section=s, program=progs[r], n_bins=len(idx),
                             usage=u, activity=float(QHD[idx, r].mean()),
                             usage_c=uc, activity_c=uc * wk, w=wk))
    long = pd.DataFrame(rows)
    long["stage"] = long.section.map(stage_of)
    long["stage_ord"] = long.stage.map(STAGE_ORD)
    long["group"] = long.section.map(group_of)
    long.to_csv(os.path.join(a.out, "activity_domain_sample.csv"), index=False)

    # domain x programme profile (equal weight per sample)
    ok = long[long.n_bins >= a.min_bins]
    prof = ok.groupby(["domain", "program"]).activity.mean().unstack()[progs]
    prof.to_csv(os.path.join(a.out, "activity_domain_mean.csv"))
    usage_prof = ok.groupby(["domain", "program"]).usage.mean().unstack()[progs]
    usage_prof.to_csv(os.path.join(a.out, "usage_domain_mean.csv"))

    # ---- contrasts ---------------------------------------------------------
    c_act = contrast(long, "activity", a.min_bins)
    c_use = contrast(long, "usage", a.min_bins)
    c_usec = contrast(long, "usage_c", a.min_bins)
    c_actc = contrast(long, "activity_c", a.min_bins)
    c_act.to_csv(os.path.join(a.out, "contrast_activity.csv"), index=False)
    c_use.to_csv(os.path.join(a.out, "contrast_usage.csv"), index=False)
    c_usec.to_csv(os.path.join(a.out, "contrast_usage_centred.csv"), index=False)
    c_actc.to_csv(os.path.join(a.out, "contrast_activity_centred.csv"), index=False)
    ok.groupby(["domain", "program"]).usage_c.mean().unstack()[progs].to_csv(
        os.path.join(a.out, "usage_centred_domain_mean.csv"))
    wdf = pd.DataFrame(w, index=secs, columns=progs)
    wdf["stage"] = stage_of.reindex(secs).values
    wdf["group"] = group_of.reindex(secs).values
    wdf.to_csv(os.path.join(a.out, "w_by_sample.csv"))
    wl = wdf.melt(id_vars=["stage", "group"], value_vars=progs, var_name="program",
                  value_name="w", ignore_index=False).reset_index(names="section")
    wl["stage_ord"] = wl.stage.map(STAGE_ORD); wl["domain"] = "ALL"; wl["n_bins"] = 10**9
    c_w = contrast(wl, "w", 0)
    c_w.to_csv(os.path.join(a.out, "contrast_w.csv"), index=False)

    print("\nprogramme amplitude w_k, AD vs CTRL (sample level):")
    print(c_w[["program", "mean_ctrl", "mean_ad", "cohen_d", "p", "fdr", "stage_rho",
               "stage_fdr"]].round(3).to_string(index=False))
    for name, c in (("ACTIVITY (Q_k H D_k)", c_act), ("USAGE (Q_k H)", c_use),
                    ("CENTRED USAGE (Q_k H minus section mean)", c_usec),
                    ("CENTRED ACTIVITY", c_actc)):
        print(f"\n{name}: strongest AD-vs-CTRL contrasts per domain x programme "
              f"(FDR <= 0.1 shown, else top 10):")
        sig = c[c.fdr <= 0.1] if "fdr" in c else c.iloc[:0]
        show = sig if len(sig) else c.sort_values("p").head(10)
        print(show[["domain", "program", "mean_ctrl", "mean_ad", "cohen_d", "p", "fdr",
                    "stage_rho", "stage_fdr"]].round(3).to_string(index=False))
        print(f"  n tests {c.p.notna().sum()}, FDR<=0.1: {int((c.fdr <= 0.1).sum())}, "
              f"stage-trend FDR<=0.1: {int((c.stage_fdr <= 0.1).sum())}")

    # ---- plots ---------------------------------------------------------------
    doms = list(prof.index)
    z = (prof - prof.mean(0)) / prof.std(0).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(1.2 + 0.6 * R, 1.2 + 0.45 * len(doms)))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(R)); ax.set_xticklabels([f"{pg}\n{enr.loc[pg, 'annotation']}"
                                                  for pg in progs], fontsize=7)
    ax.set_yticks(range(len(doms))); ax.set_yticklabels([f"domain {d}" for d in doms], fontsize=8)
    ax.set_title("mean programme activity per domain (z across domains)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "heatmap_domain_program_activity.png"), dpi=150); plt.close(fig)

    for name, c in (("activity", c_act), ("usage", c_use),
                    ("usage_centred", c_usec), ("activity_centred", c_actc)):
        M = c.pivot(index="domain", columns="program", values="cohen_d").reindex(index=doms, columns=progs)
        F = c.pivot(index="domain", columns="program", values="fdr").reindex(index=doms, columns=progs)
        fig, ax = plt.subplots(figsize=(1.2 + 0.6 * R, 1.2 + 0.45 * len(doms)))
        im = ax.imshow(M.to_numpy(), cmap="PuOr_r", vmin=-3, vmax=3, aspect="auto")
        for i in range(len(doms)):
            for j in range(R):
                f = F.iat[i, j]
                if pd.notna(f) and f <= 0.1:
                    ax.text(j, i, "*" if f > 0.05 else "**", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(R)); ax.set_xticklabels([f"{pg}\n{enr.loc[pg, 'annotation']}" for pg in progs], fontsize=7)
        ax.set_yticks(range(len(doms))); ax.set_yticklabels([f"domain {d}" for d in doms], fontsize=8)
        ax.set_title(f"AD vs CTRL {name} per domain: Cohen's d (* FDR<=0.1, ** <=0.05)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"heatmap_contrast_{name}.png"), dpi=150); plt.close(fig)

    order = sorted(secs, key=lambda s: (STAGE_ORD[stage_of[s]], s))
    W = wdf.loc[order, progs]
    fig, ax = plt.subplots(figsize=(1.2 + 0.6 * R, 1.2 + 0.35 * len(order)))
    im = ax.imshow(((W - W.mean(0)) / W.std(0)).to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(R)); ax.set_xticklabels([f"{pg}\n{enr.loc[pg, 'annotation']}" for pg in progs], fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{s} ({stage_of[s]})" for s in order], fontsize=8)
    ax.set_title("programme amplitude w_k per sample (z across samples)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "heatmap_w_by_sample.png"), dpi=150); plt.close(fig)

    json.dump(dict(run=run, clusters=a.clusters, R=R, samples=secs, domains=doms,
                   min_bins=a.min_bins,
                   n_sig_activity=int((c_act.fdr <= 0.1).sum()),
                   n_sig_usage=int((c_use.fdr <= 0.1).sum()),
                   n_sig_usage_centred=int((c_usec.fdr <= 0.1).sum()),
                   n_sig_activity_centred=int((c_actc.fdr <= 0.1).sum()),
                   n_sig_w=int((c_w.fdr <= 0.1).sum())),
              open(os.path.join(a.out, "run_record.json"), "w"), indent=2)
    print(f"\nwrote {a.out}: program_top_genes.csv program_annotation.csv "
          f"activity_domain_sample.csv activity_domain_mean.csv usage_domain_mean.csv "
          f"contrast_activity.csv contrast_usage.csv contrast_w.csv w_by_sample.csv + 4 heatmaps")


if __name__ == "__main__":
    main()
