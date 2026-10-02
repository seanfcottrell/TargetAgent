from __future__ import annotations

import argparse, json, os, sys
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)   # PSL.py and SignificanceRankingPipeline.py live here
from SignificanceRankingPipeline import build_gcn, make_radii   # noqa: E402
from units import networks, tag, load_expr                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--charges", required=True); p.add_argument("--expr", required=True)
    p.add_argument("--out", required=True); p.add_argument("--ppi", default="string400.csv")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--weight-transform", default="rank", choices=[None, "rank"])
    p.add_argument("--n-scales", type=int, default=20)
    p.add_argument("--topn", type=int, default=40)
    p.add_argument("--min-frac", type=float, default=1.0,
                   help="fraction of its present-scales a gene must be top-N at (1.0 = every scale)")
    a = p.parse_args()

    summary = {}
    for d, pr in networks(a.charges, a.domains):
        unit = tag(d, pr)
        genes, M = load_expr(a.expr, d, pr)
        ch = pd.read_csv(os.path.join(a.charges, f"{unit}_charge.csv")).set_index("gene").charge
        G = build_gcn(genes, a.ppi, pd.DataFrame(M, index=genes), ch.to_dict(),
                      score_threshold=400.0, case="upper",
                      weight_transform=a.weight_transform)
        radii = make_radii(G, n_scales=a.n_scales)
        order = list(G.nodes())                       # the row order run_sheaf scored in
        S = np.load(os.path.join(a.out, f"{unit}_scores.npy"))
        assert S.shape == (len(order), len(radii)), (S.shape, len(order), len(radii))

        hits = pd.Series(0, index=order); present = pd.Series(0, index=order)
        for s, v in enumerate(radii):
            keep = [(u, w) for u, w, dd in G.edges(data=True) if dd["weight"] <= v]
            wdeg = pd.Series(0.0, index=order)
            for u, w in keep:
                sim = 1.0 - G[u][w]["weight"]
                wdeg[u] += sim; wdeg[w] += sim
            live = wdeg > 0
            if live.sum() < 10:
                continue
            x = wdeg[live].rank().values
            y = pd.Series(S[:, s], index=order)[live].rank().values
            b, c = np.polyfit(x, y, 1)
            resid = pd.Series(y - (c + b * x), index=wdeg[live].index)
            present[live.index[live]] += 1
            hits[resid.nlargest(min(a.topn, len(resid))).index] += 1

        frac = (hits / present.replace(0, np.nan))
        res = pd.DataFrame({"gene": order, "scales_top": hits.values,
                            "scales_present": present.values, "frac": frac.values})
        res = res[res.scales_present > 0].sort_values(["frac", "scales_top"], ascending=False)
        res.to_csv(os.path.join(a.out, f"{unit}_residual_per_scale.csv"), index=False)
        sel = res[res.frac >= a.min_frac]
        summary.setdefault(d, {})[pr] = dict(n_scales=len(radii), topn=a.topn,
                                             n_intersect=len(sel), genes=sel.gene.tolist())
        print(f"=== {unit}: top-{a.topn} residual at EVERY scale it appears in ===", flush=True)
        print(f"  {len(sel)} genes: " + (", ".join(sel.gene) if len(sel) else "(empty)"), flush=True)
        nxt = res[(res.frac < a.min_frac) & (res.frac >= 0.8)]
        print(f"  at >=80% of scales, {len(nxt)} more: " + ", ".join(nxt.gene.head(15)), flush=True)

    json.dump(summary, open(os.path.join(a.out, "residual_per_scale.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
