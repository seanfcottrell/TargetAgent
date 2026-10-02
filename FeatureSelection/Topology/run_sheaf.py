from __future__ import annotations

import argparse, json, os, sys, time, types
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)   # PSL.py and SignificanceRankingPipeline.py live here

if "tqdm" not in sys.modules:
    def _progress(it, total=None, desc="", unit="", **_k):
        t0, n = time.time(), 0
        for x in it:
            n += 1
            if n == 1 or n % 25 == 0 or n == total:
                el = time.time() - t0
                rate = n / el if el else 0.0
                eta = (total - n) / rate if rate and total else 0.0
                print(f"    {desc}: {n}/{total}  {el:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
            yield x
    _m = types.ModuleType("tqdm"); _m.tqdm = _progress
    sys.modules["tqdm"] = _m

from SignificanceRankingPipeline import run as sheaf_run   # noqa: E402
from units import networks, tag, load_expr                  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--charges", required=True)
    p.add_argument("--expr", required=True)
    p.add_argument("--ppi", required=True, help="CSV gene1,gene2,combined_score")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--score-threshold", type=float, default=400.0)
    p.add_argument("--n-scales", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--weight-transform", default=None, choices=[None, "rank"],
                   help="'rank' replaces 1-|rho| by its quantile rank rescaled to "
                        "[eps,1]; single-nucleus |rho| is compressed into [0,0.2] so "
                        "the untransformed filtration spans almost nothing and the "
                        "sheaf collapses to weighted degree (rho -0.93..-0.98)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    report = {}
    print(f"weight transform: {a.weight_transform or 'none (raw 1-|rho|)'}", flush=True)
    for d, pr in networks(a.charges, a.domains):
        t0 = time.time()
        unit = tag(d, pr)
        ch = pd.read_csv(os.path.join(a.charges, f"{unit}_charge.csv"))
        genes, M = load_expr(a.expr, d, pr)                       # genes x nuclei
        expr_df = pd.DataFrame(M, index=genes)
        charge = ch.set_index("gene").charge.reindex(genes)
        print(f"\n=== {unit}: {len(genes)} nodes x {M.shape[1]:,} nuclei", flush=True)

        res = sheaf_run(genes, a.ppi, expr_df, charge.to_dict(),
                        score_threshold=a.score_threshold, case="upper",
                        corr="pearson", eps=a.eps, n_scales=a.n_scales,
                        max_workers=a.workers, drop_isolates=True,
                        weight_transform=a.weight_transform)

        rank = pd.DataFrame(res["ranking"], columns=["node", "spectral_shift"])
        rank.insert(0, "rank", np.arange(1, len(rank) + 1))
        # build_gcn normalises node names with case="upper", so a symbol such as
        # C9orf78 comes back as C9ORF78 and a join on the raw name silently
        # yields NaN -- which then breaks the astype(int) below. Join on an
        # upper-case key and take the original symbol from the charge table.
        ch2 = ch.copy(); ch2["_key"] = ch2.gene.str.upper()
        rank["_key"] = rank.node.str.upper()
        rank = rank.merge(ch2[["_key", "gene", "charge", "abs_charge", "loading"]],
                          on="_key", how="left")
        missing = rank.charge.isna().sum()
        assert not missing, f"{unit}: {missing} graph nodes have no charge row"
        rank = rank[["rank", "gene", "spectral_shift", "charge", "abs_charge", "loading"]]
        rank["charge_rank"] = rank.charge.abs().rank(ascending=False).astype(int)
        rank.to_csv(os.path.join(a.out, f"{unit}_sheaf_rank.csv"), index=False)
        np.save(os.path.join(a.out, f"{unit}_scores.npy"), res["scores"])

        G = res["graph"]
        rho = float(np.corrcoef(rank["rank"], rank.charge_rank)[0, 1]) if len(rank) > 2 else float("nan")
        report.setdefault(d, {})[pr] = dict(
            n_nodes_scored=len(rank), graph_nodes=G.number_of_nodes(),
            graph_edges=G.number_of_edges(), n_scales=len(res["radii"]),
            stable_top10=res["stable"],
            rank_vs_charge_rank_pearson=round(rho, 3),
            top20=rank.gene.head(20).tolist(),
            seconds=round(time.time() - t0, 1))
        print(f"  {len(rank)} scored; top: {', '.join(rank.gene.head(10))}", flush=True)
        print(f"  rank vs charge-rank correlation {rho:+.3f} "
              f"(low means topology, not charge magnitude, is doing the ranking)", flush=True)
        json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)

    print("\n===== summary =====")
    flat = {tag(d, pr): v for d, x in report.items() for pr, v in x.items()}
    print(pd.DataFrame(flat).T.to_string())


if __name__ == "__main__":
    main()
