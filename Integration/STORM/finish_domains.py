#!/usr/bin/env python
from __future__ import annotations

import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Utils")   # plot_domains.py, compose_domains.py


def sh(args):
    print("$ " + " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u"] + args, cwd=UTILS, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="fitted run directory")
    p.add_argument("--cells", required=True)
    p.add_argument("--composition-root", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--target-n", type=int, default=None)
    a = p.parse_args()

    chosen = json.load(open(os.path.join(a.run, "chosen_resolution_QH_sccg.json")))
    r = chosen["resolution_str"]
    clusters = f"bin_clusters_QH_sccg_res{r}.csv"
    sh(["plot_domains.py", "--run", a.run, "--clusters", clusters,
        "--out", os.path.join(a.run, f"domains_QH_sccg_res{r}.png")])
    comp = os.path.join(a.composition_root, f"res{r}")
    sh(["compose_domains.py", "--cells", a.cells, "--storm", a.run,
        "--storm-clusters", clusters, "--out", comp])

    dec = dict(resolution=chosen["resolution"], resolution_str=r,
               n_domains=int(chosen["metrics"]["n_clusters"]), target=a.target_n,
               rule=chosen["rule"], clusters_file=os.path.join(a.run, clusters),
               composition=comp, metrics=chosen["metrics"],
               refined_resolutions=chosen.get("refined_resolutions", []))
    os.makedirs(os.path.dirname(os.path.abspath(a.decision)), exist_ok=True)
    json.dump(dec, open(a.decision, "w"), indent=1, default=str)
    print(f"\nresolution {r}: {dec['n_domains']} domains ({dec['rule']}) -> {a.decision}")
    if a.target_n is not None and dec["n_domains"] != a.target_n:
        print(f"NOTE: the prior asked for {a.target_n} domains; the closest reachable "
              f"count was {dec['n_domains']}", flush=True)


if __name__ == "__main__":
    main()
