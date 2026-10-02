#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "STORM", "STORM"))   # the STORM package (ChainGraph)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))              # repo root FIRST (Utils)
from ChainGraph import ChainConfig, couple_edge   # noqa: E402
from Utils.domain_utils import read_meta          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir with bin_meta.csv")
    p.add_argument("--niches", default=None, help="default <run>/niches")
    p.add_argument("--out", default=None, help="default <run>/couplings")
    p.add_argument("--pairs", choices=["cross", "lot", "all"], default="cross")
    p.add_argument("--n-niches", type=int, default=800,
                   help="only used for the ChainConfig dense-size check")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    ndir = a.niches or os.path.join(a.run, "niches")
    out = a.out or os.path.join(a.run, "couplings")
    os.makedirs(out, exist_ok=True)
    meta = read_meta(a.run, None) if False else pd.read_csv(
        os.path.join(a.run, "bin_meta.csv"), keep_default_na=False,
        dtype={"section": str, "stage": str, "lot": str})
    info = meta.groupby("section").agg(stage=("stage", "first"), lot=("lot", "first"))
    niche = {}
    for f in glob.glob(os.path.join(ndir, "*.npz")):
        s = os.path.basename(f)[:-4]
        if s.endswith("_counts"):
            continue
        z = np.load(f)
        dens = z["density"]
        niche[s] = dict(emb=z["emb"], coords=z["coords"], mass=z["mass"],
                        density=None if np.all(np.isnan(dens)) else dens)
    secs = sorted(niche)
    ad_secs = [s for s in secs if info.loc[s, "stage"] != "NA"]
    ctrl = [s for s in secs if info.loc[s, "stage"] == "NA"]
    if a.pairs == "cross":
        pairs = [(x, c) for x in ad_secs for c in ctrl]
    elif a.pairs == "lot":
        pairs = [(x, c) for x in ad_secs for c in ctrl if info.loc[x, "lot"] == info.loc[c, "lot"]]
    else:
        pairs = list(itertools.combinations(secs, 2))
    cfg = ChainConfig(n_niches_per_sample=a.n_niches, verbose=False)
    print(f"{len(pairs)} pairs ({a.pairs}); niches per section "
          f"{ {s: len(v['mass']) for s, v in niche.items()} }", flush=True)
    t0 = time.time()
    for src, dst in pairs:
        f = os.path.join(out, f"{src}__{dst}.npz")
        if os.path.exists(f) and not a.overwrite:
            print(f"  {src}->{dst}: exists, skipped"); continue
        r = couple_edge(niche[src], niche[dst], "cross", cfg)
        np.savez_compressed(f, G=np.asarray(r["G"], np.float32), tau=np.float32(r["tau"]),
                            kind="posthoc",
                            unmatched_i=np.asarray(r["unmatched_i"], np.float32),
                            unmatched_j=np.asarray(r["unmatched_j"], np.float32),
                            concentration=np.asarray(r["concentration"], np.float32))
        print(f"  {src}->{dst}: tau={r['tau']:.3g} unmatched src={r['mean_unmatched_i']:.1%} "
              f"dst={r['mean_unmatched_j']:.1%} conc={r['mean_concentration']:.2f} "
              f"{'' if r['tau_info']['identified'] else 'TAU UNIDENTIFIED'} ({time.time()-t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
