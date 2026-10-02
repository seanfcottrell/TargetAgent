#!/usr/bin/env python
"""
STRING edge list for the sheaf, as a CSV `build_gcn` can read.

The sheaf uses a lower score cut than the fit's gene graph: a gene with no edge
has a leave-one-out spectral shift of exactly 0 and cannot be ranked, so the
network needs to be dense enough to leave few genes isolated.

Reuses `fetch_string` from the STORM runner, so the result is cached under
`cache/ppi_cache/` keyed by version, species, score cut and a hash of the gene
set: a different gene set gets a different cache file rather than silently
reusing this one. Requires network on the first call; afterwards it is a cache
read. `build_gcn` expects columns gene1, gene2, combined_score, while the cache
is STRING's TSV with protein1/protein2, so this renames and rewrites.

    python fetch_ppi.py --genes <fit>/genes.csv --min-score 400 --out <run>/sheaf/string400.csv
"""
from __future__ import annotations

import argparse, os, sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "Integration", "STORM", "STORM"))
from storm_chain_run import fetch_string   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import cpath  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", required=True, help="genes.csv of the fit")
    p.add_argument("--min-score", type=float, default=400.0)
    p.add_argument("--cache", default=cpath("cache","ppi_cache"))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    g = pd.read_csv(a.genes).gene
    df = fetch_string(pd.Index(g), cache_dir=a.cache, min_score=a.min_score)
    (df[["protein1", "protein2", "combined_score"]]
       .rename(columns={"protein1": "gene1", "protein2": "gene2"})
       .to_csv(a.out, index=False))
    print(f"{len(df)} edges at score >= {int(a.min_score)} over {len(g)} genes -> {a.out}")


if __name__ == "__main__":
    main()
