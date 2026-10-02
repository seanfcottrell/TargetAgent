#!/usr/bin/env python
"""
Ontological identity for each candidate gene: symbol -> reviewed UniProt entry.

This is the ONLY identifier resolution that happens ahead of the panel, and it
deliberately stops at UniProt. The accession is a stable ontological anchor;
a bioactivity-database target id is not, and must not be baked into a packet:

  * Database target records change between releases, so a packet carrying one
    silently goes stale.
  * Pinning a single target record hides the ambiguity a target may genuinely
    have — isoforms, complexes, multi-component targets — instead of leaving it
    where it can be reasoned about.
  * Assessing chemistry is the tractability agent's job. Handing it a
    pre-computed answer replaces the judgement the panel exists to make.

So packets carry the accession; the agent queries by accession at run time.

Resolution uses `gene_exact` against reviewed human entries, never a free-text
search. Symbol search is the failure mode this file exists to avoid: a fuzzy
query for a homeobox gene returns an unrelated enzyme with a large chemistry
corpus, which would invert every downstream judgement about it. When a symbol
does not resolve exactly, that is recorded as unresolved rather than guessed --
an unresolved identity is a usable fact, a wrong one is not.

    python resolve_identity.py --packets packets_targets --out identity.tsv
"""
from __future__ import annotations

import argparse, json, os, time, urllib.error, urllib.parse, urllib.request
import pandas as pd

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"


def _get(url, tries=4):
    last = None
    for wait in (0, 5, 15, 40)[:tries]:
        if wait:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            last = e
        except Exception as e:
            last = e
    raise RuntimeError(f"{url} failed: {last}")


def resolve(sym):
    """Exact gene-name match, reviewed, human. Returns (accession, name, n_matches)."""
    q = f"gene_exact:{sym} AND organism_id:9606 AND reviewed:true"
    r = _get(UNIPROT + "?" + urllib.parse.urlencode(
        {"query": q, "fields": "accession,protein_name,gene_names", "format": "json", "size": 5}))
    res = r.get("results", [])
    if not res:
        return None, None, 0
    e = res[0]
    desc = e.get("proteinDescription", {})
    name = (desc.get("recommendedName", {}).get("fullName", {}).get("value")
            or (desc.get("submissionNames") or [{}])[0].get("fullName", {}).get("value"))
    return e["primaryAccession"], name, len(res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packets", default=None, help="packet dir; genes read from domain_*.json")
    p.add_argument("--genes", nargs="*", default=None, help="explicit gene list instead")
    p.add_argument("--out", required=True)
    p.add_argument("--sleep", type=float, default=0.2)
    a = p.parse_args()

    if a.genes:
        genes = sorted({g.upper() for g in a.genes})
    elif a.packets:
        genes = set()
        for f in sorted(os.listdir(a.packets)):
            if f.startswith("domain_") and f.endswith(".json"):
                genes |= {c["gene"] for c in json.load(open(os.path.join(a.packets, f)))["candidates"]}
        genes = sorted(genes)
    else:
        raise SystemExit("give --packets or --genes")

    cache = {}
    if os.path.isfile(a.out):
        cache = {r.gene: r for r in pd.read_csv(a.out, sep="\t").itertuples()}

    rows = []
    for i, g in enumerate(genes, 1):
        if g in cache:
            rows.append({k: v for k, v in cache[g]._asdict().items() if k != "Index"}); continue
        try:
            acc, name, n = resolve(g)
            rows.append(dict(gene=g, uniprot=acc, protein=name, n_exact_matches=n,
                             status="resolved" if acc else "unresolved"))
        except Exception as e:
            rows.append(dict(gene=g, uniprot=None, protein=None, n_exact_matches=0,
                             status=f"error:{type(e).__name__}"))
        time.sleep(a.sleep)
        if i % 25 == 0:
            print(f"  {i}/{len(genes)}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(a.out, sep="\t", index=False)
    print(f"{len(df)} genes -> {a.out}: "
          f"{(df.status=='resolved').sum()} resolved, "
          f"{(df.status=='unresolved').sum()} unresolved, "
          f"{df.status.str.startswith('error').sum()} errors")
    amb = df[(df.status == "resolved") & (df.n_exact_matches > 1)]
    if len(amb):
        print(f"  {len(amb)} symbols matched more than one reviewed entry "
              f"(the agent is told, and should treat identity as uncertain): "
              f"{', '.join(amb.gene.head(10))}")


if __name__ == "__main__":
    main()
