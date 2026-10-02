from __future__ import annotations

import argparse
import json
import os, sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

LIBS = ["GO_Biological_Process_2023", "Reactome_2022", "KEGG_2021_Human",
        "PanglaoDB_Augmented_2021", "DisGeNET"]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import RUN_ROOT as BASE  # dataset outputs live under runs/<dataset>/


def load_lib(name, cache):
    path = os.path.join(cache, f"{name}.txt")
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={name}"
        urllib.request.urlretrieve(url, path)
    lib = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        genes = {g.split(",")[0].upper() for g in parts[2:] if g}
        lib[parts[0]] = genes
    return lib


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.empty(n)
    r[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(r, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="fitted run directory")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=os.path.join(BASE, "reference", "enrichr"))
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--max-terms", type=int, default=6)
    p.add_argument("--fdr", type=float, default=0.1)
    a = p.parse_args()
    B = np.load(os.path.join(a.run, "B_gene_loadings.npy"))
    genes = pd.read_csv(os.path.join(a.run, "genes.csv")).iloc[:, 0].astype(str).str.upper().to_numpy()
    R = B.shape[1]
    libs = {n: load_lib(n, a.cache) for n in LIBS}
    rows, summary = [], {}
    for r in range(R):
        top = set(genes[np.argsort(-B[:, r])[: a.top]])
        summary[f"F{r}"] = {}
        for name, lib in libs.items():
            universe = set().union(*lib.values())
            bg = set(genes) & universe
            fg = top & universe
            if len(fg) < 3:
                summary[f"F{r}"][name] = dict(n_top_in_library=len(fg), terms=[])
                continue
            recs = []
            for term, gs in lib.items():
                K = len(gs & bg)
                if K < 3:
                    continue
                k = len(gs & fg)
                if k < 2:
                    continue
                pval = hypergeom.sf(k - 1, len(bg), K, len(fg))
                recs.append(dict(term=term, k=k, K=K, p=pval, genes=sorted(gs & fg)))
            if not recs:
                summary[f"F{r}"][name] = dict(n_top_in_library=len(fg), terms=[])
                continue
            df = pd.DataFrame(recs); df["fdr"] = bh(df.p); df = df.sort_values("p")
            df["program"] = f"F{r}"; df["library"] = name; rows.append(df)
            sig = df[df.fdr <= a.fdr].head(a.max_terms)
            summary[f"F{r}"][name] = dict(
                n_top_in_library=len(fg), n_background=len(bg),
                terms=[dict(term=t.term, overlap=f"{t.k}/{t.K}", fdr=round(float(t.fdr), 4),
                            genes=t.genes) for t in sig.itertuples()])
    pd.concat(rows).to_csv(os.path.join(a.out, "program_enrichment.csv"), index=False)
    json.dump(dict(method=f"hypergeometric ORA of top-{a.top} loading genes per programme, background = fit HVGs present in each library, BH per library",
                   libraries=LIBS, programmes=summary),
              open(os.path.join(a.out, "program_enrichment.json"), "w"), indent=1)
    for pg, d in summary.items():
        hits = [f"{lib.split('_')[0]}: " + "; ".join(t["term"][:50] for t in v["terms"][:2]) for lib, v in d.items() if v["terms"]]
        print(f"{pg}: " + (" | ".join(hits) if hits else "no term at FDR <= %.2f" % a.fdr))
    print(f"wrote {a.out}/program_enrichment.{{csv,json}}")


if __name__ == "__main__":
    main()
