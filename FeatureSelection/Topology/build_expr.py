from __future__ import annotations

import argparse, json, os
import sys
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from units import networks, tag   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--charges", required=True, help="dir of domain<d>_charge.csv")
    p.add_argument("--cells", required=True, help="cells_with_domain.csv.gz")
    p.add_argument("--indir", required=True, help="cellbin h5ads")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    nets = {d: [pr for dd, pr in networks(a.charges, [d])] for d in a.domains}
    a.domains = [d for d in a.domains if nets[d]]          # a domain with no network has no matrix
    net_nodes = {(d, pr): pd.read_csv(os.path.join(a.charges, f"{tag(d, pr)}_charge.csv")).gene.tolist()
                 for d in a.domains for pr in nets[d]}
    nodes = {}                                             # union per domain, first-seen order
    for d in a.domains:
        seen = []
        for pr in nets[d]:
            seen += [g for g in net_nodes[(d, pr)] if g not in set(seen)]
        nodes[d] = seen

    cells = pd.read_csv(a.cells, keep_default_na=False, low_memory=False,
                        usecols=["cell_id", "section", "domain"],
                        dtype={"cell_id": str, "domain": str})
    cells = cells[cells.domain.isin(a.domains)]
    secs = sorted(cells.section.unique())
    print(f"{len(cells):,} QC-passed nuclei in domains {a.domains} across {len(secs)} sections", flush=True)

    # Gene space common to every section, so columns align.
    common = None
    for sec in secs:
        v = set(ad.read_h5ad(os.path.join(a.indir, f"{sec}_cellbin.h5ad"), backed="r").var_names)
        common = v if common is None else (common & v)
    keep_genes = {d: [g for g in nodes[d] if g in common] for d in a.domains}
    for d in a.domains:
        drop = len(nodes[d]) - len(keep_genes[d])
        print(f"  domain {d}: {len(keep_genes[d])} of {len(nodes[d])} node genes (union of "
              f"{', '.join(nets[d])}) in the common space" + (f" ({drop} dropped)" if drop else ""),
              flush=True)

    chunks = {d: [] for d in a.domains}
    meta = {d: [] for d in a.domains}
    for sec in secs:
        A = ad.read_h5ad(os.path.join(a.indir, f"{sec}_cellbin.h5ad"))
        idx = pd.Index(A.obs_names)
        sub = cells[cells.section == sec]
        for d in a.domains:
            ids = sub.loc[sub.domain == d, "cell_id"].values
            pos = idx.get_indexer(ids)
            ok = pos >= 0
            ids, pos = ids[ok], pos[ok]
            if pos.size == 0:
                continue
            X = A.X[pos]                                     # nuclei x all genes, raw counts
            lib = np.asarray(X.sum(1)).ravel()               # FULL library per nucleus
            lib[lib == 0] = 1.0
            gpos = A.var_names.get_indexer(keep_genes[d])
            assert (gpos >= 0).all(), f"{sec}/domain {d}: node gene missing after intersection"
            M = X[:, gpos]
            M = M.toarray().astype(np.float32) if sp.issparse(M) else np.asarray(M, dtype=np.float32)
            M = np.log1p(M / lib[:, None] * 1e4)             # log1p CP10K
            M -= M.mean(0, keepdims=True)                    # centre WITHIN this section
            chunks[d].append(M)
            meta[d].append(pd.DataFrame({"cell_id": ids, "section": sec}))
        del A
        print(f"  {sec}: " + " ".join(f"d{d} {len(meta[d][-1]) if meta[d] and meta[d][-1].section.iloc[0]==sec else 0}"
                                      for d in a.domains), flush=True)

    report = {}
    for d in a.domains:
        M = np.vstack(chunks[d])
        md = pd.concat(meta[d], ignore_index=True)
        assert len(md) == M.shape[0], (len(md), M.shape)
        np.save(os.path.join(a.out, f"domain{d}_expr.npy"), M.T)      # genes x nuclei
        md.to_csv(os.path.join(a.out, f"domain{d}_nuclei.csv"), index=False)
        pd.DataFrame({"gene": keep_genes[d]}).to_csv(
            os.path.join(a.out, f"domain{d}_genes.csv"), index=False)
        flat = int((M.std(0) == 0).sum())
        per_net = {}
        for pr in nets[d]:
            ng = [g for g in net_nodes[(d, pr)] if g in common]
            pd.DataFrame({"gene": ng}).to_csv(os.path.join(a.out, f"{tag(d, pr)}_genes.csv"), index=False)
            per_net[pr] = dict(n_genes=len(ng), dropped=len(net_nodes[(d, pr)]) - len(ng))
        report[d] = dict(n_nuclei=int(M.shape[0]), n_genes=int(M.shape[1]), flat_genes=flat,
                         networks=per_net,
                         sections=md.section.value_counts().sort_index().to_dict())
        print(f"[domain {d}] {M.shape[0]:,} nuclei x {M.shape[1]} genes (union); flat genes {flat}; "
              + " ".join(f"{pr} {v['n_genes']}" for pr, v in per_net.items()), flush=True)

    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"\nwrote {a.out}: domain<d>_expr.npy (union genes x nuclei, donor-centred), "
          f"domain<d>_genes.csv, domain<d>_F<r>_genes.csv, domain<d>_nuclei.csv, report.json")


if __name__ == "__main__":
    main()
