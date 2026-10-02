from __future__ import annotations

import argparse, os, sys
import pandas as pd, networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)   # PSL.py and SignificanceRankingPipeline.py live here
from SignificanceRankingPipeline import build_gcn   # noqa: E402
from units import networks, tag, load_expr          # noqa: E402


def load_gmt(path):
    sets = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2:
                sets[parts[0]] = {g.upper() for g in parts[2:] if g}
    return sets


def main():
    """Per programme network: each gene's position in the graph the sheaf ranked
    (degree, weighted degree, betweenness, clustering) next to its spectral rank,
    and its membership of the programme's enriched pathways. This is what the
    target packets report about a gene on the topology axis."""
    p = argparse.ArgumentParser()
    p.add_argument("--charges", required=True); p.add_argument("--expr", required=True)
    p.add_argument("--out", required=True, help="the directory run_sheaf.py wrote")
    p.add_argument("--enrich", required=True, help="program_enrichment.csv")
    p.add_argument("--gmt", required=True, help="directory of the Enrichr libraries")
    p.add_argument("--ppi", required=True)
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--weight-transform", default=None, choices=[None, "rank"],
                   help="MUST match the transform run_sheaf.py used for this --out, "
                        "or wdegree/betweenness are computed on a different graph "
                        "than the spectral shift was")
    p.add_argument("--fdr", type=float, default=0.05)
    a = p.parse_args()

    enr = pd.read_csv(a.enrich)
    gmts = {}

    for d, pr in networks(a.charges, a.domains):
        unit = tag(d, pr)
        rp = os.path.join(a.out, f"{unit}_sheaf_rank.csv")
        if not os.path.isfile(rp):
            print(f"[{unit}] no ranking yet, skipped", flush=True); continue
        rank = pd.read_csv(rp)
        genes, M = load_expr(a.expr, d, pr)
        ch = pd.read_csv(os.path.join(a.charges, f"{unit}_charge.csv")).set_index("gene").charge
        G = build_gcn(genes, a.ppi, pd.DataFrame(M, index=genes), ch.to_dict(),
                      score_threshold=400.0, case="upper",
                      weight_transform=a.weight_transform)
        for u, v, dd in G.edges(data=True):
            dd["sim"] = 1.0 - dd["weight"]

        cent = pd.DataFrame({
            "degree": dict(G.degree()),
            "wdegree": {n: sum(G[n][m]["sim"] for m in G[n]) for n in G},
            "betweenness": nx.betweenness_centrality(G, weight="weight"),
            "clustering": nx.clustering(G, weight="sim"),
        })
        cent.index.name = "gene"
        # cent is indexed by graph node name (upper case); rank.gene carries the
        # original symbol, so join on an upper-case key.
        cent = cent.reset_index(); cent["_key"] = cent.gene.str.upper()
        r = rank.assign(_key=rank.gene.str.upper()).merge(
            cent.drop(columns=["gene"]), on="_key", how="left").drop(columns=["_key"])

        # Pathways of the programme that built this network.
        terms = enr[(enr.program == pr) & (enr.fdr < a.fdr)]
        node_up = {g.upper() for g in r.gene}
        rows = []
        for _, t in terms.iterrows():
            lib = os.path.join(a.gmt, f"{t.library}.txt")
            if not os.path.isfile(lib):
                continue
            gmts.setdefault(t.library, load_gmt(lib))
            members = gmts[t.library].get(t.term, set()) & node_up
            if len(members) < 3:
                continue
            sub = G.subgraph([n for n in G if n.upper() in members])
            deg = dict(sub.degree())
            for g in members:
                rows.append(dict(gene=g, program=t.program, library=t.library, term=t.term,
                                 fdr=t.fdr, degree_in_pathway=deg.get(g, 0),
                                 pathway_size_in_nodes=len(members)))
        pw = pd.DataFrame(rows)
        if len(pw):
            pw = pw.merge(r[["gene", "rank"]], on="gene", how="left")
            pw.sort_values(["rank", "fdr"]).to_csv(
                os.path.join(a.out, f"{unit}_pathway_membership.csv"), index=False)

        r.sort_values("rank").to_csv(os.path.join(a.out, f"{unit}_sheaf_annotated.csv"), index=False)
        print(f"[{unit}] {len(r)} genes annotated, {pw.gene.nunique() if len(pw) else 0} in an "
              f"enriched pathway of {pr}", flush=True)

    print(f"\nwrote {a.out}: domain<d>_F<r>_sheaf_annotated.csv, domain<d>_F<r>_pathway_membership.csv")


if __name__ == "__main__":
    main()
