#!/usr/bin/env python
"""
Deterministic packets for the TARGET panel.

One packet per (gene, domain) candidate, carrying the three axes with every
number tied to its source field, plus the context the direction and tractability
agents need. Nothing here scores or filters on biology -- it assembles evidence.

CANDIDATE POOL = union, per domain, of
  * NEBULA hits (FDR < 0.05 and |log2FC| > 1) in any stratum      -> support "N"
    (default contrast, or a designed contrast comparing case with control
    that has the domain on side_a; within-group designs are context only)
  * the sheaf per-scale degree-residual intersection of ANY of the
    domain's programme networks (one network per upheld programme,
    FeatureSelection/Topology/units.py)                          -> support "S"
  * the top 20 genes by |charge|                                  -> support "C"
Union, not intersection: the axes are near-orthogonal on this cohort (NEBULA and
the residual share zero genes in domains 2, 4 and 5), because they ask different
questions of different units -- does the gene's expression move in nuclei; does
it define a programme whose usage moves in bins; is it structurally central to
that programme's network. A convergence rule would leave two usable genes.
135 of 166 packets rest on a single axis, and that is expected, not a defect.

The topology block is PER PROGRAMME NETWORK: a domain with two upheld
programmes has two networks, and a gene's structural position is reported in
each it is a node of, never merged across them.

PROCESS DIRECTION is precomputed per domain and is the anchor for the direction
agent: which programmes built this domain's charge, which way each moved, and
what the programme is. The therapeutic goal follows from the programme's
movement (a programme whose usage falls -> restore that process), and the
gene's own expression direction is carried SEPARATELY as a consistency check,
never as the recommendation. See the direction contract in prompts/system/.

    python build_target_packets.py --run <fit> --deg <run>/deg_out --sheaf <run>/sheaf \
        --progact <run>/program_activity --composition <run>/composition/res<r> \
        --domains <shortlisted domains> --out <run>/target_packets
"""
from __future__ import annotations

import argparse, ast, glob, json, os, sys, collections
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "FeatureSelection", "Topology"))
from units import tag as net_tag   # noqa: E402

def read_or_empty(path, cols):
    """A domain whose programmes were not charged has no sheaf outputs; it still
    carries its differential-expression candidates, so a missing file is an empty
    axis, not an error."""
    return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame(columns=cols)


def load_or(path, default):
    return json.load(open(path)) if os.path.isfile(path) else default


def jsonable(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return None if np.isnan(x) else round(float(x), 6)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--deg", required=True)
    p.add_argument("--sheaf", required=True); p.add_argument("--progact", required=True)
    p.add_argument("--composition", required=True)
    p.add_argument("--identity", default=None,
                   help="resolve_identity.py output: gene -> UniProt accession. Packets carry "
                        "the ACCESSION only; bioactivity is looked up by the tractability agent "
                        "at run time via an accession-addressed tool, never baked in here.")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--top-charge", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ident = {}
    if a.identity and os.path.isfile(a.identity):
        idf = pd.read_csv(a.identity, sep="\t")
        ident = {r.gene: r for r in idf.itertuples()}
    prov = load_or(os.path.join(a.sheaf, "charges", "provenance.json"), {"domains": {}})
    resid = load_or(os.path.join(a.sheaf, "out_rank", "residual_per_scale.json"), {})
    ann = pd.read_csv(os.path.join(a.progact, "program_annotation.csv"), index_col=0)
    enr = pd.read_csv(os.path.join(a.progact, "program_enrichment.csv"))
    comp = pd.read_csv(os.path.join(a.composition, "composition_by_domain.csv"),
                       keep_default_na=False)

    # ---- DEG, all strata, all domains (for cross-domain recurrence) ----------
    deg = collections.defaultdict(dict)          # domain -> gene -> {stratum: (l2fc, padj, hit)}
    for f in sorted(glob.glob(os.path.join(a.deg, "deg_domain*.csv"))):
        b = os.path.basename(f)[len("deg_domain"):-4]; dom, ct = b.split("_", 1)
        d = pd.read_csv(f)
        for _, r in d[d.hit].iterrows():
            deg[dom].setdefault(r.gene.upper(), {})[ct] = (r.log2FC, r.padj)
    # ---- designed contrasts (domain panel, deg_contrast_<id>.csv): attached to the
    # domains on their side_a. A hit enters the candidate pool only when the
    # contrast compares case with control (disease_informative); a comparison
    # within one group describes regional difference and is carried as context.
    spec_path = os.path.join(a.deg, "designed_contrasts.json")
    designed = {c["id"]: c for c in load_or(spec_path, [])}
    dz = collections.defaultdict(dict)           # domain -> contrast id -> table
    for cid, c in designed.items():
        f = os.path.join(a.deg, f"deg_contrast_{cid}.csv")
        if not os.path.isfile(f):
            continue                             # contrast failed to fit; summary.csv says why
        t = pd.read_csv(f)
        t["g"] = t.gene.astype(str).str.upper()
        for dom in c["side_a"]["domains"]:
            dz[str(dom)][cid] = t.set_index("g")
    def designed_block(dom, g):
        out = {}
        for cid, t in dz.get(dom, {}).items():
            if g in t.index:
                c, r = designed[cid], t.loc[g]
                out[cid] = dict(question=c.get("question"), test=c["test"],
                                side_a=c["side_a"], side_b=c["side_b"], cell_type=c["type"],
                                disease_informative=bool(c.get("disease_informative", True)),
                                log2FC=jsonable(r.log2FC), padj=jsonable(r.padj), hit=bool(r.hit))
        return out

    recur = collections.defaultdict(dict)
    for dom, gs in deg.items():
        for g, strata in gs.items():
            recur[g][dom] = sorted(strata)

    # Cohort facts are DERIVED from the tables, never written here: this builder
    # must produce correct packets for any dataset the upstream stages ran on.
    # Dataset-specific context (known nuisance gene classes, batch structure,
    # what the disease is) belongs in the USER turn via prompts/dataset/<name>.md,
    # and the generic semantics of the three axes belong in the system prompts.
    # Derive only from the domains under assessment. The composition table also
    # carries a "no_tile" pseudo-domain for nuclei outside any tile, whose counts
    # do not describe the cohort. keep_default_na=False also leaves blanks as ""
    # so these columns are object dtype: .max() on them compares strings.
    assessed = comp[comp.domain.astype(str).isin(a.domains)]
    n_case = int(pd.to_numeric(assessed.n_ad, errors="coerce").max())
    n_ctrl = int(pd.to_numeric(assessed.n_ctrl, errors="coerce").max())
    cell_types = sorted(assessed.type.unique())
    cohort = dict(
        n_case_samples=n_case, n_control_samples=n_ctrl,
        unit_of_replication="sample (one section per donor)",
        cell_types=cell_types,
        n_domains_assessed=len(a.domains),
        programmes_available=list(ann.index),
        programme_annotation={r: dict(dominant_cell_type=ann.loc[r, "annotation"],
                                      top5=ann.loc[r, "top5"]) for r in ann.index},
        candidate_pool_rule=(
            f"Per domain, the union of: genes called by the differential-expression "
            f"axis in any stratum of the default contrast or in any designed contrast "
            f"that compares case with control and has this domain on its side of "
            f"interest; genes in the residual set of any of the topology axis's programme "
            f"networks (one network per upheld programme); and the "
            f"top {a.top_charge} genes by |charge|. Union, not intersection: the axes ask "
            f"different questions of different measurement units. A candidate supported "
            f"by one axis is expected, not deficient."),
    )
    json.dump(cohort, open(os.path.join(a.out, "cohort.json"), "w"), indent=1)

    index = {}
    for dom in a.domains:
        dprov = prov["domains"].get(dom, {})
        progs, dvals = dprov.get("programmes", []), dprov.get("d", {})
        ch = read_or_empty(os.path.join(a.sheaf, "charges", f"domain{dom}_charge.csv"),
                           ["rank", "gene", "charge", "abs_charge", "cancellation"])
        ch["g"] = ch.gene.astype(str).str.upper()
        # One network per upheld programme (units.py): the topology axis is read per network.
        nets = list(dprov.get("networks", {}).keys())
        chn, sh, rs, pw = {}, {}, {}, {}
        for pr in nets:
            u = net_tag(dom, pr)
            chn[pr] = read_or_empty(os.path.join(a.sheaf, "charges", f"{u}_charge.csv"),
                                    ["rank", "gene", "charge", "abs_charge", "loading"])
            sh[pr] = read_or_empty(os.path.join(a.sheaf, "out_rank", f"{u}_sheaf_annotated.csv"),
                                   ["gene", "rank", "degree", "wdegree", "betweenness", "clustering"])
            rs[pr] = read_or_empty(os.path.join(a.sheaf, "out_rank", f"{u}_residual_per_scale.csv"),
                                   ["gene", "scales_top", "scales_present"])
            pw[pr] = read_or_empty(os.path.join(a.sheaf, "out_rank", f"{u}_pathway_membership.csv"),
                                   ["gene", "program", "library", "term", "fdr", "degree_in_pathway",
                                    "pathway_size_in_nodes"])
            for t in (chn, sh, rs, pw):
                t[pr]["g"] = t[pr].gene.astype(str).str.upper()
        resid_sets = {pr: set(resid.get(dom, {}).get(pr, {}).get("genes", [])) for pr in nets}
        sheaf_rep = load_or(os.path.join(a.sheaf, "out_rank", "report.json"), {}).get(dom, {})
        if not progs:
            print(f"[domain {dom}] no charged programme: differential-expression axis only", flush=True)

        top_charge = set(ch.g.head(a.top_charge))
        resid_set = set().union(*resid_sets.values()) if resid_sets else set()
        designed_hits = {g for cid, t in dz.get(dom, {}).items()
                         if designed[cid].get("disease_informative", True)
                         for g in t.index[t.hit.astype(bool)]}
        hit_set = set(deg[dom]) | designed_hits
        pool = sorted(top_charge | resid_set | hit_set)

        cshare = comp[comp.domain.astype(str) == dom]
        dom_comp = cshare.copy()
        dom_comp["mean_ctrl_f"] = pd.to_numeric(dom_comp.mean_ctrl, errors="coerce")
        top_type = dom_comp.loc[dom_comp.mean_ctrl_f.idxmax()] if dom_comp.mean_ctrl_f.notna().any() else None
        domain_block = dict(
            domain=dom,
            dominant_cell_type=(None if top_type is None else
                                dict(cell_type=top_type.type,
                                     fraction_in_controls=jsonable(top_type.mean_ctrl_f))),
            n_candidates=len(pool),
            # The topology axis's networks, one per upheld programme. Sizes only:
            # the programme's movement (its effect size) is programme-axis
            # evidence and stays in programme_shift.
            topology_networks=[
                dict(programme=pr, n_nodes=int(len(chn[pr])),
                     n_nodes_in_graph=jsonable(sheaf_rep.get(pr, {}).get("graph_nodes")),
                     n_edges=jsonable(sheaf_rep.get(pr, {}).get("graph_edges")),
                     n_residual_set=len(resid_sets[pr]))
                for pr in nets],
            # OBSERVATION ONLY. What the programme is, and whether its movement is
            # pathological or protective, is NOT stated here: that is an inference
            # requiring knowledge of what these genes do, and it belongs to the
            # direction agent, which has literature access. A falling programme is
            # not automatically one to restore -- a rising one can be protective
            # compensation, where restraining it would do harm.
            programme_shift=[
                dict(programme=pr, cohens_d=dvals[pr],
                     direction="lower in cases" if dvals[pr] < 0 else "higher in cases",
                     dominant_cell_type=ann.loc[pr, "annotation"],
                     top_loading_genes=ann.loc[pr, "top5"],
                     enriched_terms=[
                         dict(term=t.term, library=t.library, fdr=jsonable(t.fdr))
                         for t in enr[(enr.program == pr) & (enr.fdr < 0.05)]
                                    .nsmallest(8, "fdr").itertuples()])
                for pr in progs],
            composition=[dict(cell_type=r.type, mean_ctrl=jsonable(r.mean_ctrl),
                              mean_ad=jsonable(r.mean_ad), diff=jsonable(r.diff),
                              p=jsonable(r.p)) for r in cshare.itertuples()],
        )

        cands = []
        for g in pool:
            support = ("N" if g in hit_set else "") + ("S" if g in resid_set else "") + \
                      ("C" if g in top_charge else "")
            crow = ch[ch.g == g]
            a1 = dict(
                is_hit=g in hit_set,
                strata={ct: dict(log2FC=jsonable(v[0]), padj=jsonable(v[1]))
                        for ct, v in deg[dom].get(g, {}).items()},
                designed_contrasts=designed_block(dom, g),
                other_domains={d2: st for d2, st in recur.get(g, {}).items() if d2 != dom},
                n_domains_hit=len(recur.get(g, {})),
            )
            a2 = dict(
                in_top_charge=g in top_charge,
                charge=jsonable(crow.charge.iloc[0]) if len(crow) else None,
                charge_rank=jsonable(crow["rank"].iloc[0]) if len(crow) else None,
                cancellation=jsonable(crow.cancellation.iloc[0]) if len(crow) else None,
                per_programme=({pr: dict(contribution=jsonable(crow[f"contrib_{pr}"].iloc[0]),
                                         programme_d=dvals[pr])
                                for pr in progs if f"contrib_{pr}" in crow.columns}
                               if len(crow) else {}),
            )
            def net_block(pr):
                """This gene's position in ONE programme network. Not a node of it
                (not among that programme's top loaders) is a fact, not a score."""
                crow_ = chn[pr][chn[pr].g == g]
                if not len(crow_):
                    return dict(is_node=False)
                srow_ = sh[pr][sh[pr].g == g]; rrow_ = rs[pr][rs[pr].g == g]
                terms_ = pw[pr][pw[pr].g == g]
                return dict(
                    is_node=True,
                    in_graph=bool(len(srow_)),          # False: a node with no edge at this threshold
                    in_residual_set=g in resid_sets[pr],
                    scales_top=jsonable(rrow_.scales_top.iloc[0]) if len(rrow_) else None,
                    scales_present=jsonable(rrow_.scales_present.iloc[0]) if len(rrow_) else None,
                    spectral_rank=jsonable(srow_["rank"].iloc[0]) if len(srow_) else None,
                    degree=jsonable(srow_.degree.iloc[0]) if len(srow_) else None,
                    weighted_degree=jsonable(srow_.wdegree.iloc[0]) if len(srow_) else None,
                    betweenness=jsonable(srow_.betweenness.iloc[0]) if len(srow_) else None,
                    clustering=jsonable(srow_.clustering.iloc[0]) if len(srow_) else None,
                    pathways=[dict(term=t.term, library=t.library, fdr=jsonable(t.fdr),
                                   degree_in_pathway=jsonable(t.degree_in_pathway),
                                   pathway_size=jsonable(t.pathway_size_in_nodes))
                              for t in terms_.sort_values("fdr").head(6).itertuples()])
            a3 = dict(
                in_residual_set=g in resid_set,
                residual_networks=[pr for pr in nets if g in resid_sets[pr]],
                networks={pr: net_block(pr) for pr in nets},
            )
            idr = ident.get(g)
            identity = dict(
                symbol=g,
                uniprot=(None if idr is None or pd.isna(idr.uniprot) else idr.uniprot),
                protein=(None if idr is None or pd.isna(idr.protein) else idr.protein),
                identity_note=(
                    "no reviewed human entry matched this symbol exactly; treat identity as "
                    "unresolved and do not substitute a similar symbol"
                    if idr is None or pd.isna(idr.uniprot) else
                    ("symbol matched more than one reviewed entry; identity is uncertain"
                     if getattr(idr, "n_exact_matches", 1) > 1 else None)),
            )
            cands.append(dict(gene=g, domain=dom, identity=identity, support=support,
                              axis1_dysregulation=a1, axis2_programme=a2, axis3_topology=a3))

        out = dict(**domain_block, candidates=cands)
        json.dump(out, open(os.path.join(a.out, f"domain_{dom}.json"), "w"), indent=1)
        index[dom] = dict(n_candidates=len(pool), programmes=progs,
                          networks={pr: dict(n_nodes=int(len(chn[pr])), n_residual_set=len(resid_sets[pr]))
                                    for pr in nets},
                          support_counts=dict(collections.Counter(c["support"] for c in cands)))
        print(f"[domain {dom}] {len(pool)} candidates  " +
              " ".join(f"{k}:{v}" for k, v in sorted(index[dom]["support_counts"].items())), flush=True)

    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(f"\nwrote {a.out}: cohort.json, domain_<d>.json, index.json")


if __name__ == "__main__":
    main()
