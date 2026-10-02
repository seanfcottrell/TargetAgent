#!/usr/bin/env python
"""
Assemble every result of one run into the material for the final write-up.

Deterministic: reads the outputs of every stage and writes

    report_data.json   one structured record per stage -- cohort, cell types,
                       domains, programmes, the domain panel's decision, DEG,
                       topology, targets -- which is what the
                       report writer agent reads, and all it reads
    report_tables.md   the same material as tables: the appendix of the report,
                       and a complete account on its own if no agent is run

Nothing is interpreted here. Long free-text fields from the panels are clipped,
never rewritten.

Paths come from a manifest (run_all.py writes <run>/report/manifest.json), so the
same script assembles any run, including one laid out differently:

    python assemble_report.py --manifest <run>/report/manifest.json
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from paths import load_yaml  # noqa: E402


def rd(path, **kw):
    return pd.read_csv(path, keep_default_na=False, **kw)


def jl(path):
    return json.load(open(path)) if path and os.path.isfile(path) else None


def r3(x):
    try:
        x = float(x)
        return None if np.isnan(x) else round(x, 3)
    except (TypeError, ValueError):
        return x


def clip(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_none_\n"
    cols = cols or list(rows[0])
    esc = lambda v: ("" if v is None else str(v)).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(esc(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    M = json.load(open(a.manifest))
    card = load_yaml(M["card"])
    out = M["out"]
    os.makedirs(out, exist_ok=True)
    secs = [s["section"] for s in card["samples"]]
    cond = card["condition"]
    ctrl_label = str(cond["control_label"])
    D, md = {}, [f"# {card['dataset']}: results tables\n"]

    # ------------------------------------------------------------- cohort
    prep = rd(os.path.join(M["prepped"], "prep_summary.csv"), dtype={"section": str})
    prep = prep[prep.section.isin(secs)]
    qc = rd(os.path.join(M["celltypes_qc"], "composition_by_section.csv"), index_col=0)
    cohort = []
    for s in card["samples"]:
        r = prep[prep.section == s["section"]]
        q = qc.loc[s["section"]] if s["section"] in qc.index else None
        cohort.append(dict(
            section=s["section"],
            group=cond["control_group"] if str(s["stage"]) == ctrl_label else cond["case_group"],
            stage=s["stage"], chip_lot=s["chip_lot"], gef_writer=s["gef_writer"],
            age=s.get("age"), braak=s.get("braak"),
            tiles=int(r.n_bins.iloc[0]) if len(r) else None,
            median_counts_per_tile=r3(r.med_counts.iloc[0]) if len(r) else None,
            nuclei_segmented=int(q.n_all) if q is not None else None,
            nuclei_qc_pass=int(q.n_kept) if q is not None else None,
            frac_qc_pass=r3(q.frac_kept) if q is not None else None))
    D["dataset"] = dict(description=card["description"], condition=cond,
                        priors=dict(n_domains=card["priors"]["n_domains"],
                                    cell_types={k: v["label"] for k, v in card["priors"]["cell_types"].items()}),
                        cohort=cohort)
    md += ["## Cohort\n", table(cohort)]

    # ---------------------------------------------------------- cell types
    types = list(card["priors"]["cell_types"]) + ["Unk"]
    comp_sec = []
    for s in secs:
        if s in qc.index:
            comp_sec.append(dict(section=s, **{t: r3(qc.loc[s].get(f"frac_{t}")) for t in types
                                               if f"frac_{t}" in qc.columns}))
    rec = jl(os.path.join(M["celltypes_qc"], "run_record.json")) or {}
    ann = rec.get("annotate", {})
    D["cell_types"] = dict(
        n_clusters=rec.get("n_clusters"), n_nuclei_qc_pass=rec.get("n_kept"),
        clusters_all_callers_agree=ann.get("n_agree"),
        frac_nuclei_in_agreeing_clusters=r3(ann.get("frac_nuclei_agree")),
        frac_nuclei_unknown=r3(ann.get("frac_nuclei_unk")),
        composition_by_section=comp_sec)
    md += ["## Cell-type composition by section (fraction of QC-passed nuclei)\n", table(comp_sec)]

    # ------------------------------------------------------------- domains
    fit = M["fit"]
    res = jl(M.get("resolution_decision")) or jl(os.path.join(fit, "chosen_resolution_QH_sccg.json"))
    rs = res["resolution_str"]
    tri = rd(os.path.join(fit, f"cluster_triage_QH_sccg_res{rs}.csv"), dtype={"cluster": str})
    coh = rd(os.path.join(fit, f"domain_coherence_QH_sccg_res{rs}.csv"), dtype={"cl": str}).set_index("cl")
    cl = rd(os.path.join(M["composition"], "composition_long.csv"), dtype={"domain": str})
    cbd = rd(os.path.join(M["composition"], "composition_by_domain.csv"), dtype={"domain": str})
    for c in ("mean_ctrl", "mean_ad", "p"):
        cbd[c] = pd.to_numeric(cbd[c], errors="coerce")
    n_all = tri.n_bins.sum()
    doms = []
    for _, t in tri.sort_values("n_bins", ascending=False).iterrows():
        d = t.cluster
        sub = cl[cl.domain == d]
        pooled = (sub.assign(w=sub.frac * sub.n_cells).groupby("type").w.sum()
                  / sub.groupby("type").n_cells.sum()) if len(sub) else pd.Series(dtype=float)
        top = ", ".join(f"{k} {v:.2f}" for k, v in pooled.sort_values(ascending=False).head(3).items())
        shifts = cbd[(cbd.domain == d)]
        sig = [f"{r.type} {r.mean_ctrl:.3f}->{r['mean_ad']:.3f} (p {r.p:.3g})"
               for _, r in shifts.iterrows() if pd.notna(r.p) and r.p < 0.05]
        doms.append(dict(domain=d, bins=int(t.n_bins), frac_of_tiles=r3(t.n_bins / n_all),
                         coherence=r3(coh.loc[d, "spatial_coherence"]) if d in coh.index else None,
                         triage=t.classification, composition_top3=top,
                         composition_shifts_p_lt_0_05="; ".join(sig) or "none"))
    D["domains"] = dict(resolution=res.get("resolution"), n_domains=res.get("n_domains", len(tri)),
                        rule=res.get("rule"), domains=doms)
    md += [f"## Spatial domains (resolution {rs}; rule: {res.get('rule')})\n", table(doms)]

    # ---------------------------------------------------------- programmes
    pa = M["progact"]
    top = rd(os.path.join(pa, "program_top_genes.csv"))
    pann = rd(os.path.join(pa, "program_annotation.csv"), index_col=0)
    enr = pd.read_csv(os.path.join(pa, "program_enrichment.csv"))
    cu = pd.read_csv(os.path.join(pa, "contrast_usage_centred.csv"), dtype={"domain": str})
    ca = pd.read_csv(os.path.join(pa, "contrast_activity.csv"), dtype={"domain": str})
    progs, prog_rows = {}, []
    for pg in top.columns:
        terms = []
        for lib, g in enr[(enr.program == pg) & (enr.fdr < 0.05)].groupby("library"):
            terms += [f"{t} [{lib}]" for t in g.nsmallest(3, "fdr").term]
        shift = {}
        for _, r in cu[cu.program == pg].iterrows():
            a_ = ca[(ca.program == pg) & (ca.domain == r.domain)]
            shift[r.domain] = dict(
                usage_centred_d=r3(r.cohen_d), usage_centred_fdr=r3(r.fdr),
                activity_d=r3(a_.cohen_d.iloc[0]) if len(a_) else None,
                activity_fdr=r3(a_.fdr.iloc[0]) if len(a_) else None)
        progs[pg] = dict(top_genes=[g.split(" ")[0] for g in top[pg][:12]],
                         marker_enrichment=str(pann.loc[pg, "annotation"]) if pg in pann.index else None,
                         enriched_terms=terms[:12], shift_by_domain=shift)
        best = cu[(cu.program == pg) & cu.fdr.notna()].sort_values("fdr").head(1)
        prog_rows.append(dict(programme=pg, top_genes=", ".join(progs[pg]["top_genes"][:8]),
                              marker_enrichment=progs[pg]["marker_enrichment"],
                              top_terms="; ".join(t.split(" [")[0] for t in terms[:3]),
                              strongest_usage_shift=(f"domain {best.domain.iloc[0]}: d "
                                                     f"{float(best.cohen_d.iloc[0]):+.2f}, FDR "
                                                     f"{float(best.fdr.iloc[0]):.3g}") if len(best) else ""))
    D["programmes"] = dict(
        n_tests_per_table=int(len(cu)),
        semantics=("usage_centred = within-section usage with the section mean removed (depth-robust); "
                   "activity = usage x per-section amplitude. Cohen's d case vs control, "
                   "BH FDR over every domain x programme test of a table."),
        programmes=progs)
    md += ["## Gene programmes\n", table(prog_rows)]

    # -------------------------------------------------------- domain panel
    dp = M["domain_panel"]
    syn = jl(os.path.join(dp, "synthesis.json")) or {}
    rank_rows = [dict(rank=r["rank"], unit=r["unit"], decision=r["decision"],
                      relevance=r3(r["relevance_consensus"]), suitability=r3(r["suitability_consensus"]),
                      independent_lines=r["independent_lines"], strata=", ".join(r.get("strata", [])),
                      rationale=clip(r.get("rationale"), 700),
                      predictions=[clip(x, 300) for x in r.get("predictions", [])[:3]])
                 for r in sorted(syn.get("ranking", []), key=lambda r: r["rank"])]
    D["domain_panel"] = dict(rule_applied=syn.get("rule_applied"), ranking=rank_rows,
                             caveats=clip(syn.get("caveats"), 2000),
                             decision=jl(M.get("decisions_domains")))
    md += ["## Domain panel decision\n",
           table([{k: v for k, v in r.items() if k not in ("rationale", "predictions")} for r in rank_rows])]

    # ------------------------------------------------------------------ DEG
    deg = M["deg"]
    summ = rd(os.path.join(deg, "summary.csv"), dtype={"domain": str}) \
        if os.path.isfile(os.path.join(deg, "summary.csv")) else pd.DataFrame()
    hits = []
    for f in sorted(glob.glob(os.path.join(deg, "deg_domain*.csv"))):
        dom, ct = os.path.basename(f)[len("deg_domain"):-4].split("_", 1)
        d = pd.read_csv(f)
        for _, r in d[d.hit].iterrows():
            hits.append(dict(gene=r.gene, domain=dom, stratum=ct, log2FC=r3(r.log2FC), padj=float(f"{r.padj:.3g}")))
    rec_ = pd.DataFrame(hits)
    recurrence = []
    if len(rec_):
        for g, sub in rec_.groupby("gene"):
            recurrence.append(dict(gene=g, n_domains=sub.domain.nunique(), n_contrasts=len(sub),
                                   direction="up" if (sub.log2FC > 0).all() else
                                   ("down" if (sub.log2FC < 0).all() else "mixed"),
                                   strata=", ".join(sorted(set(sub.stratum)))))
        recurrence.sort(key=lambda r: (-r["n_domains"], -r["n_contrasts"], r["gene"]))
    summ_rows = summ.to_dict("records") if len(summ) else []
    # contrasts the domain panel designed beyond the default (deg_run.py)
    specs = jl(os.path.join(deg, "designed_contrasts.json")) or []
    side = lambda sp: f"{sp['group']}:{'+'.join(map(str, sp['domains']))}"
    dhits = []
    for c in specs:
        f = os.path.join(deg, f"deg_contrast_{c['id']}.csv")
        if not os.path.isfile(f):
            continue
        d = pd.read_csv(f)
        for _, r in d[d.hit].iterrows():
            dhits.append(dict(gene=r.gene, contrast=c["id"], test=c["test"], side_a=side(c["side_a"]),
                              side_b=side(c["side_b"]), cell_type=c["type"],
                              disease_informative=c.get("disease_informative"),
                              log2FC=r3(r.log2FC), padj=float(f"{r.padj:.3g}")))
    D["deg"] = dict(method="NEBULA negative-binomial mixed model, donor random intercept, library-size offset",
                    hit_rule="FDR < 0.05 and |log2FC| > 1", contrasts=summ_rows, hits=hits,
                    recurrence=recurrence, designed_contrasts=specs, designed_hits=dhits)
    md += ["## Differential expression: contrasts\n", table(summ_rows),
           "## Differential expression: genes (recurrence across domains)\n", table(recurrence),
           "## Differential expression: designed contrasts\n",
           table([dict(id=c["id"], test=c["test"], side_a=side(c["side_a"]), side_b=side(c["side_b"]),
                       cell_type=c["type"], disease_informative=c.get("disease_informative"),
                       question=c.get("question")) for c in specs]),
           "## Differential expression: designed-contrast hits\n", table(dhits)]

    # ------------------------------------------------------------- topology
    sh = M["sheaf"]
    prov = jl(os.path.join(sh, "charges", "provenance.json")) or {"domains": {}}
    resid = jl(os.path.join(sh, "out_rank", "residual_per_scale.json")) or {}
    topo = []          # one row per network: one network per upheld programme per domain
    for d, v in prov.get("domains", {}).items():
        for pr, n in v.get("networks", {}).items():
            topo.append(dict(domain=d, programme=pr, cohens_d=r3(n.get("d")),
                             nodes=n.get("n_nodes"), top_charge=", ".join(n.get("top20", [])[:10]),
                             residual_set=", ".join(resid.get(d, {}).get(pr, {}).get("genes", []))))
    D["topology"] = dict(statistic=("per-scale residual of the leave-one-out sheaf spectral shift after "
                                    "regressing out weighted degree, intersected across scales; one "
                                    "network per upheld programme per domain, never merged"),
                         domains=topo)
    md += ["## Network topology (sheaf degree residual)\n", table(topo)]

    # -------------------------------------------------------------- targets
    tp = M["target_panel"]
    ts = jl(os.path.join(tp, "synthesis.json")) or {}
    trows = [dict(rank=r["rank"], gene=r["gene"], domains=", ".join(map(str, r.get("domains", []))),
                  tier=r["tier"], biological_case=r3(r.get("biological_case")), direction=r.get("direction"),
                  supporting_axes=", ".join(r.get("supporting_axes", [])),
                  direction_basis=clip(r.get("direction_basis"), 400),
                  rationale=clip(r.get("rationale"), 500),
                  rejection_reason=clip(r.get("rejection_reason"), 250))
             for r in sorted(ts.get("ranking", []), key=lambda r: r["rank"])]
    tiers = pd.Series([r["tier"] for r in trows]).value_counts().to_dict() if trows else {}
    D["targets"] = dict(rule_applied=clip(ts.get("rule_applied"), 2000), tier_counts=tiers, ranking=trows,
                        novel_candidates=ts.get("novel_candidates", []),
                        caveats=clip(ts.get("caveats"), 2000))
    md += [f"## Targets: the panel's ranking ({', '.join(f'{k} {v}' for k, v in tiers.items())})\n",
           table([{k: r[k] for k in ("rank", "gene", "domains", "tier", "biological_case", "direction",
                                     "supporting_axes")} for r in trows if r["tier"] != "rejected"]),
           "\nRejected: " + (", ".join(r["gene"] for r in trows if r["tier"] == "rejected") or "none") + "\n"]

    # The selection on data merit (agents/Panel2/select_targets.py): tiers recomputed
    # from the critic's verdicts on DATA items only. This, not the panel's tiers, is
    # what is carried forward.
    sel = jl(M.get("decisions_targets")) if M.get("decisions_targets") else None
    if sel:
        srow = lambda r, status: dict(gene=r["gene"], status=status, data_tier=r["tier"], panel_tier=r["panel_tier"],
                                      data_lines=r["data_lines"], data_axes=", ".join(r.get("data_axes", [])),
                                      direction=r.get("direction"), reason=clip(r.get("reason"), 250))
        srows = ([srow(r, "admitted") for r in sel["ranking"]] +
                 [srow(r, "direction_open") for r in sel.get("direction_open", [])] +
                 [srow(r, "excluded") for r in sel.get("excluded", [])])
        D["target_selection"] = dict(rule=sel["rule"], rows=srows,
                                     n_admitted=len(sel["ranking"]), n_direction_open=len(sel.get("direction_open", [])),
                                     n_excluded=len(sel.get("excluded", [])))
        md += ["## Targets selected on data merit\n", f"_Rule: {sel['rule']}_\n",
               table([{k: r[k] for k in ("gene", "status", "data_tier", "panel_tier", "data_lines", "data_axes",
                                         "direction")} for r in srows if r["status"] != "excluded"]),
               "\nExcluded: " + ("; ".join(f"{r['gene']} ({r['reason']})" for r in srows
                                          if r["status"] == "excluded") or "none") + "\n"]

    # ---------------------------------------------------------- verification
    def cv(path):
        v = jl(path)
        if not v:
            return None
        return dict(n_citations=v["n_citations"], n_searches=v["n_searches"],
                    status_counts=v["status_counts"],
                    not_supporting=[dict(where=c["where"], status=c["status"], title=clip(c.get("title"), 160),
                                         pmid=c.get("pmid"), claim=clip(c.get("claim"), 200))
                                    for c in v["citations"] if not c["supports"]][:60])
    D["verification"] = dict(
        rule=("literature citations are machine-checked: a web search in the run must have returned "
              "them and any PMID must resolve in PubMed to the same paper; only 'verified' and "
              "'verified_session' support a claim."),
        domain_panel=cv(os.path.join(M["domain_panel"], "citation_verification.json")),
        target_panel=cv(os.path.join(M["target_panel"], "citation_verification.json")))
    vrows = []
    for panel in ("domain_panel", "target_panel"):
        v = D["verification"][panel]
        if v:
            vrows.append(dict(panel=panel, citations=v["n_citations"], searches=v["n_searches"],
                              **{k: v["status_counts"].get(k, 0) for k in
                                 ("verified", "verified_session", "exists_not_searched", "untraceable",
                                  "pmid_mismatch", "pmid_not_found")}))
    md += ["## Verification of literature citations\n", table(vrows)]

    json.dump(D, open(os.path.join(out, "report_data.json"), "w"), indent=1, default=str)
    open(os.path.join(out, "report_tables.md"), "w").write("\n".join(md))
    n_tok = len(json.dumps(D, default=str)) // 4
    print(f"wrote {out}/report_data.json (~{n_tok:,} tokens) and report_tables.md")


if __name__ == "__main__":
    main()
