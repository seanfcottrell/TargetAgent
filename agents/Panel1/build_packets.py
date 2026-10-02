#!/usr/bin/env python
"""
Evidence packets for the cluster-selection agents.

    python build_packets.py --run <fit> --res <resolution> --composition <dir> \
        --progact <dir> --card <card.yaml> --out <run>/agents_packets

Deterministic: reads the final result tables and
writes one JSON per spatial domain plus a cohort packet. Every number carries
the file and field it came from, and every metric that needs a reference gets
one that is interpretable on its own: the share of tiles each section would
contribute if a domain were spread evenly, the stage fractions expected if
uniform, and the within-stage versus cross-stage transport baselines measured
in this cohort. Agents see labelled quantities with a reference scale, never raw
matrices.

Units are spatial domains (>= --min-bins bins). Domain x cell-type strata are
described inside each domain packet (composition per type, per group) so agents
can recommend strata; they are not separate units. Each packet also carries the
tile boundary its domain shares with every other domain (per group) and where its
tissue lands under transport in both directions (case to control, control to
case), so agents can propose comparisons across domains.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import RUN_ROOT as BASE, load_card  # dataset outputs live under runs/<dataset>/<run>/
FIN = None                                       # the fitted run directory, from --run
NA = dict(keep_default_na=False)


def rd(path, **kw):
    return pd.read_csv(path, **NA, **kw)


def r3(x):
    try:
        return None if pd.isna(x) else round(float(x), 3)
    except Exception:
        return x


def contiguity(mask_by_section, bx, by, sections):
    """Connected components of a domain's tiles per section under rook adjacency:
    fraction of the domain's bins that sit in components of >= 50 tiles, and
    components per 1000 bins (salt-and-pepper = many tiny components)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n_big, n_all, n_comp = 0, 0, 0
    per_sec = {}
    for s in sections:
        idx = np.where(mask_by_section == s)[0]
        if len(idx) == 0:
            continue
        key = {(int(bx[i]), int(by[i])): j for j, i in enumerate(idx)}
        rows, cols = [], []
        for (x, y), j in key.items():
            for dx, dy in ((1, 0), (0, 1)):
                k = key.get((x + dx, y + dy))
                if k is not None:
                    rows.append(j); cols.append(k)
        A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(idx), len(idx)))
        nc, lab = connected_components(A, directed=False)
        sizes = np.bincount(lab)
        big = int(sizes[sizes >= 50].sum())
        per_sec[s] = dict(n_bins=int(len(idx)), n_components=int(nc), frac_in_components_ge50=round(big / len(idx), 3),
                          largest_component_frac=round(float(sizes.max() / len(idx)), 3))
        n_big += big; n_all += len(idx); n_comp += nc
    return dict(frac_bins_in_components_ge50=round(n_big / max(n_all, 1), 3),
                components_per_1000_bins=round(1000 * n_comp / max(n_all, 1), 2), per_section=per_sec)


def adjacency(sec, bx, by, dom, group_of):
    """Shared tile boundary between domains, per group: for every pair of 4-neighbour
    tiles in one section that carry different domains, one boundary edge for each
    side. Returns {group: {domain: {other: share of that domain's boundary}}}."""
    t = pd.DataFrame(dict(section=sec, bx=bx, by=by, dom=dom))
    pairs = []
    for dx, dy in ((1, 0), (0, 1)):
        nb = t.assign(bx=t.bx - dx, by=t.by - dy)
        m = t.merge(nb, on=["section", "bx", "by"], suffixes=("", "_nb"))
        pairs.append(m[m.dom != m.dom_nb][["section", "dom", "dom_nb"]])
    e = pd.concat(pairs)
    e = pd.concat([e, e.rename(columns={"dom": "dom_nb", "dom_nb": "dom"})])   # both sides
    e["group"] = e.section.map(group_of)
    out = {}
    for g, sub in e.groupby("group"):
        c = sub.groupby(["dom", "dom_nb"]).size()
        share = c / c.groupby(level=0).transform("sum")
        out[g] = {d: {o: r3(v) for o, v in share.loc[d].sort_values(ascending=False).items()}
                  for d in share.index.get_level_values(0).unique()}
    return out


def main():
    global FIN
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(BASE, "agents_packets"))
    p.add_argument("--min-bins", type=int, default=100)
    p.add_argument("--run", required=True, help="fitted run directory")
    p.add_argument("--res", default=None,
                   help="resolution suffix of the partition, e.g. 0.5 -> bin_clusters_QH_sccg_res0.5.csv; "
                        "not needed when the three partition files are given explicitly")
    p.add_argument("--composition", required=True)
    p.add_argument("--progact", required=True)
    p.add_argument("--card", default=None, help="data card (default $WALKTHROUGH_CARD)")
    # a relabelled partition (STORM/pool_partition.py) brings its own tables
    p.add_argument("--clusters-file", default=None, help="partition file (overrides --res naming)")
    p.add_argument("--triage-file", default=None)
    p.add_argument("--coherence-file", default=None)
    p.add_argument("--transport-dir", default=None,
                   help="dir holding transport_chain/ and transport_posthoc/ (default: the fit dir)")
    p.add_argument("--only-units", nargs="*", default=None, help="write packets for these domain ids only")
    a = p.parse_args()
    FIN = a.run
    COMP, PROG = a.composition, a.progact
    card = load_card(a.card)
    cond = card["condition"]
    CTRL_LABEL, CASE_G, CTRL_G = str(cond["control_label"]), cond["case_group"], cond["control_group"]

    def rel(path):
        """Source paths as the agents see them: relative to the run root."""
        return os.path.relpath(path, BASE)

    def fin_file(kind):
        """Partition-derived file: the one given explicitly, else the fit's file
        named by the resolution."""
        given = {"triage": a.triage_file, "coherence": a.coherence_file, "clusters": a.clusters_file}[kind]
        if given:
            return given
        if a.res is None:
            raise SystemExit(f"pass --res, or --{kind}-file")
        named = {"triage": f"cluster_triage_QH_sccg_res{a.res}.csv",
                 "coherence": f"domain_coherence_QH_sccg_res{a.res}.csv",
                 "clusters": f"bin_clusters_QH_sccg_res{a.res}.csv"}
        if os.path.isfile(os.path.join(FIN, named[kind])):
            return os.path.join(FIN, named[kind])
        raise SystemExit(f"missing {kind} file in {FIN}: {named[kind]}")
    TDIR = a.transport_dir or FIN
    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------ cohort
    prep = rd(os.path.join(BASE, "prepped", "prep_summary.csv"), dtype={"section": str, "stage": str})
    # the cohort is whatever the fit actually used: a run on a subset of sections
    # must not describe donors that were excluded before fitting
    run_secs = list(dict.fromkeys(rd(os.path.join(FIN, "bin_meta.csv"), dtype={"section": str}).section))
    prep = prep[prep.section.isin(run_secs)].reset_index(drop=True)
    print(f"cohort: {len(prep)} sections from {FIN}: {', '.join(prep.section)}", flush=True)
    qc = rd(os.path.join(BASE, "celltypes_qc", "composition_by_section.csv"), index_col=0)
    meta_obs = {}
    import anndata as ad
    for s in prep.section:
        A = ad.read_h5ad(os.path.join(BASE, "prepped", f"{s}_prepped.h5ad"), backed="r")
        o = A.obs.iloc[0]
        e4 = pd.to_numeric(pd.Series([o.get("apoe_e4")]), errors="coerce").iloc[0]
        meta_obs[s] = dict(braak=r3(o["braak"]), cerad=r3(o["cerad"]), thal=r3(o["thal"]),
                           age_mid=r3(o["age_mid"]), apoe=str(o["apoe"]),
                           apoe_e4=None if pd.isna(e4) else int(e4),
                           sex=str(o["sex"]), gef_writer=str(o["gef_writer"]))
        A.file.close()
    samples = []
    for _, r in prep.iterrows():
        s = r.section
        samples.append(dict(section=s, stage=r.stage, group=CTRL_G if r.stage == CTRL_LABEL else CASE_G,
                            chip_lot=r.lot, n_bins=int(r.n_bins), median_counts_per_bin=r3(r.med_counts),
                            nuclei_per_bin_median=r3(r.nuclei_med),
                            n_nuclei_segmented=int(qc.loc[s, "n_all"]), n_nuclei_qc_pass=int(qc.loc[s, "n_kept"]),
                            frac_nuclei_qc_pass=r3(qc.loc[s, "frac_kept"]), **meta_obs[s]))
    S = pd.DataFrame(samples)
    order = [str(x) for x in cond.get("stage_order", [CTRL_LABEL])]
    order += sorted(set(S.stage) - set(order))
    STAGES = [st for st in order if (S.stage == st).any()]
    stage_ord = S.stage.map({st: i for i, st in enumerate(order)})
    case, ctrl = S[S.group == CASE_G], S[S.group == CTRL_G]
    desc = card["description"]
    cohort = dict(
        study=(f"{desc['technology']} Tissue: {desc['tissue']}. Disease: {desc['disease']}. "
               f"{len(S)} sections, {desc['design']}: {len(case)} {CASE_G} "
               f"({', '.join(f'{n} {st}' for st, n in case.stage.value_counts().items())}) and "
               f"{len(ctrl)} {CTRL_G} (stage label '{CTRL_LABEL}')."),
        samples=samples,
        design=dict(n_case=len(case), n_ctrl=len(ctrl), case_group=CASE_G, control_group=CTRL_G,
                    stages=S.stage.value_counts().to_dict(),
                    unit_of_replication="donor section (one section per donor)",
                    contrasts_supported=(f"{CASE_G} vs {CTRL_G} ({len(case)} vs {len(ctrl)}, Welch on sample means)"
                                         + (f"; stage trend {'<'.join(STAGES)} (Spearman)" if len(STAGES) > 2
                                            else "; only two stage levels, so no stage trend"))),
        confounds=dict(
            depth_vs_stage_spearman=r3(pd.Series(S.median_counts_per_bin).corr(stage_ord, method="spearman")),
            qc_pass_vs_stage_spearman=r3(pd.Series(S.frac_nuclei_qc_pass).corr(stage_ord, method="spearman")),
            chip_lot_by_stage=S.groupby("stage").chip_lot.apply(lambda x: sorted(set(x))).to_dict(),
            note=(f"Depth (median counts per bin) is "
                  f"{case.median_counts_per_bin.min():.0f}-{case.median_counts_per_bin.max():.0f} in the {CASE_G} sections "
                  f"against {ctrl.median_counts_per_bin.min():.0f}-{ctrl.median_counts_per_bin.max():.0f} in {CTRL_G}; "
                  f"nucleus QC pass rate {case.frac_nuclei_qc_pass.min():.2f}-{case.frac_nuclei_qc_pass.max():.2f} against "
                  f"{ctrl.frac_nuclei_qc_pass.min():.2f}-{ctrl.frac_nuclei_qc_pass.max():.2f}. "
                  f"Chip lots shared between the groups: "
                  f"{', '.join(sorted(set(case.chip_lot) & set(ctrl.chip_lot))) or 'none'}.")),
        sources=dict(prep="prepped/prep_summary.csv + prepped/*_prepped.h5ad obs", qc="celltypes_qc/composition_by_section.csv"))

    # programme descriptions (global, shared by every domain packet)
    enrich = json.load(open(os.path.join(PROG, "program_enrichment.json")))
    ann = rd(os.path.join(PROG, "program_annotation.csv"), index_col=0)
    top = rd(os.path.join(PROG, "program_top_genes.csv"))
    wct = rd(os.path.join(PROG, "contrast_w.csv"))
    wbs = rd(os.path.join(PROG, "w_by_sample.csv"), index_col=0)
    smu = rd(os.path.join(PROG, "sample_mean_usage.csv"), index_col=0)
    progs = list(top.columns)
    programmes = {}
    for pg in progs:
        w = wct[wct.program == pg].iloc[0]
        programmes[pg] = dict(
            top_genes=[g.split(" ")[0] for g in top[pg][:12]],
            marker_enrichment_max_type=str(ann.loc[pg, "annotation"]),
            amplitude_w_mean_by_stage={st: r3(wbs[wbs.stage == st][pg].mean()) for st in STAGES},
            amplitude_w_AD_vs_CTRL=dict(mean_ctrl=r3(w.mean_ctrl), mean_ad=r3(w.mean_ad), cohen_d=r3(w.cohen_d), p=r3(w.p), fdr=r3(w.fdr), stage_rho=r3(w.stage_rho), stage_fdr=r3(w.stage_fdr)),
            section_mean_usage_by_stage={st: r3(smu.loc[[s for s in smu.index if S.set_index("section").loc[s, "stage"] == st], pg].mean()) for st in STAGES},
            enrichment=enrich["programmes"].get(pg, {}))
    cohort["programmes"] = programmes
    cohort["programme_enrichment_method"] = enrich["method"]
    cohort["programme_semantics"] = ("X_k ~ Q_k H D_k B^T. B: gene loadings (nonnegative, unit columns) = the programme. "
        "usage = (Q_k H)[bin, r]: how much a bin uses programme r (signed; same scale in every sample). "
        "amplitude w_kr = D_k: one number per sample per programme, the programme's overall strength in that sample "
        "(absorbs section-wide effects such as depth). activity = usage x amplitude. usage_c = usage minus the "
        "section-wide mean usage of r (every section its own control; section-wide effects cancel).")

    # ------------------------------------------------------------ per-domain tables
    tri = rd(fin_file("triage"), dtype={"cluster": str}).set_index("cluster")
    coh = rd(fin_file("coherence"), dtype={"cl": str}).set_index("cl")
    bm = rd(os.path.join(FIN, "bin_meta.csv"), dtype={"section": str, "stage": str, "lot": str})
    bm["dom"] = rd(fin_file("clusters"), dtype={"cluster": str}).cluster.values
    sec_share = pd.crosstab(bm.dom, bm.section, normalize="index")
    sec_n = pd.crosstab(bm.dom, bm.section)
    # per-bin depth from the prepped files (same row order as bin_meta)
    import anndata as ad
    depth = np.concatenate([np.asarray(ad.read_h5ad(os.path.join(BASE, "prepped", f"{s}_prepped.h5ad"), backed="r").obs["n_counts"], float)
                            for s in list(dict.fromkeys(bm.section))])
    assert len(depth) == len(bm)
    bm["depth"] = depth
    bx = np.round((bm.x.astype(float) - 55) / 110).astype(np.int64); by = np.round((bm.y.astype(float) - 55) / 110).astype(np.int64)
    group_of = dict(zip(S.section, S.group))
    adj = adjacency(bm.section.to_numpy(), bx, by, bm.dom.to_numpy(), group_of)
    comp_long = rd(os.path.join(COMP, "composition_long.csv"), dtype={"domain": str, "section": str, "stage": str})
    comp_dom = rd(os.path.join(COMP, "composition_by_domain.csv"), dtype={"domain": str})
    cu = rd(os.path.join(PROG, "contrast_usage_centred.csv"), dtype={"domain": str})
    ca = rd(os.path.join(PROG, "contrast_activity.csv"), dtype={"domain": str})
    ucm = rd(os.path.join(PROG, "usage_centred_domain_mean.csv"), index_col=0)
    ucm.index = ucm.index.astype(str)
    tch = rd(os.path.join(TDIR, "transport_chain", "domain_transport_summary.csv"), dtype={"src_domain": str, "top_partner": str})
    tph = rd(os.path.join(TDIR, "transport_posthoc", "domain_transport_summary.csv"), dtype={"src_domain": str, "top_partner": str})
    tpt = rd(os.path.join(TDIR, "transport_posthoc", "transport_tables.csv"), dtype={"src_domain": str, "dst_domain": str})
    doms = [d for d in tri.index if tri.loc[d, "n_bins"] >= a.min_bins
            and (a.only_units is None or d in set(a.only_units))]

    # Cohort-level transport references, COMPUTED from the tables: the bin-weighted
    # mean over every domain and both directions of each coupling kind. A domain's
    # own unmatched / self_map values are read against these.
    def wmean(df, col):
        w = df.src_n_bins.astype(float)
        return r3((df[col].astype(float) * w).sum() / w.sum()) if len(df) and w.sum() > 0 else None
    def baseline(df):
        return dict(n_couplings=int(df.edge.nunique()), unmatched=wmean(df, "unmatched"),
                    self_map=wmean(df, "self_map"))
    cohort["transport_baselines"] = dict(
        semantics=("Bin-weighted means over all domains and both directions. chain_within_stage: couplings "
                   "between sections of the same group; chain_cross_stage: the chain's couplings between "
                   "groups; posthoc_case_to_control: every case x control pair coupled after the fit."),
        chain_within_stage=baseline(tch[tch.kind == "within"]),
        chain_cross_stage=baseline(tch[tch.kind == "cross"]),
        posthoc_case_to_control=baseline(tph),
        sources=f"{rel(os.path.join(TDIR, 'transport_chain'))}/domain_transport_summary.csv, "
                f"{rel(os.path.join(TDIR, 'transport_posthoc'))}/domain_transport_summary.csv")
    json.dump(cohort, open(os.path.join(a.out, "cohort.json"), "w"), indent=1)
    stage_index = {st: i for i, st in enumerate(order)}
    index = []
    for d in doms:
        t = tri.loc[d]
        n_by_sec = sec_n.loc[d]
        stages_present = [st for st in order if f"frac_{st}" in t.index]
        stage_frac = {st: r3(t[f"frac_{st}"]) for st in stages_present}
        expected = {st: r3((bm.stage == st).mean()) for st in stages_present}
        spatial = dict(
            n_bins=int(t.n_bins), frac_of_all_bins=r3(t.n_bins / len(bm)),
            spatial_coherence_k8=r3(coh.loc[d, "spatial_coherence"]),
            bins_per_section={s: int(n_by_sec[s]) for s in n_by_sec.index},
            max_single_section_share=r3(sec_share.loc[d].max()), top_section=str(sec_share.loc[d].idxmax()),
            stage_fraction_of_bins=stage_frac, stage_fraction_expected_if_uniform=expected,
            section_share_expected_if_even={s_: r3(v) for s_, v in (bm.section.value_counts(normalize=True)).items()},
            contiguity=contiguity(np.where(bm.dom.to_numpy() == d, bm.section.to_numpy(), ""), bx, by, list(dict.fromkeys(bm.section))),
            adjacent_domains=dict(
                semantics=("Share of this domain's tile boundary (4-neighbour tile pairs where the other tile "
                           "is a different domain) shared with each other domain, per group. Domains too small "
                           "for their own packet are included."),
                by_group={g: adj.get(g, {}).get(d, {}) for g in (CASE_G, CTRL_G)}),
            sources=f"{rel(fin_file('triage'))}, {rel(fin_file('coherence'))}, {rel(os.path.join(FIN, 'bin_meta.csv'))} (contiguity computed from tile indices)")
        cl = comp_long[comp_long.domain == d]
        pooled = (cl.assign(w=cl.frac * cl.n_cells).groupby("type").w.sum() / cl.groupby("type").n_cells.sum()).round(3).to_dict()
        by_group = {g: cl[cl.group == g].groupby("type").frac.mean().round(3).to_dict() for g in (CTRL_G, CASE_G)}
        by_stage = {st: cl[cl.stage == st].groupby("type").frac.mean().round(3).to_dict() for st in STAGES}
        tests = comp_dom[comp_dom.domain == d]
        composition = dict(
            n_typed_nuclei=int(cl.groupby("section").n_cells.first().sum()),
            nuclei_per_section={s: int(v) for s, v in cl.groupby("section").n_cells.first().items()},
            fraction_pooled=pooled, fraction_mean_by_group=by_group, fraction_mean_by_stage=by_stage,
            ad_vs_ctrl_welch={str(r["type"]): dict(mean_ctrl=r3(r["mean_ctrl"]), mean_ad=r3(r["mean_ad"]), diff=r3(r["diff"]), t=r3(r["t"]), p=r3(r["p"])) for _, r in tests.iterrows()},
            caveat="Only QC-passed nuclei are typed; the pass rate per section is in cohort.samples[].frac_nuclei_qc_pass.",
            sources=f"{rel(os.path.join(COMP, 'composition_long.csv'))}, {rel(os.path.join(COMP, 'composition_by_domain.csv'))}")
        fu = cu[cu.domain == d].set_index("program"); fa = ca[ca.domain == d].set_index("program")
        factors = dict(
            usage_centred_domain_mean={pg: r3(ucm.loc[d, pg]) for pg in progs},
            usage_centred_ad_vs_ctrl={pg: dict(mean_ctrl=r3(fu.loc[pg, "mean_ctrl"]), mean_ad=r3(fu.loc[pg, "mean_ad"]), cohen_d=r3(fu.loc[pg, "cohen_d"]), p=r3(fu.loc[pg, "p"]), fdr=r3(fu.loc[pg, "fdr"]), stage_rho=r3(fu.loc[pg, "stage_rho"]), stage_fdr=r3(fu.loc[pg, "stage_fdr"])) for pg in progs if pg in fu.index},
            activity_ad_vs_ctrl={pg: dict(mean_ctrl=r3(fa.loc[pg, "mean_ctrl"]), mean_ad=r3(fa.loc[pg, "mean_ad"]), cohen_d=r3(fa.loc[pg, "cohen_d"]), p=r3(fa.loc[pg, "p"]), fdr=r3(fa.loc[pg, "fdr"]), stage_rho=r3(fa.loc[pg, "stage_rho"]), stage_fdr=r3(fa.loc[pg, "stage_fdr"])) for pg in progs if pg in fa.index},
            n_tests_per_table=int(len(cu)),
            note=(f"{len(cu)} domain x programme tests per table; FDR is BH over all {len(cu)}. "
                  "usage_centred isolates within-section shifts; activity includes sample amplitude."),
            sources=f"{rel(os.path.join(PROG, 'contrast_usage_centred.csv'))}, contrast_activity.csv, usage_centred_domain_mean.csv; programme definitions in cohort.programmes")
        ch = tch[tch.src_domain == d]; ph = tph[(tph.src_domain == d) & (tph.direction == "fwd")]
        dest = tpt[(tpt.src_domain == d) & (tpt.direction == "fwd")].groupby("dst_domain").frac.mean().round(3).to_dict()
        rh = tph[(tph.src_domain == d) & (tph.direction == "rev")]
        dest_rev = tpt[(tpt.src_domain == d) & (tpt.direction == "rev")].groupby("dst_domain").frac.mean().round(3).to_dict()
        transport = dict(
            semantics="Unbalanced FGW couplings between niches (800 per section). unmatched = share of this domain's mass the plan refused to transport at the accepted tau (no counterpart in the partner at that cost); self_map = share landing in the same domain id in the partner; top_partner = partner domain receiving most. posthoc_ad_to_ctrl reads this domain's tissue in case sections against every control section; posthoc_ctrl_to_ad reads its tissue in control sections against every case section (unmatched there = control tissue with no counterpart in that case section); destination_domain_share_mean says where the transported mass lands. Cohort-level references are in cohort.transport_baselines.",
            chain_within_stage=dict(n_edges=int((ch.kind == "within").sum()), unmatched_mean=r3(ch[ch.kind == "within"].unmatched.mean()), self_map_mean=r3(ch[ch.kind == "within"].self_map.mean())),
            chain_cross_stage=[dict(src=str(r["src"]), dst=str(r["dst"]), unmatched=r3(r["unmatched"]), self_map=r3(r["self_map"]), top_partner=str(r["top_partner"]), top_partner_frac=r3(r["top_partner_frac"])) for _, r in ch[ch.kind == "cross"].iterrows()],
            posthoc_ad_to_ctrl=dict(n_pairs=int(len(ph)), unmatched_mean=r3(ph.unmatched.mean()), self_map_mean=r3(ph.self_map.mean()),
                                    unmatched_by_ad_section=ph.groupby("src").unmatched.mean().round(3).to_dict(),
                                    self_map_by_ad_section=ph.groupby("src").self_map.mean().round(3).to_dict(),
                                    destination_domain_share_mean=dest),
            posthoc_ctrl_to_ad=dict(n_pairs=int(len(rh)), unmatched_mean=r3(rh.unmatched.mean()), self_map_mean=r3(rh.self_map.mean()),
                                    unmatched_by_ad_section=rh.groupby("dst").unmatched.mean().round(3).to_dict(),
                                    unmatched_by_ctrl_section=rh.groupby("src").unmatched.mean().round(3).to_dict(),
                                    destination_domain_share_mean=dest_rev),
            sources=f"{rel(os.path.join(TDIR, 'transport_chain'))}/, {rel(os.path.join(TDIR, 'transport_posthoc'))}/ (domain_transport_summary.csv, transport_tables.csv)")
        dsub = bm[bm.dom == d]
        dep = dsub.groupby("section").depth.median().round(0)
        dep_stage = dsub.groupby("section").stage.first().map(stage_index)
        design = dict(
            depth_median_counts_by_section={k: float(v) for k, v in dep.items()},
            depth_vs_stage_spearman_within_domain=r3(dep.corr(dep_stage.reindex(dep.index), method="spearman")),
            bins_share_by_stage_vs_expected=dict(observed=stage_frac, expected=expected),
            triage_classification=str(t.classification),
            donors_with_ge50_bins={st: int(t[f"donors_{st}"]) for st in stages_present if f"donors_{st}" in t.index},
            min_bins_per_section=int(t.min_bins_per_section),
            nuclei_per_section_min=int(min(composition["nuclei_per_section"].values())) if composition["nuclei_per_section"] else 0,
            mean_unmatched_chain=r3(t.mean_unmatched),
            sources=f"{rel(fin_file('triage'))}; cohort.json for covariates")
        packet = dict(unit=f"domain_{d}", domain=d,
                      identity=dict(n_bins=int(t.n_bins), frac_of_all_bins=r3(t.n_bins / len(bm))),
                      spatial=spatial, composition=composition, factors=factors, transport=transport, design=design)
        json.dump(packet, open(os.path.join(a.out, f"domain_{d}.json"), "w"), indent=1)
        index.append(dict(unit=f"domain_{d}", file=f"domain_{d}.json", n_bins=int(t.n_bins)))
        print(f"domain {d}: {int(t.n_bins):,} bins, coherence {spatial['spatial_coherence_k8']}, contiguity {spatial['contiguity']['frac_bins_in_components_ge50']}, "
              f"posthoc unmatched {transport['posthoc_ad_to_ctrl']['unmatched_mean']}, self_map {transport['posthoc_ad_to_ctrl']['self_map_mean']}")
    json.dump(dict(units=index, cohort="cohort.json"), open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(f"wrote {len(index)} domain packets + cohort.json + index.json -> {a.out}")


if __name__ == "__main__":
    main()
