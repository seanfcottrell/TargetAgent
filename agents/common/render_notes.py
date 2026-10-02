#!/usr/bin/env python
"""
Dataset notes for every agent, RENDERED from the data card and the method
defaults. Nothing here is typed by hand for a particular dataset.

The notes say what the data IS (tissue, technology, disease, cohort, the prior
cell types and domain count, how the analysed objects were made) and the rules
every agent follows. They never say what the analysis FOUND: no statistics, no
reading of any programme, domain or gene, no named nuisance. Those are for the
agents to establish from the packets.

The output uses the `## <section>` layout the agent drivers read. Only
`## common` is written, so every agent receives the same description.

    python render_notes.py --card datasets/<name>/card.yaml --out <run>/prompts/dataset_notes.md
"""
from __future__ import annotations

import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import load_card, load_methods  # noqa: E402


PRESENTATION = {
    "diffuse": ("Disease is therefore read by comparing case donors with control donors; a "
                "comparison within the diseased sections describes regional difference, not disease."),
    "focal": ("Comparisons within the diseased sections (affected tissue against its unaffected "
              "neighbour) are therefore disease comparisons, alongside case against control donors."),
}


def render(card: dict, methods: dict) -> str:
    d, cond = card["description"], card["condition"]
    samples = card["samples"]
    ctrl_label = str(cond["control_label"])
    case = [s for s in samples if str(s["stage"]) != ctrl_label]
    ctrl = [s for s in samples if str(s["stage"]) == ctrl_label]
    case_stages = sorted({str(s["stage"]) for s in case})
    ct = card["priors"]["cell_types"]
    ctm, st, pr = methods["celltype"], methods["storm"], methods["prep"]
    gene_pool = {"coding": "HGNC protein-coding", "all": "annotated"}.get(st["gene_pool"], st["gene_pool"])

    pres = d.get("presentation")
    if pres not in PRESENTATION:
        raise SystemExit(f"card description.presentation must be one of {sorted(PRESENTATION)} "
                         f"(how the disease is laid out in tissue); got {pres!r}")
    lines = [
        "## common",
        f"Tissue: {d['tissue']}. Disease: {d['disease']}. Technology: {d['technology']} "
        f"Measurement: {d['measurement']}. Design: {d['design']}.",
        "",
        f"Disease presentation: {pres}. {d.get('presentation_note', '').strip()} {PRESENTATION[pres]}".rstrip(),
        "",
        f"Cohort: {len(samples)} sections from {len(samples)} donors: {len(case)} "
        f"{cond['case_group']} (stage label{'s' if len(case_stages) > 1 else ''} "
        f"{', '.join(repr(x) for x in case_stages)}) and {len(ctrl)} {cond['control_group']} "
        f"(stage label '{ctrl_label}', a literal string). Per-section batch identifiers "
        f"(chip lot, GEF writer version), demographics and pathology scores are in the "
        f"cohort data you are given.",
        "",
        "How the analysed objects were made (definitions, not results):",
        f"- Nuclei with at least {ctm['min_genes_per_nucleus']} detected genes were clustered "
        f"and typed by a consensus of three independent callers (CellTypist "
        f"{ctm['celltypist_models'][0]} on pseudo-bulk pseudo-cells, correlation with Allen "
        f"human cortex SMART-seq profiles, and marker-set scores over the prior markers "
        f"below), then collapsed to the prior cell types. Nuclei in clusters the callers "
        f"could not resolve are labelled Unk.",
        f"- Tiles are bin{pr['bin_size']} bins with at least {pr['min_counts_per_bin']} counts.",
        f"- Gene programmes come from a rank-{st['rank']} graph-regularised PARAFAC2 "
        f"factorisation (STORM) over the {st['n_top_genes']} most variable {gene_pool} genes; "
        f"that gene set is the universe of every programme-based quantity.",
        f"- Spatial domains are a Leiden clustering of programme usage over an embedding "
        f"kNN graph united with the tile lattice, at the resolution whose domain count "
        f"matches the prior below (or comes closest to it).",
        "",
        "Prior cell types (names and marker genes supplied with the dataset):",
    ]
    for name, spec in ct.items():
        lines.append(f"- {name} ({spec['label']}): {', '.join(spec['markers'])}")
    lines += ["", f"Prior number of spatial domains: {card['priors']['n_domains']}.", ""]
    if card.get("policies"):
        lines.append("Rules:")
        lines += [f"- {p.strip()}" for p in card["policies"]]
    return "\n".join(lines).rstrip() + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", default=None)
    p.add_argument("--methods", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    text = render(load_card(a.card), load_methods(a.methods))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(text)
    print(text)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
