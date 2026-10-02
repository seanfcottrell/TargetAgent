#!/usr/bin/env python
"""
Multi-agent selection of disease-relevant spatial domains for DEG.

    python run_agents.py --packets <run>/agents_packets --dataset <run>/prompts/dataset_notes.md \
        --out <run>/agents_out/domain_panel          (needs ANTHROPIC_API_KEY)

Flow: five perspective agents (spatial, composition, factor, transport, design)
each score every unit independently from their own evidence blocks; the critic
labels every cited evidence item independent / redundant / explained_away /
unverifiable and counts surviving lines; the synthesis applies the stated rule
and ranks. Specialists may also propose comparisons beyond the default
(counterpart domains across conditions, neighbours within a condition,
interactions, pooled domains); each proposal gets an id (contrast_proposals.json),
the critic judges it sound or unsound, and the synthesis adopts the sound ones
whose side of interest includes a shortlisted domain. A pooling proposal is
judged on its pool packet (<packets>/pools, one per adjacent pair) and, if
sound, stands on its own; an adopted pool replaces its members downstream. Every request, raw response, parsed output and token usage is saved
under --out; a run is fully reproducible from those files.

Every model call goes through agents/common/llm.py `call()` (LangChain):
claude-opus-5 (adaptive thinking on by default), structured JSON output via
output_config.format, server-side refusal fallbacks enabled; the shared system
prompt (generic, dataset-agnostic) is cached with cache_control; dataset notes
and packets travel in the user turn.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
from schemas import AgentReport, CriticReport, SynthesisReport   # noqa: E402
from agents.common.citations import verify, load_session, PubMed                        # noqa: E402
from agents.common.llm import (call, load_env_file, clean_schema, cost, append_usage,    # noqa: E402
                               PUBMED_CACHE, ENV_FILE)

PERSPECTIVES = {
    "spatial":     ["identity", "spatial", "design"],
    "composition": ["identity", "composition", "design"],
    "factor":      ["identity", "factors", "design"],
    "transport":   ["identity", "transport", "design"],
    "design":      ["identity", "spatial", "design"],
}
COHORT_KEYS = {
    "spatial":     ["study", "samples", "design", "confounds"],
    "composition": ["study", "samples", "design", "confounds"],
    "factor":      ["study", "samples", "design", "confounds", "programmes", "programme_semantics"],
    "transport":   ["study", "samples", "design", "confounds"],
    "design":      ["study", "samples", "design", "confounds"],
}
def load_prompt(name):
    """Generic (dataset-agnostic) system prompt for an agent."""
    return open(os.path.join(HERE, "prompts", f"{name}.md")).read().strip()


def load_dataset_notes(path):
    """Dataset-specific notes, sectioned by '## <agent>' headings; '## common' goes to everyone."""
    notes, cur = {}, None
    for line in open(path):
        if line.startswith("## "):
            cur = line[3:].strip(); notes[cur] = []
        elif cur is not None:
            notes[cur].append(line.rstrip("\n"))
    return {k: "\n".join(v).strip() for k, v in notes.items()}


def system_prompt(name):
    return load_prompt("_common") + "\n\n" + load_prompt(name)


def dataset_text(notes, name):
    parts = [notes.get("common", "")]
    if name in notes:
        parts.append(notes[name])
    return "\n\n".join(p_ for p_ in parts if p_)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packets", default=os.path.join(HERE, "packets"))
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--agents", nargs="*", default=list(PERSPECTIVES))
    p.add_argument("--skip-critic", action="store_true")
    p.add_argument("--skip-synthesis", action="store_true")
    p.add_argument("--dataset", required=True,
                   help="dataset-specific notes (user turn), sectioned by '## <agent>'")
    p.add_argument("--web-search", nargs="*", default=["factor"],
                   help="agents that get the server-side web_search tool (literature grounding); "
                        "pass with no value to disable")
    p.add_argument("--max-searches", type=int, default=8,
                   help="web searches allowed per agent call")
    p.add_argument("--env-file", default=ENV_FILE,
                   help="file with ANTHROPIC_API_KEY=... (never committed); env vars take precedence")
    p.add_argument("--resume", action="store_true",
                   help="reuse <agent>.json already present in --out instead of calling again")
    a = p.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)

    idx = json.load(open(os.path.join(a.packets, "index.json")))
    cohort = json.load(open(os.path.join(a.packets, "cohort.json")))
    packets = {u["unit"]: json.load(open(os.path.join(a.packets, u["file"]))) for u in idx["units"]}
    # pooled pairs of adjacent domains (STORM/pool_partition.py pairs): read by the
    # critic to judge pooling proposals; the synthesis sees which pools exist
    pidx = os.path.join(a.packets, "pools", "index.json")
    pool_idx = json.load(open(pidx))["units"] if os.path.isfile(pidx) else []
    pool_packets = [json.load(open(os.path.join(a.packets, "pools", u["file"]))) for u in pool_idx]
    pool_units = [dict(unit=u["unit"], members=u["members"]) for u in pool_idx]
    notes = load_dataset_notes(a.dataset)
    schemas = {"agent": clean_schema(AgentReport.model_json_schema()),
               "critic": clean_schema(CriticReport.model_json_schema()),
               "synthesis": clean_schema(SynthesisReport.model_json_schema())}
    json.dump(schemas, open(os.path.join(out, "schemas.json"), "w"), indent=1)

    load_env_file(a.env_file)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(f"ANTHROPIC_API_KEY is not set: put ANTHROPIC_API_KEY=sk-ant-... in {a.env_file} "
                 f"(chmod 600) or export it in the shell. Nothing was sent.")
    ledger = []

    def run(name, system, user_obj, schema, model_cls):
        ws = name in (a.web_search or [])
        req = dict(agent=name, model=a.model, effort=a.effort, web_search=ws, system=system, user=user_obj)
        json.dump(req, open(os.path.join(out, f"{name}.request.json"), "w"), indent=1)
        with open(os.path.join(out, f"{name}.prompt.md"), "w") as f:
            f.write(f"# SYSTEM\n\n{system}\n\n# USER\n\n```json\n{json.dumps(user_obj, indent=1)}\n```\n")
        approx = (len(system) + len(json.dumps(user_obj))) / 4
        done_path = os.path.join(out, f"{name}.json")
        if a.resume and os.path.isfile(done_path):
            print(f"  [{name}] resumed from {os.path.basename(done_path)} (no call)", flush=True)
            return model_cls.model_validate_json(open(done_path).read())
        t0 = time.time()
        print(f"  [{name}] calling {a.model} (~{approx:.0f} input tokens) ...", flush=True)
        slog = []
        text, usage = call(a.model, system, json.dumps(user_obj), schema, a.max_tokens, a.effort,
                           web_search=a.max_searches if ws else 0, log=slog,
                           fallbacks=True, text="last", check_stop=True, turns=6)
        open(os.path.join(out, f"{name}.raw.txt"), "w").write(text)
        if slog:
            json.dump(slog, open(os.path.join(out, f"{name}.searches.json"), "w"), indent=1)
        parsed = model_cls.model_validate_json(text)
        json.dump(parsed.model_dump(), open(os.path.join(out, f"{name}.json"), "w"), indent=1)
        c = cost(usage, a.model)
        ledger.append(dict(agent=name, **usage, cost_usd=round(c, 4), seconds=round(time.time() - t0, 1)))
        append_usage(out, ledger[-1])
        print(f"  [{name}] done in {time.time()-t0:.0f}s, in={usage['input']} cached={usage['cache_read']} out={usage['output']} searches={usage['searches']} ~${c:.2f}", flush=True)
        return parsed

    # ---- perspectives (independent) ----------------------------------------
    reports = {}
    for name in a.agents:
        blocks = PERSPECTIVES[name]
        user_obj = dict(
            instructions=f"Assess every unit below from the {name} perspective. Return the JSON object described by the schema; one assessment per unit, in the order given.",
            dataset_notes=dataset_text(notes, name),
            cohort={k: cohort[k] for k in COHORT_KEYS[name] if k in cohort},
            units=[{k: v for k, v in packets[u["unit"]].items() if k in ("unit", "domain") or k in blocks} for u in idx["units"]])
        system = system_prompt(name)
        r = run(name, system, user_obj, schemas["agent"], AgentReport)
        if r is not None:
            reports[name] = r.model_dump()

    # ---- contrast proposals: one id each, so critic, synthesis and the selector
    # refer to the same object (select_domains.py checks adoption against these)
    proposals = []
    for n, r in reports.items():
        for asm in r["assessments"]:
            for i, pc in enumerate(asm.get("proposed_contrasts", []), 1):
                proposals.append(dict(id=f"{n}.{asm['unit']}.{i}", agent=n, unit=asm["unit"], **pc))
    if reports:
        json.dump(proposals, open(os.path.join(out, "contrast_proposals.json"), "w"), indent=1)
        print(f"  [contrasts] {len(proposals)} proposed beyond the default", flush=True)

    # ---- citation verification (machine, before the critic reads anything) ----
    verification = None
    if reports:
        items = [dict(agent=n, where=asm["unit"], citation=c) for n, r in reports.items()
                 for asm in r["assessments"] for c in asm.get("citations", [])]
        session = load_session(sorted(glob.glob(os.path.join(out, "*.searches.json"))))
        verification = verify(items, session, PubMed(PUBMED_CACHE))
        json.dump(verification, open(os.path.join(out, "citation_verification.json"), "w"), indent=1)
        print(f"  [citations] {verification['n_citations']} checked against {verification['n_searches']} "
              f"searches: {verification['status_counts']}", flush=True)

    # ---- critic ---------------------------------------------------------------
    critic = None
    if not a.skip_critic and reports:
        critic = run("critic", system_prompt("critic"),
                     dict(instructions="Review every specialist's cited evidence against the packets and cohort; label each item; count surviving lines per unit.",
                          dataset_notes="\n\n".join(notes.values()), cohort=cohort, packets=list(packets.values()),
                          specialist_reports=reports, contrast_proposals=proposals,
                          pool_packets=pool_packets, citation_verification=verification),
                     schemas["critic"], CriticReport)
    # ---- synthesis --------------------------------------------------------------
    synth = None
    if not a.skip_synthesis and critic is not None:
        synth = run("synthesis", system_prompt("synthesis"),
                    dict(instructions="Apply the rule exactly and rank every unit.", dataset_notes=notes.get("common", ""),
                         units=[dict(unit=u["unit"], identity=packets[u["unit"]]["identity"]) for u in idx["units"]],
                         critic_report=critic.model_dump(), specialist_reports=reports,
                         contrast_proposals=proposals, pool_units=pool_units),
                    schemas["synthesis"], SynthesisReport)
    # ---- outputs ---------------------------------------------------------------
    import pandas as pd
    rows = []
    for name, rep in reports.items():
        for asm in rep["assessments"]:
            rows.append(dict(agent=name, unit=asm["unit"], relevance=asm["relevance"], suitability=asm["suitability"],
                             confidence=asm["confidence"], strata=";".join(asm["recommended_strata"]),
                             n_proposed_contrasts=len(asm.get("proposed_contrasts", [])),
                             main_confound=asm["main_confound"], prediction=asm["falsifiable_prediction"]))
    pd.DataFrame(rows).to_csv(os.path.join(out, "assessments.csv"), index=False)
    if synth is not None:
        R = pd.DataFrame([r for r in synth.model_dump()["ranking"]])
        R.to_csv(os.path.join(out, "ranking.csv"), index=False)
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write(f"# Cluster selection report ({a.model}, effort {a.effort})\n\nRule: {synth.rule_applied}\n\n")
            for r in synth.ranking:
                f.write(f"## {r.rank}. {r.unit} -> {r.decision.upper()}  (relevance {r.relevance_consensus:.2f}, suitability {r.suitability_consensus:.2f}, independent lines {r.independent_lines})\n\n{r.rationale}\n\nStrata: {', '.join(r.strata) or 'none'}\n\nPre-registered predictions:\n" + "".join(f"- {p_}\n" for p_ in r.predictions) + "\n")
            f.write("## Additional contrasts\n\n" + ("".join(
                f"- {c.test}: {c.side_a.group} {'+'.join(c.side_a.domains)} vs {c.side_b.group} "
                f"{'+'.join(c.side_b.domains)}, {c.cell_type} (from {', '.join(c.from_proposals)}): {c.question}\n"
                for c in synth.contrasts) or "none adopted\n") + "\n")
            f.write(f"## Caveats\n\n{synth.caveats}\n")
        print("\n" + R[["rank", "unit", "decision", "relevance_consensus", "suitability_consensus", "independent_lines"]].to_string(index=False))
    print(f"\ntotal cost ~${sum(l['cost_usd'] for l in ledger):.2f}; outputs in {out}")


if __name__ == "__main__":
    main()
