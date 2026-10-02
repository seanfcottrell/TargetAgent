#!/usr/bin/env python
"""
Driver for the TARGET panel: three axis specialists, then direction and
tractability, then a critic and a synthesis.

Evidence flow, and where knowledge comes from:

  dysregulation / programme / topology
      read ONLY their own packet blocks. Everything they say comes from this
      analysis. Isolation is the `blocks` filter in AXIS_BLOCKS -- an agent
      literally cannot see another axis's numbers, which is what makes "two
      agents agreed" mean something to the critic.

  direction
      composes. WHICH process is broken and which way it moves comes from the
      packet (process_direction, straight out of the factor shifts); the gene's
      VALENCE in that process cannot come from this analysis at all and must be
      established from literature, so this agent gets web search. Expression
      direction is given to it only as a cross-check.

  critic / synthesis
      see the packets and the reports.

Chemistry is NOT assessed here. Whether compounds exist against a target gates
nothing at this stage -- it cannot veto a biological case, and the ranking does
not move on it -- so asking now would buy no decision.

The single seam: every model call goes through agents/common/llm.py `call()`
(LangChain); nothing else here knows about the framework. Keep it that way.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
COMMON = os.path.join(REPO, "agents", "common")
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
from schemas_targets import (AxisReport, DirectionReport,                       # noqa: E402
                             TargetCriticReport, TargetSynthesisReport)
from agents.common.citations import verify, load_session, PubMed               # noqa: E402
from agents.common.llm import (call, dataset_notes, clean_schema, cost, append_usage,   # noqa: E402
                               load_env_file, PUBMED_CACHE, ENV_FILE)

# Which packet blocks each axis specialist may see. This IS the isolation.
AXIS_BLOCKS = {
    "dysregulation": ["axis1_dysregulation"],
    "programme":     ["axis2_programme"],
    "topology":      ["axis3_topology"],
}
ALWAYS = ["gene", "domain", "support"]


# Roles whose prompt file is not named after the role: critic.md and
# synthesis.md belong to the DOMAIN panel.
ROLE_PROMPT = {"critic": "target_critic", "synthesis": "target_synthesis"}


def sysprompt(name):
    """Generic, dataset-free. _common_target.md is shared by every role."""
    common = open(os.path.join(COMMON, "prompts", "_common_target.md")).read().strip()
    role = open(os.path.join(HERE, "prompts", f"{ROLE_PROMPT.get(name, name)}.md")).read().strip()
    return common + "\n\n---\n\n" + role


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packets", default=os.path.join(HERE, "packets_targets"))
    p.add_argument("--dataset", required=True, help="rendered dataset notes")
    p.add_argument("--out", required=True)
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--agents", nargs="+",
                   default=["dysregulation", "programme", "topology",
                            "direction", "critic", "synthesis"])
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="high")
    p.add_argument("--max-tokens", type=int, default=64000)  # streaming, so headroom is cheap
    p.add_argument("--batch", type=int, default=15, help="candidates per call")
    p.add_argument("--direction-batch", type=int, default=None,
                   help="candidates per direction call (default --batch); smaller batches give "
                        "each candidate more of the call's searches")
    p.add_argument("--web-search", type=int, default=6, help="max searches for direction")
    p.add_argument("--resume", action="store_true",
                   help="reuse <agent>.json already in --out instead of calling again "
                        "(a failed run then re-pays only what did not finish)")
    p.add_argument("--only", nargs="*", default=None, metavar="GENE@DOMAIN",
                   help="restrict to these candidates and MERGE into an existing "
                        "<agent>.json in --out. For repairing dropped assessments "
                        "without paying to re-run a whole agent.")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cohort = json.load(open(os.path.join(a.packets, "cohort.json")))
    index = json.load(open(os.path.join(a.packets, "index.json")))
    domains = a.domains or sorted(index)
    packs = {d: json.load(open(os.path.join(a.packets, f"domain_{d}.json"))) for d in domains}

    load_env_file(ENV_FILE)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set and not found in agents/common/.env")

    usage_rows = []

    def stage(name, user_obj, model_cls, tools=None, web_search=0, suffix=""):
        system = sysprompt(name)
        tag = name + suffix
        req = dict(agent=name, model=a.model, system=system, user=user_obj,
                   tools=[t.get("name") for t in (tools or [])], web_search=web_search)
        json.dump(req, open(os.path.join(a.out, f"{tag}.request.json"), "w"), indent=1)
        with open(os.path.join(a.out, f"{tag}.prompt.md"), "w") as f:
            # Header, because this file looks editable and is not: it is
            # regenerated from prompts/ on every run and edits made here are
            # silently discarded.
            f.write(f"<!-- GENERATED FILE - DO NOT EDIT.\n"
                    f"     Rendered from agents/common/prompts/_common_target.md + "
                    f"agents/Panel2/prompts/{name}.md\n"
                    f"     and the rendered dataset notes. Edit those; this is overwritten "
                    f"on every run. -->\n\n")
            f.write(f"# SYSTEM\n\n{system}\n\n# USER\n\n```json\n"
                    f"{json.dumps(user_obj, indent=1)}\n```\n")
        approx = (len(system) + len(json.dumps(user_obj))) // 4
        done = os.path.join(a.out, f"{tag}.json")
        if a.resume and os.path.isfile(done):
            print(f"[{tag}] resumed from {os.path.basename(done)} (no call)", flush=True)
            return json.load(open(done))
        print(f"[{tag}] ~{approx:,} input tokens", flush=True)
        t0 = time.time(); tlog = []
        text, usage = call(a.model, system, json.dumps(user_obj),
                           clean_schema(model_cls.model_json_schema()), a.max_tokens, a.effort,
                           tools=tools, web_search=web_search, log=tlog)
        open(os.path.join(a.out, f"{tag}.raw.txt"), "w").write(text)
        if tlog:
            json.dump(tlog, open(os.path.join(a.out, f"{tag}.tools.json"), "w"), indent=1)
        obj = model_cls.model_validate_json(text)
        json.dump(obj.model_dump(), open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
        usage_rows.append(dict(agent=tag, model=a.model, seconds=round(time.time() - t0, 1),
                               **usage, cost_usd=round(cost(usage, a.model), 4)))
        append_usage(a.out, usage_rows[-1])
        print(f"[{tag}] ok  {usage['output']:,} out, {usage['tool_calls']} tool calls, "
              f"{time.time() - t0:.0f}s", flush=True)
        return obj.model_dump()

    def candidates(blocks=None, extra=None, programme_shift=False, only=None):
        """Candidate list per domain, narrowed to the blocks this agent may see.

        `programme_shift` describes the domain's disease state in terms of the
        programmes and their effect sizes -- that IS the programme axis's
        evidence. It is withheld from the dysregulation and topology agents, or
        they would be reasoning from another axis's numbers and the critic could
        no longer tell independent agreement from shared input.
        """
        units = []
        for d in domains:
            pk = packs[d]
            keep = ALWAYS + (blocks or []) + (extra or [])
            u = dict(domain=d, dominant_cell_type=pk.get("dominant_cell_type"),
                     candidates=[{k: v for k, v in c.items() if k in keep}
                                 for c in pk["candidates"]
                                 if only is None or (d, c["gene"]) in only])
            if programme_shift:
                u["programme_shift"] = pk["programme_shift"]
            if "axis3_topology" in (blocks or []):
                # one network per upheld programme: sizes only, no effect sizes
                u["topology_networks"] = pk.get("topology_networks", [])
            if u["candidates"]:
                units.append(u)
        return units

    def stage_batched(name, build_user, model_cls, key, note_key, batch=None, **kw):
        """Run a per-candidate agent over chunks and merge.

        A single call cannot hold every candidate: the structured reply is one
        object per candidate, and at 40+ candidates the reply is truncated
        mid-string and the whole batch is lost. Chunking bounds each reply, and
        each chunk's raw text is written separately so a failure costs one chunk
        rather than the agent.
        """
        merged_path = os.path.join(a.out, f"{name}.json")
        if a.resume and not a.only and os.path.isfile(merged_path):
            print(f"[{name}] resumed from {name}.json (no call)", flush=True)
            return json.load(open(merged_path))
        sel = None
        if a.only:
            sel = {tuple(x.rsplit("@", 1)[::-1]) for x in a.only}   # (domain, gene)
        chunks, buf = [], []
        for d in domains:
            for c in packs[d]["candidates"]:
                if sel is not None and (d, c["gene"]) not in sel:
                    continue
                buf.append((d, c["gene"]))
                if len(buf) >= (batch or a.batch):
                    chunks.append(buf); buf = []
        if buf:
            chunks.append(buf)
        merged, notes_all = [], []
        for i, ch in enumerate(chunks, 1):
            want = {(d, g) for d, g in ch}
            # NEVER let a chunk write the canonical <agent>.json: stage() writes
            # "{name}{suffix}.json", so an empty suffix clobbers the merged file
            # before the merge happens. In repair mode that destroyed a complete
            # 166-assessment file and left only the two repaired rows.
            sfx = ".fill" if a.only else ("" if len(chunks) == 1 and not a.only
                                          else f".b{i}")
            obj = stage(name, build_user(want), model_cls, suffix=sfx, **kw)
            merged += obj[key]
            if obj.get(note_key):
                notes_all.append(obj[note_key])

        if a.only:                       # repair mode: merge into what exists
            prev_path = os.path.join(a.out, f"{name}.json")
            prev = json.load(open(prev_path)) if os.path.isfile(prev_path) else None
            if prev:
                have = {(str(x.get("domain")), x.get("gene")) for x in merged}
                kept = [x for x in prev[key]
                        if (str(x.get("domain")), x.get("gene")) not in have]
                merged = kept + merged
                out = {"agent": name, key: merged,
                       note_key: prev.get(note_key, "")}
                json.dump(out, open(prev_path, "w"), indent=1)
                print(f"[{name}] repaired: {len(merged)} assessments "
                      f"({len(merged) - len(kept)} new)", flush=True)
                return out

        # COVERAGE CHECK. A structured reply of N objects sometimes comes back
        # with N-1: the model drops or duplicates a candidate, and nothing about
        # the response says so. A missing assessment is indistinguishable
        # downstream from a candidate that was never proposed, so verify and
        # back-fill rather than trust the count.
        want = {(d, c["gene"]) for d in domains for c in packs[d]["candidates"]}
        # The reverse also happens: a reply can carry an item for something that
        # was never a candidate (a filler entry). It must not reach the critic.
        stray = [x for x in merged if (str(x.get("domain")), x.get("gene")) not in want]
        if stray:
            print(f"[{name}] dropped {len(stray)} assessment(s) for non-candidates: "
                  f"{', '.join(str(x.get('gene')) + '@' + str(x.get('domain')) for x in stray[:8])}", flush=True)
            merged = [x for x in merged if (str(x.get("domain")), x.get("gene")) in want]
        got = {(str(x.get("domain")), x.get("gene")) for x in merged}
        missing = sorted(want - got)
        if missing:
            print(f"[{name}] MISSING {len(missing)} of {len(want)}: "
                  f"{', '.join(g + '@' + d for d, g in missing[:8])}"
                  f"{' …' if len(missing) > 8 else ''}", flush=True)
            obj = stage(name, build_user(set(missing)), model_cls,
                        suffix=".fill", **kw)
            if obj:
                merged += obj[key]
                got = {(str(x.get("domain")), x.get("gene")) for x in merged}
                still = sorted(want - got)
                print(f"[{name}] after back-fill: "
                      + (f"STILL MISSING {len(still)}: {still}" if still
                         else "complete"), flush=True)
        dupes = len(merged) - len({(str(x.get("domain")), x.get("gene")) for x in merged})
        if dupes:
            print(f"[{name}] {dupes} duplicate assessment(s) returned", flush=True)

        out = {"agent": name, key: merged, note_key: "\n\n".join(notes_all)}
        json.dump(out, open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
        print(f"[{name}] merged {len(merged)} assessments from {len(chunks)} batch(es)",
              flush=True)
        return out

    def _only_domain(report, d):
        """Narrow a specialist report to one domain, so the critic audits the
        same slice of evidence it is shown packets for."""
        if not isinstance(report, dict) or "assessments" not in report:
            return report
        return {**report, "assessments": [x for x in report["assessments"]
                                          if str(x.get("domain")) == str(d)]}

    reports = {}
    for name in ("dysregulation", "programme", "topology"):
        if name not in a.agents:
            continue
        reports[name] = stage_batched(
            name,
            lambda only, name=name: dict(
                instructions=(f"Assess every candidate below from the {name} axis only. "
                              "Return one assessment per (gene, domain) in the order given."),
                dataset_notes=dataset_notes(a.dataset, name),
                cohort=cohort,
                units=candidates(AXIS_BLOCKS[name],
                                 programme_shift=(name == "programme"), only=only)),
            AxisReport, "assessments", "cross_candidate_notes")

    if "direction" in a.agents:
        reports["direction"] = stage_batched(
            "direction",
            lambda only: dict(
            instructions=("For every candidate, derive the intervention direction in the "
                          "three steps your contract defines. Work out from programme_shift "
                          "what each programme is and whether its movement is harmful; "
                          "establish the gene's valence in that process from the literature; "
                          "then compose. Expression is a cross-check only."),
            dataset_notes=dataset_notes(a.dataset, "direction"),
            cohort=cohort,
                units=candidates(["axis1_dysregulation", "axis2_programme", "axis3_topology"],
                                 extra=["identity"], programme_shift=True, only=only)),
            DirectionReport, "assessments", "notes", web_search=a.web_search,
            batch=a.direction_batch)

    # ---- citation verification (machine, before the critic reads anything) ----
    verification = "<computed at run time from the direction agent's citations>"
    if isinstance(reports.get("direction"), dict):
        items = [dict(agent="direction", where=f"{x['gene']}@{x['domain']}", citation=c)
                 for x in reports["direction"]["assessments"] for c in x.get("citations", [])]
        session = load_session(sorted(glob.glob(os.path.join(a.out, "direction*.tools.json"))))
        verification = verify(items, session, PubMed(PUBMED_CACHE))
        json.dump(verification, open(os.path.join(a.out, "citation_verification.json"), "w"), indent=1)
        print(f"[citations] {verification['n_citations']} checked against {verification['n_searches']} "
              f"searches: {verification['status_counts']}", flush=True)

    def verification_for(d=None):
        if not isinstance(verification, dict):
            return verification
        keep = [c for c in verification["citations"] if d is None or c["where"].endswith(f"@{d}")]
        return {**{k: v for k, v in verification.items() if k != "citations"}, "citations": keep}

    if "critic" in a.agents and a.resume \
            and os.path.isfile(os.path.join(a.out, "critic.json")):
        reports["critic"] = json.load(open(os.path.join(a.out, "critic.json")))
        print("[critic] resumed from critic.json (no call)", flush=True)
    elif "critic" in a.agents:
        # Batched BY DOMAIN. The critic must return a verdict per cited item, so
        # over the whole pool its reply is several times larger than any single
        # agent's and would truncate. Per-candidate cross-domain caution is
        # preserved because each packet carries that gene's calls in the other
        # domains (axis1.other_domains / n_domains_hit).
        crit = []
        for d in domains:
            obj = stage("critic", dict(
            instructions=("Audit every cited item against the packets. Label each verdict, "
                          "count surviving independent lines, and discharge your three "
                          "duties: novelty penalties, ungrounded directions, disqualifying "
                          "concerns."),
            dataset_notes=dataset_notes(a.dataset, "critic"),
                cohort=cohort, packets=[packs[d]],
                reports={k: _only_domain(v, d) for k, v in reports.items()},
                citation_verification=verification_for(d)),
                TargetCriticReport, suffix=("" if len(domains) == 1 else f".d{d}"))
            if obj:
                crit.append(obj)
        if crit:
            merged = dict(agent="critic",
                          candidates=[c for o in crit for c in o["candidates"]],
                          general_notes="\n\n".join(o["general_notes"] for o in crit))
            json.dump(merged, open(os.path.join(a.out, "critic.json"), "w"), indent=1)
            reports["critic"] = merged
            print(f"[critic] merged {len(merged['candidates'])} audited candidates", flush=True)

    if "synthesis" in a.agents:
        stage("synthesis", dict(
            instructions=("Rank the candidates on biological_case; chemistry is not "
                          "assessed at this stage. Populate single_axis_strong_mechanism "
                          "and novel_candidates."),
            dataset_notes=dataset_notes(a.dataset, "synthesis"),
            cohort=cohort, reports=reports,
            citation_verification=(
                {k: v for k, v in verification.items() if k != "citations"}
                | {"not_supporting": [c for c in verification["citations"] if not c["supports"]]}
                if isinstance(verification, dict) else verification)), TargetSynthesisReport)

    if usage_rows:
        import pandas as pd
        print("\n" + pd.DataFrame(usage_rows).to_string(index=False))
        print(f"this invocation ~${sum(r['cost_usd'] for r in usage_rows):.2f}; "
              f"full ledger in {os.path.join(a.out, 'usage.csv')}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
