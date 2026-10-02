#!/usr/bin/env python
"""
The report writer: one agent that turns report_data.json into the final write-up.

It reads only what assemble_report.py collected -- every stage's results and
every panel's decisions -- and the dataset notes. It has no web search: the
report records what this analysis established, and literature enters only as
the panels already argued it, attributed to them.

Output: <out>/report.md = the written sections, then the full results tables
(report_tables.md) as an appendix, so every sentence can be checked against a
number on the same page. The model call goes through agents/common/llm.py call(),
the pipeline's single model-call seam (LangChain).

    python run_report.py --data <run>/report/report_data.json \
        --tables <run>/report/report_tables.md --dataset <notes.md> --out <run>/report
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
from schemas_report import FinalReport                                       # noqa: E402
from check_report import check                                               # noqa: E402
from agents.common import llm                                                # noqa: E402
from agents.common.llm import clean_schema, load_env_file, cost, append_usage, ENV_FILE  # noqa: E402

SECTIONS = [("data_and_cohort", "Data and cohort"),
            ("spatial_domains", "The domains and their programmes"),
            ("domain_panel", "Panel 1: domain selection"),
            ("measured_evidence", "Measured evidence: expression and topology"),
            ("target_panel", "Panel 2: the targets"),
            ("biological_narrative", "Does it converge?"),
            ("caveats_and_limitations", "Caveats and limitations")]


def render(rep: dict, tables_md: str, meta: dict) -> str:
    title = rep.get("title") or "Spatial transcriptomic target identification: report"
    out = [f"# {title}\n", f"_{meta}_\n", "## Summary\n", rep["executive_summary"].strip(), ""]
    for key, head in SECTIONS:
        out += [f"## {head}\n", rep[key].strip(), ""]
    body = tables_md.split("\n", 1)[1] if tables_md.startswith("# ") else tables_md
    out += ["---\n", "# Appendix: results tables\n", body.replace("\n## ", "\n### ")]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tables", required=True)
    p.add_argument("--dataset", required=True, help="rendered dataset notes")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="high")
    p.add_argument("--max-tokens", type=int, default=64000)
    p.add_argument("--no-revise", action="store_true", help="check only; do not send flags back")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    system = open(os.path.join(HERE, "prompts", "report.md")).read().strip()
    user = dict(instructions=("Write the final report of this analysis from report_data, following your "
                              "rules. Cover every section of the schema; cite fields for every number."),
                dataset_notes=llm.dataset_notes(a.dataset, "report"),
                report_data=json.load(open(a.data)))
    tag = "report_writer"
    json.dump(dict(agent=tag, model=a.model, system=system, user=user),
              open(os.path.join(a.out, f"{tag}.request.json"), "w"), indent=1)
    print(f"[{tag}] ~{(len(system) + len(json.dumps(user))) // 4:,} input tokens", flush=True)

    done = os.path.join(a.out, f"{tag}.json")
    raw = os.path.join(a.out, f"{tag}.raw.txt")
    if a.resume and os.path.isfile(done):
        rep = json.load(open(done))
        print(f"[{tag}] resumed from {os.path.basename(done)} (no call)", flush=True)
    elif a.resume and os.path.isfile(raw):
        # A reply that was paid for but failed validation: parse it again under the
        # current contract rather than paying for another call.
        rep = FinalReport.model_validate_json(open(raw).read()).model_dump()
        json.dump(rep, open(done, "w"), indent=1)
        print(f"[{tag}] re-parsed the saved reply {os.path.basename(raw)} (no call)", flush=True)
    else:
        load_env_file(ENV_FILE)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set and not found in agents/common/.env")
        t0 = time.time()
        text, usage = llm.call(a.model, system, json.dumps(user),
                               clean_schema(FinalReport.model_json_schema()), a.max_tokens, a.effort)
        open(raw, "w").write(text)
        # Record the spend BEFORE validating: a reply that fails validation was
        # still paid for, and must not vanish from the ledger.
        row = dict(agent=tag, model=a.model, seconds=round(time.time() - t0, 1), **usage,
                   cost_usd=round(cost(usage, a.model), 4))
        append_usage(a.out, row)
        rep = FinalReport.model_validate_json(text).model_dump()
        json.dump(rep, open(done, "w"), indent=1)
        print(f"[{tag}] ok  {usage['output']:,} out, ~${row['cost_usd']:.2f}", flush=True)

    # CONSISTENCY: every number must be in the data, every stated count must match
    # its list. Flags go back to the writer for ONE revision; whatever survives it
    # is printed in the report, so a mismatch is never silently published.
    data = json.load(open(a.data))
    notes_text = open(a.dataset).read() if os.path.isfile(a.dataset) else ""
    issues = check(rep, data, notes_text)
    rev_done = os.path.join(a.out, f"{tag}.revised.json")
    if issues and not a.no_revise:
        if a.resume and os.path.isfile(rev_done):
            rep = json.load(open(rev_done))
            print(f"[{tag}.revise] resumed (no call)", flush=True)
        else:
            load_env_file(ENV_FILE)
            rev_user = dict(
                instructions=("A mechanical check compared your draft with report_data and found the issues "
                              "below: a number that is not in report_data or the dataset notes, or a stated "
                              "count that does not match the items listed after it. For each, correct the "
                              "statement to match the data. If a flagged number is correct because it is "
                              "derived (a sum, a difference, a count of listed items), keep it and state the "
                              "derivation in the sentence. Change nothing else. Return the full report."),
                issues=issues, draft=rep, dataset_notes=llm.dataset_notes(a.dataset, "report"),
                report_data=data)
            t0 = time.time()
            text, usage = llm.call(a.model, system, json.dumps(rev_user),
                                   clean_schema(FinalReport.model_json_schema()), a.max_tokens, a.effort)
            open(os.path.join(a.out, f"{tag}.revised.raw.txt"), "w").write(text)
            append_usage(a.out, dict(agent=tag + ".revise", model=a.model, seconds=round(time.time() - t0, 1),
                                     **usage, cost_usd=round(cost(usage, a.model), 4)))
            rep = FinalReport.model_validate_json(text).model_dump()
            json.dump(rep, open(rev_done, "w"), indent=1)
            print(f"[{tag}.revise] {len(issues)} flag(s) sent back; revised", flush=True)
        issues_after = check(rep, data, notes_text)
    else:
        issues_after = issues
    json.dump(dict(before_revision=issues, after_revision=issues_after),
              open(os.path.join(a.out, "report_check.json"), "w"), indent=1)
    print(f"[{tag}] consistency check: {len(issues)} flag(s) on the draft, "
          f"{len(issues_after)} after revision", flush=True)

    meta = (f"Written by {a.model} from report_data.json on "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}; every number cites its field; "
            f"the tables in the appendix are the computed results")
    md = render(rep, open(a.tables).read(), meta)
    if issues_after:
        note = ["", "---", "", "## Consistency check: statements still flagged", "",
                "A mechanical check compares every number and stated count in the text above with "
                "report_data. These remained flagged after one revision; read them against the tables.", ""]
        note += [f"- **{i['section']}**: " + (f"number {i['value']} not found in the data"
                                               if i["kind"] == "number_not_in_data" else
                                               f"states {i['stated']} but lists {i['listed']}")
                 + f' -- "{i["sentence"][:200]}"' for i in issues_after]
        head, sep, tail = md.partition("\n---\n")
        md = head + "\n".join(note) + "\n" + sep + tail
    open(os.path.join(a.out, "report.md"), "w").write(md)
    print(f"wrote {os.path.join(a.out, 'report.md')}")


if __name__ == "__main__":
    main()
