#!/usr/bin/env python
"""
Mechanical consistency check of the report writer's prose against report_data.

The writer is told to cite a field for every number, but a prompt rule is not a
check: a draft can still state "three targets" and list four. This is:

    numbers   every number in the prose must be a value in report_data, allowing
              for rounding to the precision written and for percentages. Years,
              and digits inside identifiers (SLC1A2, NCT01739348, PF00001), are
              ignored.
    counts    a stated count followed by a list -- "three targets (A, B, C, D)",
              "two: A and B" -- must equal the number of items listed.

It flags; it does not rewrite. run_report.py hands the flags back to the writer
for one revision, re-checks, and prints whatever is still flagged in the report.

    python check_report.py --report <run>/report/report_writer.json --data <run>/report/report_data.json
"""
from __future__ import annotations

import argparse, json, re

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
NUM = re.compile(r"(?<![A-Za-z0-9_.,\-])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%?)(?![A-Za-z0-9_])")
# A stated count, a short run of words (no digits: "went to train-alone", "BBB
# rules"), then a list in parentheses or after a colon. The count may not be part
# of a decimal or an identifier ("0.78", "domain 5", "F1", "rank 18").
COUNT_LIST = re.compile(
    r"(?<![\d.,])(?<!domain )(?<!domains )(?<!rank )(?<!stratum )\b(" + "|".join(WORDS) + r"|\d+)\b"
    r"(?![.,]\d)(?!\s+of\b)([^.;:()\d|]{1,24}?)(?:\(|:\s)([^().;|]{3,400}?)(?:\)|[.;]|$)", re.I)
ANY_NUM = re.compile(r"\d+(?:\.\d+)?")


def data_numbers(obj, out=None):
    """Every numeric value in report_data, including numbers written inside strings."""
    out = set() if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for t in ANY_NUM.findall(obj.replace(",", "")):   # loose: "91-95" gives 91 and 95
            out.add(float(t))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            data_numbers(k, out); data_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            data_numbers(v, out)
    return out


def supported(x, decimals, pct, pool):
    for v in pool:
        for cand in ((v, v * 100) if pct else (v,)):
            if round(cand, decimals) == round(x, decimals) or \
                    (decimals == 0 and abs(cand - x) < 0.5 and abs(cand) >= 1):
                return True
    return False


def split_items(s):
    s = re.sub(r"\s+(and|or)\s+", ",", s.strip())
    items = [t.strip(" ,") for t in s.split(",")]
    return [t for t in items if t]


def check(sections: dict, data: dict, notes: str = "") -> list:
    """`notes` is the dataset-notes text the writer also receives: a legitimate source."""
    pool = data_numbers(data)
    data_numbers(notes, pool)
    issues = []
    for key, text in sections.items():
        if not isinstance(text, str):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            for m in NUM.finditer(sent):
                whole, frac, pct = m.group(1), m.group(2) or "", m.group(3)
                x = float(whole.replace(",", "") + frac)
                if not frac and not pct and 1900 <= x <= 2100:
                    continue
                if x in (0.0, 1.0, 2.0) and not frac and not pct:
                    continue                  # too common to test meaningfully
                if not supported(x, len(frac) - 1 if frac else 0, bool(pct), pool):
                    issues.append(dict(section=key, kind="number_not_in_data",
                                       value=m.group(0), sentence=sent.strip()[:300]))
            for m in COUNT_LIST.finditer(sent):
                if m.group(0).rstrip().endswith(";"):
                    continue                  # a list broken by ';' is not safely countable
                word = m.group(1).lower()
                n = WORDS.get(word, int(word) if word.isdigit() else None)
                items = split_items(m.group(3))
                # only lists of short tokens (names), not a clause that happens to follow a number
                # a LIST is of names: items that start with a letter and are short. A
                # parenthesis of figures ("(3 verified, 2 verified_session)") is not one.
                if n is None or n < 2 or not (2 <= len(items) <= 30) or \
                        any(len(t.split()) > 4 or not t[0].isalpha() for t in items):
                    continue
                if n != len(items):
                    issues.append(dict(section=key, kind="count_mismatch", stated=n, listed=len(items),
                                       items=items, sentence=sent.strip()[:300]))
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, help="report_writer.json")
    p.add_argument("--data", required=True, help="report_data.json")
    p.add_argument("--notes", default=None, help="the dataset notes the writer received")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    notes = open(a.notes).read() if a.notes else ""
    issues = check(json.load(open(a.report)), json.load(open(a.data)), notes)
    for i in issues:
        print(json.dumps(i)[:400])
    print(f"{len(issues)} issue(s)")
    if a.out:
        json.dump(issues, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
