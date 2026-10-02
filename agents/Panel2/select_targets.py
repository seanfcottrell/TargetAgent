#!/usr/bin/env python
"""
The target panel's decision, carried forward on the merit of the DATA. No
judgement here: a stated rule applied to the critic's verdicts.

Why this exists. The critic labels every cited item, and a verified literature
citation can be labelled `independent` just as a packet field can. Its
`independent_lines` count then mixes the two, so a target with one data line
and one good paper reads as "two lines" and can reach the convergent tier, and
a target whose only upheld support is literature can reach an admitted tier
at all. Selection must rest on what THIS analysis measured. Literature keeps
the job only it can do -- the direction (inhibit or activate) and the harm
assessment -- and never makes a target eligible.

THE RULE
    data line   one of the three data axes (dysregulation, programme, topology)
                for which the critic labelled at least one of that axis's cited
                items `independent`, in that (gene, domain). Items from the
                direction agent (literature) are never data lines. Several
                independent items on one axis are one line.
    data_lines  per gene, the maximum over the domains in which it was audited
    tier        from the data alone:  >= 2 convergent, 1 single_axis_strong_mechanism,
                0 not eligible. The panel's own tier does NOT cap it: a gene the
                panel held for want of literature is not discounted here.
    excluded    only for a DATA-BASED concern: the panel rejected it with a named
                reason, or the critic recorded a disqualifying concern (an
                artefact class, broad essentiality). "Under-studied", "no
                literature" and publication volume are never reasons.
    direction   the panel's, which literature legitimately grounds. It is carried
                as context and gates nothing.
                Targets whose direction is undetermined are marked
                `direction_open` and go forward like the rest.

What literature may do: ground a stated pathway role or disease association
(a claim without a verified citation is unverifiable), and set direction. What
it may not do: make a target eligible, raise its tier, or -- by its absence --
lower either.

Writes <out> with the same "ranking" shape as synthesis.json (so it replaces it
as the selection downstream stages read), plus every excluded or demoted
target with its reason, and the per-(gene, domain) line table behind it.

    python select_targets.py --synthesis <run>/agents_out/target_panel/synthesis.json \
        --critic <run>/agents_out/target_panel/critic.json \
        --tiers convergent single_axis_strong_mechanism --out <run>/decisions/targets.json
"""
from __future__ import annotations

import argparse, json, os
from datetime import datetime, timezone

DATA_AXES = ("dysregulation", "programme", "topology")
ORDER = {"rejected": 0, "hold": 1, "single_axis_strong_mechanism": 2, "convergent": 3}


def data_tier(n):
    return "convergent" if n >= 2 else ("single_axis_strong_mechanism" if n == 1 else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthesis", required=True)
    p.add_argument("--critic", required=True)
    p.add_argument("--tiers", nargs="+", default=["convergent", "single_axis_strong_mechanism"])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    syn = json.load(open(a.synthesis))
    critic = json.load(open(a.critic))["candidates"]

    lines = []                  # per (gene, domain): which data axes survived, and what else the critic said
    for c in critic:
        ind = [v for v in c.get("verdicts", []) if v.get("verdict") == "independent"]
        axes = sorted({v["agent"] for v in ind if v.get("agent") in DATA_AXES})
        lines.append(dict(gene=c["gene"], domain=str(c["domain"]), data_axes=axes, data_lines=len(axes),
                          literature_independent=sum(1 for v in ind if v.get("agent") not in DATA_AXES),
                          critic_independent_lines=c.get("independent_lines"),
                          disqualifying_concern=c.get("disqualifying_concern")))
    # A critic concern is about one gene IN ONE DOMAIN. A gene counts in the
    # domains without a concern; it is excluded only when every domain that
    # gives it a data line carries one.
    best, disq = {}, {}
    for r in lines:
        if r["disqualifying_concern"]:
            disq.setdefault(r["gene"], []).append(f"domain {r['domain']}: {r['disqualifying_concern']}")
            continue
        if r["gene"] not in best or r["data_lines"] > best[r["gene"]]["data_lines"]:
            best[r["gene"]] = r
    only_flagged = {g for g in disq if g not in best or best[g]["data_lines"] == 0}

    admitted, direction_open, excluded = [], [], []
    for x in sorted(syn["ranking"], key=lambda r: r["rank"]):
        g, panel_tier = x["gene"], x["tier"]
        b = best.get(g, dict(data_lines=0, data_axes=[], domain=None))
        tier = data_tier(b["data_lines"])
        rec = dict(x, panel_tier=panel_tier, tier=tier or "not_eligible", data_lines=b["data_lines"],
                   data_axes=b["data_axes"], data_lines_domain=b["domain"])
        if g in only_flagged:
            excluded.append(dict(rec, reason="critic's disqualifying concern in every domain with a data line: "
                                             + " | ".join(disq[g])))
            continue
        if tier is None:
            if panel_tier in a.tiers:
                excluded.append(dict(rec, reason="no data axis upheld as independent by the critic; "
                                                 "its only support is literature"))
            continue
        if panel_tier == "rejected":
            excluded.append(dict(rec, reason=f"panel rejected it: {x.get('rejection_reason')}"))
            continue
        if g in disq:
            rec["concern_elsewhere"] = disq[g]      # counted only in its unflagged domains
        if tier not in a.tiers:
            excluded.append(dict(rec, reason=f"data tier {tier} is outside {a.tiers}"))
            continue
        if tier != panel_tier:
            rec["panel_tier_differs"] = panel_tier
        if x.get("direction") == "undetermined":
            rec["direction_open"] = True
            direction_open.append(rec)
        else:
            admitted.append(rec)

    out = dict(
        made=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rule=("tier from DATA lines only: one line per data axis (dysregulation, programme, topology) "
              "with at least one critic-independent item; >=2 convergent, 1 single_axis_strong_mechanism, "
              "0 not eligible; the panel's tier does not cap it; excluded only for a data-based concern "
              "(panel rejection with a named reason, critic's disqualifying concern); literature grounds "
              "claims and sets direction, never eligibility; publication volume is never considered"),
        tiers=a.tiers, source_synthesis=os.path.abspath(a.synthesis), source_critic=os.path.abspath(a.critic),
        ranking=admitted + direction_open,
        direction_open=direction_open, excluded=excluded, lines=lines)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    n_panel = sum(1 for x in syn["ranking"] if x["tier"] in a.tiers)
    print(f"panel put {n_panel} targets in {a.tiers}; admitted on data merit: {len(admitted)}"
          f" (+{len(direction_open)} with direction open; direction gates nothing)")
    for r in admitted + direction_open:
        diff = f"   panel said {r['panel_tier_differs']}" if "panel_tier_differs" in r else ""
        opn = "   DIRECTION OPEN" if r in direction_open else ""
        print(f"  {r['gene']:9s} {r['tier']:29s} data lines {r['data_lines']} ({', '.join(r['data_axes'])}){diff}{opn}")
    for r in excluded:
        print(f"  EXCLUDED {r['gene']:9s} panel {r['panel_tier']}: {r['reason'][:150]}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
