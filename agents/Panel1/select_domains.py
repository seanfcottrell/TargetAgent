#!/usr/bin/env python
"""
The domain panel's decision, carried forward mechanically. No judgement here.

Reads the synthesis of the domain panel and writes:

    decisions/domains.json    shortlisted domains in the panel's rank order, the
                              cell-type strata the panel named for each, and the
                              rule that produced them
    decisions/contrasts.json  the DEG contrasts: for every shortlisted domain the
                              whole domain (cell type in the design) and one
                              contrast per named stratum ({domain, type}); then the
                              additional contrasts the panel adopted ({id, test,
                              side_a, side_b, type, ...}), each checked here
                              against the proposals and the critic's verdicts

RULE: a domain goes forward iff the synthesis decision is "shortlist". Its strata
are the synthesis `strata` list, kept only where the name is a prior cell type
from the data card; anything else is recorded under rejected_strata, never
silently dropped. If nothing is shortlisted, nothing is tested: the contrast list
is empty, run_all.py skips every stage that tests domains and goes to the
report, which records the panel's reasons. Testing a domain anyway would be a
human override, which this run does not make.

    python select_domains.py --synthesis <run>/agents_out/domain_panel/synthesis.json \
        --card <card.yaml> --out <run>/decisions
"""
from __future__ import annotations

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import load_card  # noqa: E402


def select(synth: dict, cell_types: list[str]) -> dict:
    rank = sorted(synth["ranking"], key=lambda r: r["rank"])
    chosen, strata, rejected = [], {}, {}
    for r in rank:
        if r["decision"] != "shortlist":
            continue
        d = str(r["unit"]).replace("domain_", "")
        chosen.append(d)
        strata[d] = [t for t in r.get("strata", []) if t in cell_types]
        bad = [t for t in r.get("strata", []) if t not in cell_types]
        if bad:
            rejected[d] = bad
    return dict(
        rule=("domains with synthesis decision 'shortlist', in the panel's rank order; "
              "strata = the synthesis strata that name a prior cell type"),
        shortlisted=chosen, strata=strata, rejected_strata=rejected,
        not_shortlisted={str(r["unit"]).replace("domain_", ""): r["decision"]
                         for r in rank if r["decision"] != "shortlist"})


def contrasts(dec: dict) -> list[dict]:
    out = []
    for d in dec["shortlisted"]:
        out.append({"domain": d, "type": "all"})
        out += [{"domain": d, "type": t} for t in dec["strata"][d]]
    return out + dec.get("designed", [])


def pool_label(members):
    return "+".join(sorted(map(str, members), key=lambda x: (len(x), x)))


def is_pool(c: dict) -> bool:
    """A pooling contrast: the same set of two or more domains on both sides, case vs control."""
    return (c["test"] == "difference" and {c["side_a"]["group"], c["side_b"]["group"]} == {"case", "control"}
            and len(c["side_a"]["domains"]) >= 2
            and set(map(str, c["side_a"]["domains"])) == set(map(str, c["side_b"]["domains"])))


def designed(synth: dict, proposals: list[dict], verdicts: list[dict], shortlisted: list[str],
             units: set[str], cell_types: list[str], presentation: str | None = None,
             pool_sets: set | None = None):
    """The synthesis's additional contrasts, each checked mechanically: it adopts at
    least one proposal the critic judged sound, its side_a holds a shortlisted
    domain, its domains exist, its cell type is 'all' or a prior type, its groups
    fit its test, and (for a difference) no nucleus can fall on both sides.
    Returns (kept, rejected-with-reasons)."""
    prop = {p_["id"]: p_ for p_ in proposals}
    sound = {v["proposal_id"] for v in verdicts if v["verdict"] == "sound"}
    kept, rejected = [], []
    for i, c in enumerate(synth.get("contrasts", []), 1):
        why = []
        ids = c.get("from_proposals", [])
        if not any(x in prop for x in ids):
            why.append("adopts no proposal that was made")
        elif not any(x in sound for x in ids):
            why.append("no adopted proposal was judged sound by the critic")
        dom = lambda side: [str(u).replace("domain_", "") for u in c[side]["domains"]]
        da, db = dom("side_a"), dom("side_b")
        bad = [u for u in c["side_a"]["domains"] + c["side_b"]["domains"] if u not in units]
        if bad:
            why.append(f"unknown domains {bad}")
        if not da or not db:
            why.append("a side has no domain")
        standalone = (is_pool(c) and frozenset(str(u).replace("domain_", "") for u in c["side_a"]["domains"])
                      in (pool_sets or set()))
        if is_pool(c):
            if not standalone:
                why.append("a pool is adopted only on its own pool packet, and this pool has none")
        elif not set(da) & set(shortlisted):
            why.append("side_a holds no shortlisted domain")
        if c["cell_type"] != "all" and c["cell_type"] not in cell_types:
            why.append(f"cell type {c['cell_type']} is not a prior type")
        ga, gb = c["side_a"]["group"], c["side_b"]["group"]
        if c["test"] == "difference":
            if "both" in (ga, gb):
                why.append("a difference compares one group per side")
            elif ga == gb and set(da) & set(db):
                why.append("the sides share a domain in the same group")
            elif ga != gb and len(da) == 1 and da == db:
                why.append("same domain, case vs control: that is the default contrast")
        elif (ga, gb) != ("both", "both"):
            why.append("an interaction takes both groups on both sides")
        elif set(da) & set(db):
            why.append("an interaction needs disjoint domains on the two sides")
        spec = dict(id=f"c{i}", test=c["test"], side_a=dict(group=ga, domains=da),
                    side_b=dict(group=gb, domains=db), type=c["cell_type"],
                    question=c["question"], from_proposals=ids,
                    # a hit is disease dysregulation if case and control are compared, or
                    # if both sides are diseased tissue of a FOCAL disease (lesion vs
                    # neighbour, per the card); otherwise it describes regional difference
                    disease_informative=(c["test"] == "interaction" or ga != gb
                                         or (presentation == "focal" and ga == gb == "case")))
        (rejected.append(dict(**spec, reasons=why)) if why else kept.append(spec))
    return kept, rejected


def resolve_pools(kept, synth, dec, cell_types):
    """A domain shortlisted on its own is analysed on its own: a pool containing a
    shortlisted domain is never adopted (the member is kept, not the pool). A pool
    of domains none of which was shortlisted can be adopted, and then replaces its
    members. Pools must be disjoint: where two share a
    domain, the one with more distinct proposals behind it is kept (ties: the one
    the synthesis listed first) and the other is rejected. A kept pool becomes one
    unit of the final partition, placed at its best-ranked shortlisted member (or
    after the shortlist); its strata are its members' synthesis strata plus the
    cell types of its pooling contrasts. Other designed contrasts that name a
    pooled member are rejected: a member is analysed only as part of its pool."""
    pooled, rejected = [], []
    for c in (c for c in kept if is_pool(c)):
        own = [d for d in c["side_a"]["domains"] if d in dec["shortlisted"]]
        (rejected.append(dict(**c, reasons=[f"member(s) {own} are shortlisted on their own: the member is "
                                            f"analysed, not the pool"])) if own else pooled.append(c))
    others = [c for c in kept if not is_pool(c)]
    support, first = {}, {}
    for i, c in enumerate(pooled):
        P = frozenset(c["side_a"]["domains"])
        support.setdefault(P, []).append(c); first.setdefault(P, i)
    n_prop = lambda P: len({x for c in support[P] for x in c["from_proposals"]})
    chosen = []
    for P in sorted(support, key=lambda P: (-n_prop(P), first[P])):
        clash = [Q for Q in chosen if P & Q]
        if clash:
            rejected += [dict(**c, reasons=[f"overlaps pool {pool_label(clash[0])}, which has more "
                                            f"proposals behind it"]) for c in support[P]]
        else:
            chosen.append(P)
    absorbed = {d: pool_label(P) for P in chosen for d in P}
    rest = []
    for c in others:
        hit = [d for d in c["side_a"]["domains"] + c["side_b"]["domains"] if d in absorbed]
        (rejected.append(dict(**c, reasons=[f"domain(s) {hit} are analysed only as part of pool(s) "
                                            f"{sorted({absorbed[d] for d in hit})}"])) if hit else rest.append(c))
    syn_strata = {str(r["unit"]).replace("domain_", ""): [t for t in r.get("strata", []) if t in cell_types]
                  for r in synth["ranking"]}
    final, strata = [], {}
    for d in dec["shortlisted"]:
        u = absorbed.get(d, d)
        if u not in final:
            final.append(u)
    final += [pool_label(P) for P in chosen if pool_label(P) not in final]
    for u in final:
        P = next((P for P in chosen if pool_label(P) == u), None)
        if P is None:
            strata[u] = dec["strata"][u]
        else:
            st = [t for m in sorted(P, key=lambda x: (len(x), x)) for t in syn_strata.get(m, [])]
            st += [c["type"] for c in support[P] if c["type"] != "all"]
            strata[u] = list(dict.fromkeys(st))
    pools = [dict(label=pool_label(P), members=sorted(P, key=lambda x: (len(x), x)),
                  from_proposals=sorted({x for c in support[P] for x in c["from_proposals"]}),
                  shortlisted_members=[m for m in P if m in dec["shortlisted"]]) for P in chosen]
    return final, strata, pools, absorbed, rest, rejected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthesis", required=True)
    p.add_argument("--card", default=None)
    p.add_argument("--packets", default=None, help="domain packet dir; its pools/index.json lists the "
                                                     "pairs a pooling proposal can stand on")
    p.add_argument("--out", required=True, help="decisions directory")
    a = p.parse_args()
    card = load_card(a.card)
    synth = json.load(open(a.synthesis))
    types = list(card["priors"]["cell_types"])
    dec = select(synth, types)
    dec["source"] = os.path.abspath(a.synthesis)
    os.makedirs(a.out, exist_ok=True)
    # additional contrasts: checked against the proposals and the critic's verdicts
    # written next to the synthesis by run_agents.py
    panel = os.path.dirname(os.path.abspath(a.synthesis))
    prop_path, crit_path = os.path.join(panel, "contrast_proposals.json"), os.path.join(panel, "critic.json")
    proposals = json.load(open(prop_path)) if os.path.isfile(prop_path) else []
    verdicts = json.load(open(crit_path)).get("contrast_verdicts", []) if os.path.isfile(crit_path) else []
    units = {str(r["unit"]) for r in synth["ranking"]}
    pool_idx = os.path.join(a.packets, "pools", "index.json") if a.packets else None
    pool_sets = ({frozenset(u["members"]) for u in json.load(open(pool_idx))["units"]}
                 if pool_idx and os.path.isfile(pool_idx) else set())
    kept, rejected = designed(synth, proposals, verdicts, dec["shortlisted"], units, types,
                              card["description"].get("presentation"), pool_sets)
    final, strata, pools, absorbed, rest, rej2 = resolve_pools(kept, synth, dec, types)
    dec["shortlisted_domains"] = list(dec["shortlisted"])
    dec["shortlisted"], dec["strata"] = final, strata
    dec["pools"], dec["absorbed"] = pools, absorbed
    dec["designed"], dec["rejected_designed"] = rest, rejected + rej2
    dec["designed_rule"] = ("a synthesis contrast is kept iff it adopts a proposal the critic judged sound, "
                            "side_a holds a shortlisted domain, and its domains, cell type and groups are valid "
                            "for its test. A pooling contrast needs a pool packet and is adopted only if none of "
                            "its members is shortlisted (a shortlisted domain is analysed on its own); adopted "
                            "pools are disjoint (more proposals wins) and replace their members")
    json.dump(dict(pools=pools, absorbed=absorbed), open(os.path.join(a.out, "partition.json"), "w"), indent=1)
    json.dump(dec, open(os.path.join(a.out, "domains.json"), "w"), indent=1)
    cs = contrasts(dec)
    json.dump(cs, open(os.path.join(a.out, "contrasts.json"), "w"), indent=1)
    if not dec["shortlisted"]:
        print("THE DOMAIN PANEL SHORTLISTED NO DOMAIN: nothing is tested downstream "
              f"(decisions: {dec['not_shortlisted']}).")
        return
    print(f"shortlisted {dec['shortlisted']}; {len(cs)} contrasts")
    for pl in dec["pools"]:
        print(f"  POOL {pl['label']} replaces domains {pl['members']} (from {', '.join(pl['from_proposals'])})")
    for d in dec["shortlisted"]:
        print(f"  {'pool' if '+' in d else 'domain'} {d}: all + {', '.join(dec['strata'][d]) or '(no strata)'}")
    for c in dec["designed"]:
        print(f"  {c['id']} {c['test']}: {c['side_a']['group']} {'+'.join(c['side_a']['domains'])} vs "
              f"{c['side_b']['group']} {'+'.join(c['side_b']['domains'])}, {c['type']}"
              f"{'' if c['disease_informative'] else ' (regional, not a disease contrast)'}")
    for c in dec["rejected_designed"]:
        print(f"  REJECTED {c['id']}: {'; '.join(c['reasons'])}")
    if dec["rejected_strata"]:
        print(f"  strata not among the prior cell types: {dec['rejected_strata']}")


if __name__ == "__main__":
    main()
