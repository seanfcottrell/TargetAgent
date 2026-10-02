from __future__ import annotations

import argparse, json, os, re
import numpy as np, pandas as pd

# A usage-centred movement citation: the path must be followed by .F<digit>.
QUALIFY = re.compile(r"factors\.usage_centred_ad_vs_ctrl\.F\d")
FTOKEN = re.compile(r"\bF(\d)\b")
# The critic's wording when an item is upheld as accurate but counted for
# neither relevance nor irrelevance.
NEUTRAL = re.compile(r"neutral|neither side|counted for neither", re.I)


def derive_subsets(agents_dir, domains):
    crit = json.load(open(os.path.join(agents_dir, "critic.json")))
    subsets, record = {}, {}
    for u in crit["units"]:
        dom = str(u["unit"]).replace("domain_", "")
        if dom not in domains:
            continue
        keep, log = set(), []
        for v in u.get("verdicts", []):
            fld, verdict = v.get("field", ""), v.get("verdict", "")
            reason = v.get("reason", "")
            if not QUALIFY.search(fld):
                if FTOKEN.search(fld):
                    log.append(dict(field=fld, verdict=verdict, used=False,
                                    why="not a usage-centred per-programme movement citation"))
                continue
            if verdict != "independent":
                log.append(dict(field=fld, verdict=verdict, used=False,
                                why=f"critic verdict is {verdict!r}, not independent"))
                continue
            if NEUTRAL.search(reason):
                log.append(dict(field=fld, verdict=verdict, used=False,
                                why=f"critic neutralised it: {reason.strip()[:120]}"))
                continue
            fs = sorted({int(x) for x in FTOKEN.findall(fld)})
            keep.update(fs)
            log.append(dict(field=fld, verdict=verdict, used=True,
                            programmes=[f"F{r}" for r in fs]))
        subsets[dom] = sorted(keep)
        record[dom] = log
    return subsets, record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="STORM fit dir (B_gene_loadings.npy, genes.csv)")
    p.add_argument("--progact", required=True, help="program_activity dir (contrast_usage_centred.csv)")
    p.add_argument("--agents", required=True, help="agent run dir (critic.json)")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--top", type=int, default=300)
    p.add_argument("--subsets", default=None, help='JSON override, e.g. {"1":[4],"2":[4,8]}')
    p.add_argument("--partition", default=None,
                   help="decisions/partition.json: a pooled unit (e.g. '1+3') takes the union of its "
                        "members' upheld programmes, weighted by the POOL's own usage shift")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    B = np.load(os.path.join(a.run, "B_gene_loadings.npy"))
    genes = pd.read_csv(os.path.join(a.run, "genes.csv")).gene.values
    cu = pd.read_csv(os.path.join(a.progact, "contrast_usage_centred.csv"))
    cu["r"] = cu.program.str.lstrip("F").astype(int)
    assert B.shape[0] == len(genes), (B.shape, len(genes))

    pools = {pl["label"]: pl["members"] for pl in
             (json.load(open(a.partition)).get("pools", []) if a.partition and os.path.isfile(a.partition) else [])}
    members = {m for d in a.domains for m in pools.get(d, [])}
    derived, record = derive_subsets(a.agents, set(a.domains) | members)
    for d in a.domains:
        if d in pools:        # the panel assessed members, never the pool as a unit
            derived[d] = sorted({r for m in pools[d] for r in derived.get(m, [])})
            record[d] = [dict(pool_of=pools[d], used=True, programmes=[f"F{r}" for r in derived[d]],
                              why="union of the members' upheld usage-centred programmes")]
    for m in members - set(a.domains):
        derived.pop(m, None); record.pop(m, None)
    subsets = {k: list(v) for k, v in derived.items()}
    if a.subsets:
        subsets.update({str(k): list(v) for k, v in json.loads(a.subsets).items()})

    os.makedirs(a.out, exist_ok=True)
    prov = dict(run=a.run, progact=a.progact, agents=a.agents, top=a.top,
                shift_column="contrast_usage_centred.csv:cohen_d",
                derived_subsets=derived, used_subsets=subsets,
                overridden=bool(a.subsets), citations=record, domains={})

    for dom in a.domains:
        sub = subsets.get(dom, [])
        if not sub:
            print(f"[domain {dom}] NO surviving programme citation -- skipped", flush=True)
            continue
        d = cu[cu.domain.astype(str) == dom].sort_values("r").cohen_d.values
        contrib = B[:, sub] * d[sub]                    # genes x |subset|
        charge = contrib.sum(1)
        absum = np.abs(contrib).sum(1)
        order = np.argsort(-np.abs(charge))[:a.top]

        out = pd.DataFrame({"gene": genes[order], "charge": charge[order],
                            "abs_charge": np.abs(charge[order]),
                            "cancellation": np.where(absum[order] > 0,
                                                     np.abs(charge[order]) / np.where(absum[order] > 0, absum[order], 1),
                                                     np.nan)})
        for j, r in enumerate(sub):
            out[f"contrib_F{r}"] = contrib[order, j]
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        out.to_csv(os.path.join(a.out, f"domain{dom}_charge.csv"), index=False)

        npos = int((out.charge > 0).sum())
        prov["domains"][dom] = dict(
            programmes=[f"F{r}" for r in sub],
            d={f"F{r}": float(d[r]) for r in sub},
            n_nodes=len(out), n_positive=npos, n_negative=len(out) - npos,
            median_cancellation=float(np.nanmedian(out.cancellation)),
            top20=out.gene.head(20).tolist(), networks={})
        print(f"[domain {dom}] combined " + " ".join(f"F{r}{d[r]:+.2f}" for r in sub) +
              f" -> {len(out)} nodes, {npos}+/{len(out)-npos}-, "
              f"median cancellation {np.nanmedian(out.cancellation):.3f}", flush=True)
        print(f"           top: {', '.join(out.gene.head(10))}", flush=True)

        # One network per upheld programme: the topology axis's input.
        for r in sub:
            pr = f"F{r}"
            if d[r] == 0:
                print(f"[domain {dom} {pr}] usage shift is exactly 0: no network (zero charge is "
                      f"a degenerate sheaf)", flush=True)
                continue
            ch = B[:, r] * d[r]
            o = np.argsort(-np.abs(ch))[:a.top]
            o = o[np.abs(ch[o]) > 0]                       # a zero-loading gene is not a node
            net = pd.DataFrame({"rank": np.arange(1, len(o) + 1), "gene": genes[o],
                                "charge": ch[o], "abs_charge": np.abs(ch[o]), "loading": B[o, r]})
            net.to_csv(os.path.join(a.out, f"domain{dom}_{pr}_charge.csv"), index=False)
            shared = {f"F{q}": int(len(set(net.gene) & set(genes[np.argsort(-B[:, q])[:a.top]])))
                      for q in sub if q != r}
            prov["domains"][dom]["networks"][pr] = dict(
                d=float(d[r]), n_nodes=len(net), sign="negative" if d[r] < 0 else "positive",
                nodes_shared_with=shared, top20=net.gene.head(20).tolist())
            print(f"[domain {dom} {pr}] d {d[r]:+.2f} -> {len(net)} nodes"
                  + (f"; nodes shared with " + ", ".join(f"{k} {v}" for k, v in shared.items())
                     if shared else "") + f"; top: {', '.join(net.gene.head(8))}", flush=True)

    json.dump(prov, open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
    print(f"\nwrote {a.out}: domain<d>_charge.csv (programme axis), domain<d>_F<r>_charge.csv (one network per programme), provenance.json")


if __name__ == "__main__":
    main()
