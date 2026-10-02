#!/usr/bin/env python
from __future__ import annotations

import argparse, glob, json, os, shutil, subprocess, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
UTILS = os.path.join(REPO, "Utils")
sys.path.insert(0, os.path.join(HERE, "STORM"))   # the STORM package: bare imports, as it uses itself
sys.path.insert(0, REPO)                          # repo root FIRST: `Utils` must not resolve to STORM/Utils
from Utils.domain_utils import read_meta, spatial_coherence    # noqa: E402
from storm_chain_run import triage                             # noqa: E402

GLOBAL_PROGACT = ("program_enrichment.json", "program_enrichment.csv")   # label-independent


def sh(cwd, script, *args):
    cmd = [sys.executable, "-u", script, *map(str, args)]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def pool_label(members):
    return "+".join(sorted(map(str, members), key=lambda s: (len(s), s)))


def relabel(run, base_clusters, pools):
    lab = pd.read_csv(os.path.join(run, base_clusters))["cluster"].astype(str).to_numpy().copy()
    for members in pools:
        lab[np.isin(lab, [str(m) for m in members])] = pool_label(members)
    return lab


def recompute(a, lab, work, packets_for=None):
    """Tables for one relabelled partition under `work`; with packets_for, also the
    packet of those domain ids."""
    os.makedirs(work, exist_ok=True)
    clusters = os.path.join(work, "bin_clusters.csv")
    pd.DataFrame(dict(cluster=lab)).to_csv(clusters, index=False)
    sh(UTILS, "compose_domains.py", "--cells", a.cells, "--storm", a.run,
       "--storm-clusters", clusters, "--out", os.path.join(work, "composition"))
    pa = os.path.join(work, "program_activity")
    sh(os.path.join(REPO, "FeatureSelection", "FactorAnnotation"), "program_activity.py", "--run", a.run, "--clusters", clusters, "--out", pa)
    for f in GLOBAL_PROGACT:
        if os.path.isfile(os.path.join(a.progact, f)):
            shutil.copy(os.path.join(a.progact, f), os.path.join(pa, f))
    if packets_for is None:
        return clusters
    meta = read_meta(a.run, clusters)
    _, coh = spatial_coherence(meta, k=8)
    coh.index.name = "cl"
    coh.to_csv(os.path.join(work, "coherence.csv"))
    um_path = os.path.join(a.run, "unmatched_per_bin.npy")
    um = np.load(um_path) if os.path.isfile(um_path) else np.zeros(len(meta))
    stages = sorted(set(meta.stage))
    order = [s for s in a.stage_order if s in stages] + [s for s in stages if s not in a.stage_order]
    triage(meta, meta.cl.to_numpy(), um, order, 2, 50).to_csv(os.path.join(work, "triage.csv"), index=False)
    for kind, coup in (("transport_chain", None), ("transport_posthoc", os.path.join(a.run, "couplings_posthoc"))):
        sh(os.path.join(REPO, "Integration", "OptimalTransport"), "transport_tables.py", "--run", a.run, "--clusters", clusters,
           *(["--couplings", coup] if coup else []), "--out", os.path.join(work, kind))
    sh(os.path.join(REPO, "agents", "Panel1"), "build_packets.py", "--run", a.run, "--card", a.card,
       "--composition", os.path.join(work, "composition"), "--progact", pa,
       "--clusters-file", clusters, "--triage-file", os.path.join(work, "triage.csv"),
       "--coherence-file", os.path.join(work, "coherence.csv"), "--transport-dir", work,
       "--only-units", *packets_for, "--out", os.path.join(work, "packets"))
    return clusters


def adjacent_pairs(packets, min_share):
    """Pairs of packet domains sharing >= min_share of either one's tile boundary in
    either group (spatial.adjacent_domains)."""
    idx = json.load(open(os.path.join(packets, "index.json")))
    doms = [str(u["unit"]).replace("domain_", "") for u in idx["units"]]
    share = {}
    for d in doms:
        P = json.load(open(os.path.join(packets, f"domain_{d}.json")))
        for g, row in P["spatial"]["adjacent_domains"]["by_group"].items():
            for o, v in row.items():
                if o in doms and o != d and v is not None:
                    k = frozenset((d, o))
                    share.setdefault(k, {})[f"{d}->{o} ({g})"] = v
    return {k: v for k, v in share.items() if max(v.values()) >= min_share}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["pairs", "final"])
    p.add_argument("--run", required=True, help="fitted run directory")
    p.add_argument("--res", required=True, help="resolution of the base partition")
    p.add_argument("--cells", required=True)
    p.add_argument("--progact", required=True, help="base program_activity dir (global enrichment files)")
    p.add_argument("--card", default=None)
    p.add_argument("--packets", help="pairs: base packet dir; pool packets go to <packets>/pools")
    p.add_argument("--min-share", type=float, default=0.1)
    p.add_argument("--partition", help="final: decisions/partition.json")
    p.add_argument("--out-root", help="final: run root (writes partition_final/)")
    a = p.parse_args()
    sys.path.insert(0, REPO)
    from paths import load_card
    card = load_card(a.card)
    a.card = a.card or os.environ.get("WALKTHROUGH_CARD")
    a.stage_order = [str(x) for x in card["condition"].get("stage_order", [card["condition"]["control_label"]])]
    base = f"bin_clusters_QH_sccg_res{a.res}.csv"

    if a.mode == "pairs":
        out = os.path.join(a.packets, "pools")
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        index = []
        for k, sh_ in sorted(adjacent_pairs(a.packets, a.min_share).items(), key=lambda kv: pool_label(kv[0])):
            label = pool_label(k)
            print(f"\n=== pool {label}: boundary shares {sh_}", flush=True)
            work = os.path.join(out, "work", label)
            recompute(a, relabel(a.run, base, [sorted(k)]), work, packets_for=[label])
            src = os.path.join(work, "packets", f"domain_{label}.json")
            if not os.path.isfile(src):
                print(f"  no packet for {label} (below the packet floor)", flush=True)
                continue
            P = json.load(open(src))
            P.update(unit=f"pool_{label}", domain=label, members=sorted(k, key=lambda s: (len(s), s)),
                     member_boundary_share=sh_)
            json.dump(P, open(os.path.join(out, f"pool_{label}.json"), "w"), indent=1)
            index.append(dict(unit=f"pool_{label}", file=f"pool_{label}.json", members=P["members"],
                              n_bins=P["identity"]["n_bins"]))
        json.dump(dict(units=index, min_share=a.min_share,
                       rule="pairs of packet domains sharing >= min_share of a tile boundary in either group"),
                  open(os.path.join(out, "index.json"), "w"), indent=1)
        print(f"\nwrote {len(index)} pool packets -> {out}")
        return

    part = json.load(open(a.partition))
    out = os.path.join(a.out_root, "partition_final")
    pools = [pl["members"] for pl in part.get("pools", [])]
    if not pools:
        print("no pools adopted: the final partition is the base partition")
        json.dump(dict(pooled=False, clusters=os.path.join(a.run, base)),
                  open(os.path.join(os.path.dirname(a.partition), "partition_final.json"), "w"), indent=1)
        return
    if os.path.isdir(out):
        shutil.rmtree(out)
    clusters = recompute(a, relabel(a.run, base, pools), out)
    info = dict(pooled=True, pools=part["pools"], clusters=clusters,
                composition=os.path.join(out, "composition"),
                program_activity=os.path.join(out, "program_activity"))
    json.dump(info, open(os.path.join(os.path.dirname(a.partition), "partition_final.json"), "w"), indent=1)
    print(f"final partition with pools {[pool_label(m) for m in pools]} -> {out}")


if __name__ == "__main__":
    main()
