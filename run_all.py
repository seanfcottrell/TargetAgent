#!/usr/bin/env python
"""
End-to-end runner: a data card -> cell types, spatial domains, gene programmes,
differential expression, network topology, targets -> a written report. Every
judgement between stages is made by an agent panel or by a stated rule; none is
made by hand.

    python run_all.py --card datasets/ad_pfc_stereoseq/card.yaml --run run1 --list
    python run_all.py --card ... --run run1 --submit         # run it as a SLURM driver job
    python run_all.py --card ... --run run1                  # run in this shell instead
    python run_all.py --card ... --run run1 --submit --allow-paid   # agent stages included
    python run_all.py --card ... --run run1 --from deg       # redo from a stage onward
    python run_all.py --card ... --run run1 --until packets  # stop after a stage
    python run_all.py --card ... --run run1 --graph          # the stage graph, as Mermaid

INPUTS. The data card (datasets/<name>/card.yaml) is the only description of the
data: tissue, technology, disease, the sections and their metadata, the STORM
links between sections, the prior cell types with markers, the prior number of
domains. Method defaults are config/methods.yaml. Agents
see the card only through notes rendered from it (agents/common/render_notes.py).

OUTPUTS. Everything lands in runs/<dataset>/<run>/, so runs never overwrite one
another. Decisions that pass between stages are files in <run>/decisions/:
resolution.json (the domain count search), domains.json + contrasts.json (the
domain panel's shortlist, carried forward mechanically). The final write-up is
<run>/report/report.md.

EXECUTION. Heavy stages are SLURM jobs; the runner waits for each to finish and
reads its final state. Light stages run in
this process's allocation via `conda run -n <env>`. `--submit` runs the whole
runner as one SLURM job (the driver); stage jobs are submitted from inside it.
The gateway blocks the `sbatch` shell function; this calls /usr/bin/sbatch.

DONE MEANS DONE. A stage is skipped when <run>/.state/<stage>.done exists and
its outputs are present. Timestamps are never compared, so touching an upstream
file cannot silently re-run a paid agent stage. `--from` / `--only` / `--force`
re-run explicitly; agent stages pass --resume to their drivers, so a re-run
pays only for calls that did not finish (`--fresh-agents` to re-pay).

THE GRAPH. The run is a LangGraph state graph (build_graph): one node per stage,
in pipeline order, with two branches. After select_domains the run goes straight
to the report if the domain panel shortlisted nothing, skipping every stage that
tests domains; and a paid stage ends the run when --allow-paid was not given.
The graph state records only how the run went (RunState). Results never travel
through it: stages read and write files under the run directory, and the done
markers on disk are what make a re-launch resume, so no checkpointer is used.
"""
from __future__ import annotations

import argparse, glob, json, operator, os, re, shlex, shutil, subprocess, sys, time
from datetime import datetime, timezone
from typing import Annotated, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    raise SystemExit("run_all.py needs langgraph (see envs/st-graph-parafac2-gpu.yml): "
                     "pip install langgraph langchain-anthropic")

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from paths import load_yaml, METHODS  # noqa: E402

SBATCH, SQUEUE, SACCT = "/usr/bin/sbatch", "/usr/bin/squeue", "/usr/bin/sacct"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ctx:
    """Everything a stage needs to build its commands. Decisions are read
    lazily, when the stage that needs them runs."""

    def __init__(self, card_path, methods_path, run_id):
        self.card_path, self.methods_path = os.path.abspath(card_path), os.path.abspath(methods_path)
        self.card, self.m = load_yaml(self.card_path), load_yaml(self.methods_path)
        self.run_id = run_id
        self.dataset = self.card["dataset"]
        self.run = os.path.join(REPO, "runs", self.dataset, run_id)
        cond = self.card["condition"]
        self.secs = [str(s["section"]) for s in self.card["samples"]]
        self.case = [str(s["section"]) for s in self.card["samples"]
                     if str(s["stage"]) != str(cond["control_label"])]
        st = self.m["storm"]
        self.fit = self.r("storm_out", f"n{len(self.secs)}_r{st['rank']}_g{st['gamma']:g}_"
                                       f"{st['within_graph']}_{st['cluster_rep']}_{run_id}")
        self.notes = self.r("prompts", "dataset_notes.md")

    def r(self, *p):
        return os.path.join(self.run, *p)

    @property
    def conda(self):
        """The conda executable, as an absolute path for job scripts: envs.conda of
        the methods file if it is a path or on PATH, else $CONDA_EXE (set inside
        any conda environment, so running this from one is enough)."""
        exe = shutil.which(str(self.m["envs"]["conda"])) or os.environ.get("CONDA_EXE")
        if not exe or not os.path.isfile(exe):
            raise SystemExit("cannot find conda: set envs.conda in the methods file to its path, "
                             "or run run_all.py from an activated conda environment")
        return os.path.abspath(exe)

    def env(self):
        e = dict(os.environ)
        e.update(WALKTHROUGH_RUN=self.run, WALKTHROUGH_CARD=self.card_path,
                 WALKTHROUGH_METHODS=self.methods_path,
                 WALKTHROUGH_MARKERS=self.r("inputs", "markers.json"))
        return e

    def decision(self, name, key):
        path = self.r("decisions", f"{name}.json")
        if not os.path.isfile(path):
            raise SystemExit(f"decision {name}.json missing: run the stage that makes it first")
        return json.load(open(path))[key]

    def res(self):
        return self.decision("resolution", "resolution_str")

    def clusters(self):
        return f"bin_clusters_QH_sccg_res{self.res()}.csv"

    def comp(self):
        return self.r("composition", f"res{self.res()}")

    def final(self):
        """decisions/partition_final.json (final_partition stage): where the partition
        downstream of the domain panel lives. Adopted pools replace their members;
        without pools it is the base partition."""
        path = self.r("decisions", "partition_final.json")
        return json.load(open(path)) if os.path.isfile(path) else {"pooled": False}

    def fclusters(self):
        f = self.final()
        return f["clusters"] if f.get("pooled") else self.clusters()

    def fcomp(self):
        f = self.final()
        return f["composition"] if f.get("pooled") else self.comp()

    def fprogact(self):
        f = self.final()
        return f["program_activity"] if f.get("pooled") else self.r("program_activity")

    def shortlist(self):
        return self.decision("domains", "shortlisted")

    def charged(self):
        """Shortlisted domains that got a sheaf charge (some programme movement was
        upheld for them); the others carry the DEG axis only."""
        return [d for d in self.shortlist()
                if os.path.isfile(self.r("sheaf", "charges", f"domain{d}_charge.csv"))]

    def agents(self):
        return self.m["agents"]

    def no_shortlist(self):
        """The domain panel shortlisted nothing: every stage that tests domains is
        skipped, and the run goes straight to the report, which says so."""
        path = self.r("decisions", "domains.json")
        return os.path.isfile(path) and not json.load(open(path))["shortlisted"]


# --------------------------------------------------------------------------- stages
# A stage is (name, kind, env, build, done, note), plus needs_shortlist for the
# stages that test shortlisted domains. build(ctx) returns a list of
# steps; a step is ("py", cwd, script, args) or ("fn", callable). kind is
# "local" (this process's allocation), "slurm" (its own job), "storm" (the
# STORM job writer's own job script) or "paid" (local; calls the model API).

def py(cwd, script, *args):
    return ("py", cwd, script, [str(x) for x in args])


def fn(f):
    return ("fn", f)


def stage_inputs(c: Ctx):
    card = c.card
    if len(set(c.secs)) != len(c.secs):
        raise SystemExit("card: duplicate section names")
    for l in card.get("storm_links", []):
        if l["src"] not in c.secs or l["dst"] not in c.secs or l.get("kind", "within") not in ("within", "cross"):
            raise SystemExit(f"card: bad storm link {l}")
    for name, spec in card["priors"]["cell_types"].items():
        if not spec.get("markers"):
            raise SystemExit(f"card: cell type {name} has no markers")
    if int(card["priors"]["n_domains"]) < 2:
        raise SystemExit("card: n_domains must be >= 2")
    inp, bs = card["inputs"], c.m["prep"]["bin_size"]
    for sub, key, pat in (("cellbin", "cellbin_dir", "{s}_cellbin.h5ad"),
                          ("bin110", "bin_dir", "{s}_bin%d.h5ad" % bs)):
        os.makedirs(c.r(sub), exist_ok=True)
        for s in c.secs:
            src = os.path.join(REPO, inp[key], pat.format(s=s))
            if not os.path.isfile(src):
                raise SystemExit(f"missing input {src}: convert it first (Utils/build_cellbin.py, "
                                 f"Utils/build_bin110.py)")
            dst = os.path.join(c.r(sub), os.path.basename(src))
            if not os.path.lexists(dst):
                os.symlink(src, dst)
    os.makedirs(c.r("gef"), exist_ok=True)
    for s in card["samples"]:
        hits = glob.glob(os.path.join(REPO, inp["gef_dir"], f"*_{s['chip']}.cellbin.gef"))
        if not hits:
            print(f"  note: no cellbin.gef for {s['section']}; its tiles get no nuclear morphology")
        for h in hits:
            dst = os.path.join(c.r("gef"), os.path.basename(h))
            if not os.path.lexists(dst):
                os.symlink(h, dst)
    os.makedirs(c.r("inputs"), exist_ok=True)
    cols = ["section", "chip", "stage", "chip_lot", "gef_writer", "sex", "age", "apoe",
            "thal", "cerad", "braak"]
    with open(c.r("inputs", "samples.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for s in card["samples"]:
            fh.write(",".join(str(s.get(k, "")) for k in cols) + "\n")
    json.dump({k: v["markers"] for k, v in card["priors"]["cell_types"].items()},
              open(c.r("inputs", "markers.json"), "w"), indent=1)
    json.dump(card["storm_links"], open(c.r("inputs", "edges.json"), "w"), indent=1)
    shutil.copy(os.path.join(os.path.dirname(c.card_path), card["priors"]["reference_label_map"]),
                c.r("inputs", "coarse_map.csv"))
    shutil.copy(c.card_path, c.r("card.yaml"))
    shutil.copy(c.methods_path, c.r("methods.yaml"))
    print(f"  {len(c.secs)} sections linked: {', '.join(c.secs)} (case: {', '.join(c.case)})")


def write_manifest(c: Ctx):
    m = dict(card=c.card_path, prepped=c.r("prepped"), celltypes_qc=c.r("celltypes_qc"),
             fit=c.fit, resolution_decision=c.r("decisions", "resolution.json"),
             composition=c.fcomp(), progact=c.fprogact(),
             partition=c.r("decisions", "partition_final.json"),
             domain_panel=c.r("agents_out", "domain_panel"),
             decisions_domains=c.r("decisions", "domains.json"), deg=c.r("deg_out"),
             decisions_targets=c.r("decisions", "targets.json"),
             sheaf=c.r("sheaf"), target_panel=c.r("agents_out", "target_panel"),
             out=c.r("report"))
    os.makedirs(c.r("report"), exist_ok=True)
    json.dump(m, open(c.r("report", "manifest.json"), "w"), indent=1)


def paid(c, script_cwd, script, out, args):
    """An agent driver. Resumable: a re-run pays only for calls that did not finish."""
    real = list(args) + ["--out", out]
    if not ARGS.fresh_agents:
        real.append("--resume")
    return [py(script_cwd, script, *real)]


def build_stages():
    S = []
    gated = False           # stages added after select_domains need a shortlist

    def add(name, kind, env, build, done, note):
        S.append(dict(name=name, kind=kind, env=env, build=build, done=done, note=note,
                      needs_shortlist=gated))

    add("inputs", "local", "storm", lambda c: [fn(stage_inputs)],
        lambda c: [c.r("inputs", "samples.csv"), c.r("inputs", "edges.json")],
        "validate the card; link the card's sections; write samples, markers, links")

    add("prep", "slurm", "storm", lambda c: [py("Utils", "prep_bins.py",
        "--bindir", c.r("bin110"), "--bin-size", c.m["prep"]["bin_size"], "--gefdir", c.r("gef"),
        "--samples", c.r("inputs", "samples.csv"), "--outdir", c.r("prepped"),
        "--min-counts", c.m["prep"]["min_counts_per_bin"])],
        lambda c: [c.r("prepped", "prep_summary.csv")],
        "gene intersection over the card's sections, empty-tile floor, metadata from the card")

    add("celltype", "slurm", "storm", lambda c: [
        py("CellBinAnalysis", "celltype_nuclei.py", "--indir", c.r("cellbin"), "--out", c.r("celltypes_qc"),
           "--min-genes", c.m["celltype"]["min_genes_per_nucleus"],
           "--resolution", c.m["celltype"]["leiden_resolution"], "--plots"),
        py("CellBinAnalysis", "annotate_clusters.py", "--run", c.r("celltypes_qc"), "--indir", c.r("cellbin"),
           "--models", *c.m["celltype"]["celltypist_models"],
           "--coarse-map", c.r("inputs", "coarse_map.csv"), "--plots")],
        lambda c: [c.r("celltypes_qc", "cells.csv.gz"), c.r("celltypes_qc", "composition_by_section.csv")],
        "nucleus QC, Harmony, Leiden; three callers vote onto the card's cell types")

    def storm_cmd(c, phase):
        st = c.m["storm"]
        res = st["slurm_graph" if phase == "graph" else "slurm_fit"]
        graph = c.r("storm_out", "graph")
        extra = (["--graph-only", "--save-graph", graph] if phase == "graph"
                 else ["--load-graph", graph])
        return [py("Integration/STORM", "write_storm_job.py", "--jobdir", c.r("jobs"), "--job-label", phase,
                   "--sections", *c.secs, "--edges", c.r("inputs", "edges.json"), "--tag", c.run_id,
                   "--rank", st["rank"], "--gamma", st["gamma"],
                   "--iters", st["iters"], "--n-niches", st["n_niches"], "--n-top-genes", st["n_top_genes"],
                   "--gene-pool", st["gene_pool"], "--hvg-tiebreak", st["hvg_tiebreak"],
                   "--within-graph", st["within_graph"], "--expr-k", st["expr_k"],
                   "--lambda-cross", st["lambda_cross"], "--cluster-rep", st["cluster_rep"],
                   "--ppi-min-score", st["ppi_min_score"], "--string-version", st["string_version"],
                   "--resolutions", *st["resolutions"], "--time", res["time"],
                   "--mem", res["mem"], "--cpus", res["cpus"], "--gpu", res["gpu"], *extra)]
    add("storm_graph", "storm", "storm", lambda c: storm_cmd(c, "graph"),
        lambda c: [c.r("storm_out", "graph", "graph_key.json"), c.r("storm_out", "graph", "L_s.npz")],
        "STORM graph: per-section kNN U lattice, FGW couplings between sections (CPU, ~5 h)")
    add("storm_fit", "storm", "storm", lambda c: storm_cmd(c, "fit"),
        lambda c: [os.path.join(c.fit, "shared_coords_QH.npy"), os.path.join(c.fit, "B_gene_loadings.npy")],
        "graph-regularised PARAFAC2 on the saved graph (GPU, minutes)")

    add("domains", "slurm", "storm", lambda c: [
        py("Integration/STORM", "select_domain_resolution.py", "--run", c.fit, "--rep", c.m["domains"]["rep"],
           "--cluster-graph", c.m["domains"]["cluster_graph"], "--resolutions", *c.m["domains"]["grid"],
           "--target-n", c.card["priors"]["n_domains"], "--refine-steps", c.m["domains"]["refine_steps"],
           "--write-all"),
        py("Integration/STORM", "finish_domains.py", "--run", c.fit, "--cells", c.r("celltypes_qc", "cells.csv.gz"),
           "--composition-root", c.r("composition"), "--decision", c.r("decisions", "resolution.json"),
           "--target-n", c.card["priors"]["n_domains"])],
        lambda c: [c.r("decisions", "resolution.json")],
        "resolution searched for the card's domain count; map, composition, decision")

    add("progact", "local", "storm", lambda c: [py("FeatureSelection/FactorAnnotation", "program_activity.py", "--run", c.fit,
                                                   "--clusters", c.clusters(), "--out", c.r("program_activity"))],
        lambda c: [c.r("program_activity", "contrast_usage_centred.csv")],
        "programme usage, amplitude and activity; case-control contrasts per domain")

    add("enrich", "local", "storm", lambda c: [py("FeatureSelection/FactorAnnotation", "enrich_programmes.py", "--run", c.fit,
                                                  "--out", c.r("program_activity"),
                                                  "--cache", os.path.join(REPO, "reference", "enrichr"))],
        lambda c: [c.r("program_activity", "program_enrichment.json")],
        "curated over-representation of each programme's top genes")

    add("transport", "slurm", "storm", lambda c: [
        py("Integration/OptimalTransport", "transport_tables.py", "--run", c.fit, "--clusters", c.clusters(),
           "--out", os.path.join(c.fit, "transport_chain")),
        py("Integration/OptimalTransport", "couple_pairs.py", "--run", c.fit, "--pairs", "cross",
           "--out", os.path.join(c.fit, "couplings_posthoc")),
        py("Integration/OptimalTransport", "transport_tables.py", "--run", c.fit, "--clusters", c.clusters(),
           "--couplings", os.path.join(c.fit, "couplings_posthoc"),
           "--out", os.path.join(c.fit, "transport_posthoc"))],
        lambda c: [os.path.join(c.fit, "transport_posthoc", "domain_transport_summary.csv")],
        "optimal-transport tables: the chain's couplings and every case x control pair")

    add("packets", "local", "storm", lambda c: [py("agents/Panel1", "build_packets.py", "--run", c.fit,
        "--res", c.res(), "--composition", c.comp(), "--progact", c.r("program_activity"),
        "--out", c.r("agents_packets"), "--card", c.card_path,
        "--min-bins", c.m["domains"]["min_bins_per_domain_packet"]),
        py("Integration/STORM", "pool_partition.py", "pairs", "--run", c.fit, "--res", c.res(),
           "--cells", c.r("celltypes_qc", "cells.csv.gz"), "--packets", c.r("agents_packets"),
           "--progact", c.r("program_activity"), "--card", c.card_path,
           "--min-share", c.m["domains"]["pool_min_boundary_share"])],
        lambda c: [c.r("agents_packets", "index.json"), c.r("agents_packets", "pools", "index.json")],
        "deterministic evidence packets, one per domain and one per adjacent pair pooled, plus the cohort")

    add("notes", "local", "agents", lambda c: [py("agents/common", "render_notes.py", "--card", c.card_path,
                                                  "--methods", c.methods_path, "--out", c.notes)],
        lambda c: [c.notes], "dataset notes rendered from the card (no findings, no statistics)")

    add("domain_panel", "paid", "agents", lambda c: paid(
        c, "agents/Panel1", "run_agents.py", c.r("agents_out", "domain_panel"),
        ["--packets", c.r("agents_packets"), "--dataset", c.notes, "--model", c.agents()["model"],
         "--effort", c.agents()["effort"], "--max-tokens", c.agents()["max_tokens"],
         "--web-search", *c.agents()["domain_web_search"],
         "--max-searches", c.agents()["domain_max_searches"]]),
        lambda c: [c.r("agents_out", "domain_panel", "synthesis.json")],
        "five specialists, critic, synthesis: which domains and strata to test (PAID)")

    add("select_domains", "local", "agents", lambda c: [py("agents/Panel1", "select_domains.py",
        "--synthesis", c.r("agents_out", "domain_panel", "synthesis.json"), "--card", c.card_path,
        "--packets", c.r("agents_packets"), "--out", c.r("decisions"))],
        lambda c: [c.r("decisions", "contrasts.json")],
        "the panel's shortlist and strata, carried forward mechanically")
    gated = True

    add("final_partition", "local", "storm", lambda c: [py("Integration/STORM", "pool_partition.py", "final",
        "--run", c.fit, "--res", c.res(), "--cells", c.r("celltypes_qc", "cells.csv.gz"),
        "--progact", c.r("program_activity"), "--card", c.card_path,
        "--partition", c.r("decisions", "partition.json"), "--out-root", c.run)],
        lambda c: [c.r("decisions", "partition_final.json")],
        "adopted pools replace their members: composition and programme tables on the final partition")

    add("deg", "slurm", "deg", lambda c: [py("FeatureSelection/Nebula", "deg_run.py",
        "--contrasts", c.r("decisions", "contrasts.json"), "--run", c.fit, "--clusters", c.fclusters(),
        "--sections", *c.secs, "--case", *c.case, "--min-frac", c.m["deg"]["min_frac"],
        "--ncore", c.m["deg"]["ncore"], "--out", c.r("deg_out"),
        *(["--depth-covariate"] if c.m["deg"].get("depth_covariate") else []))],
        lambda c: [c.r("deg_out", "summary.csv")],
        "NEBULA mixed model per contrast, donor random intercept")

    sh = lambda c, *p: c.r("sheaf", *p)
    add("sheaf_ppi", "local", "storm", lambda c: [py("Utils", "fetch_ppi.py",
        "--genes", os.path.join(c.fit, "genes.csv"), "--min-score", c.m["sheaf"]["string_score"],
        "--out", sh(c, "string400.csv"))],
        lambda c: [sh(c, "string400.csv")], "STRING edges over the fit's genes (cached by gene set)")

    add("sheaf_charge", "local", "deg", lambda c: [py("FeatureSelection/Topology", "build_charge.py", "--run", c.fit,
        "--progact", c.fprogact(), "--agents", c.r("agents_out", "domain_panel"),
        "--partition", c.r("decisions", "partition.json"),
        "--domains", *c.shortlist(), "--top", c.m["sheaf"]["top_nodes"], "--out", sh(c, "charges"))],
        lambda c: [sh(c, "charges", "provenance.json")],
        "charge = loading x upheld programme shift, per shortlisted domain")

    def sheaf_steps(steps):
        return lambda c: steps(c) if c.charged() else [fn(lambda c: print("  no charged domain: skipped"))]

    add("sheaf_expr", "local", "deg", sheaf_steps(lambda c: [py("FeatureSelection/Topology", "build_expr.py",
        "--charges", sh(c, "charges"), "--indir", c.r("cellbin"),
        "--cells", os.path.join(c.fcomp(), "cells_with_domain.csv.gz"), "--domains", *c.charged(),
        "--out", sh(c, "expr"))]),
        lambda c: [], "donor-centred co-expression per charged domain")

    add("sheaf_rank", "slurm", "deg", sheaf_steps(lambda c: [py("FeatureSelection/Topology", "run_sheaf.py",
        "--charges", sh(c, "charges"), "--expr", sh(c, "expr"), "--ppi", sh(c, "string400.csv"),
        "--domains", *c.charged(), "--n-scales", c.m["sheaf"]["n_scales"],
        "--workers", c.m["sheaf"]["workers"], "--weight-transform", c.m["sheaf"]["weight_transform"],
        "--out", sh(c, "out_rank"))]),
        lambda c: [], "leave-one-gene-out sheaf spectra (BLAS pinned)")

    add("sheaf_residual", "local", "deg", sheaf_steps(lambda c: [
        py("FeatureSelection/Topology", "residual_rank.py", "--charges", sh(c, "charges"), "--expr", sh(c, "expr"),
           "--out", sh(c, "out_rank"), "--ppi", sh(c, "string400.csv"), "--domains", *c.charged(),
           "--weight-transform", c.m["sheaf"]["weight_transform"], "--n-scales", c.m["sheaf"]["n_scales"],
           "--topn", c.m["sheaf"]["topn"]),
        py("FeatureSelection/Topology", "annotate_sheaf.py", "--charges", sh(c, "charges"), "--expr", sh(c, "expr"),
           "--out", sh(c, "out_rank"), "--enrich", os.path.join(c.fprogact(), "program_enrichment.csv"),
           "--gmt", os.path.join(REPO, "reference", "enrichr"), "--ppi", sh(c, "string400.csv"),
           "--domains", *c.charged(), "--weight-transform", c.m["sheaf"]["weight_transform"])]),
        lambda c: [], "degree residual per scale, intersected; graph position and pathway membership per gene")

    tp_args = lambda c: ["--run", c.fit, "--deg", c.r("deg_out"), "--sheaf", c.r("sheaf"),
                         "--progact", c.fprogact(), "--composition", c.fcomp(),
                         "--domains", *c.shortlist(), "--top-charge", c.m["targets"]["top_charge"],
                         "--out", c.r("target_packets")]
    add("target_packets", "local", "agents", lambda c: [
        py("agents/Panel2", "build_target_packets.py", *tp_args(c)),
        py("agents/Panel2", "resolve_identity.py", "--packets", c.r("target_packets"), "--out", c.r("identity.tsv")),
        py("agents/Panel2", "build_target_packets.py", *tp_args(c), "--identity", c.r("identity.tsv"))],
        lambda c: [c.r("target_packets", "index.json"), c.r("identity.tsv")],
        "three axes per candidate gene; UniProt identity by exact symbol")

    add("target_panel", "paid", "agents", lambda c: paid(
        c, "agents/Panel2", "run_target_agents.py", c.r("agents_out", "target_panel"),
        ["--packets", c.r("target_packets"), "--dataset", c.notes, "--model", c.agents()["model"],
         "--effort", c.agents()["effort"], "--max-tokens", c.agents()["max_tokens"],
         "--batch", c.agents()["target_batch"], "--direction-batch", c.agents()["direction_batch"],
         "--web-search", c.agents()["target_web_search"]]),
        lambda c: [c.r("agents_out", "target_panel", "synthesis.json")],
        "axis specialists, direction (literature), critic, synthesis (PAID)")

    add("select_targets", "local", "agents", lambda c: [py("agents/Panel2", "select_targets.py",
        "--synthesis", c.r("agents_out", "target_panel", "synthesis.json"),
        "--critic", c.r("agents_out", "target_panel", "critic.json"),
        "--tiers", *c.m["targets"]["tiers_admitted"], "--out", c.r("decisions", "targets.json"))],
        lambda c: [c.r("decisions", "targets.json")],
        "tiers recomputed from DATA lines only; literature never sets eligibility, direction gates nothing")

    gated = False
    add("report", "paid", "agents", lambda c: [
        fn(write_manifest),
        py("Report", "assemble_report.py", "--manifest", c.r("report", "manifest.json"))] + paid(
        c, "Report", "run_report.py", c.r("report"),
        ["--data", c.r("report", "report_data.json"), "--tables", c.r("report", "report_tables.md"),
         "--dataset", c.notes, "--model", c.agents()["model"], "--effort", c.agents()["effort"],
         "--max-tokens", c.agents()["max_tokens"]]),
        lambda c: [c.r("report", "report.md")],
        "results assembled from every stage; the writer agent drafts the report (PAID)")

    # Plots are drawn from the run's own tables, so they follow a refit without
    # being redrawn. `deg` is the env because the network plot imports the
    # sheaf's own graph builder, which needs gudhi.
    add("figures", "local", "deg", lambda c: [fn(write_manifest), py("Report/figures", "make_figures.py",
        "--run", c.r(), "--out", c.r("report", "figures"))],
        lambda c: [],
        "plots of the run's outputs: QC, spatial domains, factor matrices, DEG, PPI networks")
    return S


# --------------------------------------------------------------------------- execution
def argv_for(c: Ctx, env_key, step):
    _, cwd, script, args = step
    return os.path.join(REPO, cwd), [c.conda, "run", "--no-capture-output", "-n", c.m["envs"][env_key],
                                     "python", "-u", script, *args]


def run_local(c: Ctx, st, steps, log):
    for step in steps:
        if step[0] == "fn":
            print(f"  [fn] {step[1].__name__}", flush=True)
            step[1](c)
            continue
        cwd, argv = argv_for(c, st["env"], step)
        print(f"  $ (cd {os.path.relpath(cwd, REPO)}) {' '.join(shlex.quote(x) for x in argv[5:])}", flush=True)
        with open(log, "a") as fh:
            fh.write(f"\n### {now()} {' '.join(argv)}\n")
            p = subprocess.Popen(argv, cwd=cwd, env=c.env(), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                sys.stdout.write(line); fh.write(line)
            rc = p.wait()
        if rc != 0:
            raise SystemExit(f"[{st['name']}] failed (exit {rc}); log {log}")


def job_script(c: Ctx, st, steps):
    res = c.m["slurm"][st["name"]]
    pin = st["name"] == "sheaf_rank"          # process pool: one BLAS thread per worker
    threads = 1 if pin else res["cpus"]
    lines = ["#!/bin/bash --login",
             f"#SBATCH --job-name={c.dataset}_{c.run_id}_{st['name']}",
             f"#SBATCH --time={res['time']}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
             f"#SBATCH --cpus-per-task={res['cpus']}", f"#SBATCH --mem={res['mem']}",
             f"#SBATCH --output={c.r('jobs', st['name'] + '.out')}"]
    if res.get("gpu"):
        lines.append(f"#SBATCH --gpus={res['gpu']}")
    lines += ["", "set -euo pipefail", "module purge || true", "unset PYTHONPATH"]
    for k, v in c.env().items():
        if k.startswith("WALKTHROUGH_"):
            lines.append(f"export {k}={shlex.quote(v)}")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        lines.append(f"export {k}={threads}")
    lines.append('echo "host=$(hostname) start=$(date -Is) job=$SLURM_JOB_ID"')
    for step in steps:
        if step[0] == "fn":
            raise SystemExit(f"[{st['name']}] a SLURM stage cannot run an in-process step")
        cwd, argv = argv_for(c, st["env"], step)
        lines.append(f"cd {shlex.quote(cwd)} && " + " ".join(shlex.quote(x) for x in argv))
    lines.append('echo "end=$(date -Is)"')
    return "\n".join(lines) + "\n"


def wait_for(job, poll=60):
    """Block until the job leaves the queue; then its final state from sacct,
    which can lag the queue by a few seconds."""
    while True:
        r = subprocess.run([SQUEUE, "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
        if not r.stdout.strip():
            break
        time.sleep(poll)
    state = "UNKNOWN|"
    for _ in range(12):
        r = subprocess.run([SACCT, "-n", "-P", "-X", "-j", job, "-o", "State,ExitCode"],
                           capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        state = lines[0] if lines else state
        if not state.split("|")[0] in ("", "UNKNOWN", "PENDING", "RUNNING", "COMPLETING"):
            break
        time.sleep(10)
    return state.startswith("COMPLETED"), state


def submit_wait(c: Ctx, st, sb):
    """sbatch --wait, remembering the job id so a restarted driver waits on a job
    still running instead of submitting a duplicate."""
    pending = c.r(".state", st["name"] + ".job")
    if os.path.isfile(pending):
        job = open(pending).read().strip()
        r = subprocess.run([SQUEUE, "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
        if r.stdout.strip():
            print(f"  job {job} from an earlier driver is still {r.stdout.strip()}; waiting", flush=True)
            ok, state = wait_for(job)
            os.remove(pending)
            if not ok:
                raise SystemExit(f"[{st['name']}] job {job} ended {state}")
            return
        # it finished while no driver was watching: reuse it if it completed
        ok, state = wait_for(job)
        os.remove(pending)
        if ok:
            print(f"  job {job} from an earlier driver already completed; using its outputs", flush=True)
            return
        print(f"  job {job} from an earlier driver ended {state}; resubmitting", flush=True)
    # Submit WITHOUT --wait and poll: sbatch buffers its stdout on a pipe, so under
    # --wait the job id only arrives when the job ends -- too late to record it
    # for a restarted driver. The site prints notices around the id; the id is
    # the first line that is only digits (--parsable may append ";cluster").
    r = subprocess.run([SBATCH, "--parsable", sb], capture_output=True, text=True)
    ids = re.findall(r"^(\d+)(?:;\S+)?\s*$", r.stdout, re.M)
    if r.returncode != 0 or not ids:
        raise SystemExit(f"[{st['name']}] sbatch failed: {r.stderr.strip()} {r.stdout.strip()}")
    job = ids[0]
    open(pending, "w").write(job)
    print(f"  submitted job {job} ({os.path.relpath(sb, c.run)}); waiting", flush=True)
    ok, state = wait_for(job)
    os.remove(pending)
    if not ok:
        raise SystemExit(f"[{st['name']}] job {job} ended {state}; "
                         f"log {c.r('jobs', st['name'] + '.out')}")


def run_stage(c: Ctx, st):
    steps = st["build"](c)
    log = c.r("logs", st["name"] + ".log")
    if st["kind"] in ("local", "paid"):
        return run_local(c, st, steps, log)
    if st["kind"] == "slurm":
        if not any(s[0] == "py" for s in steps):          # a skipped stage
            return run_local(c, st, steps, log)
        sb = c.r("jobs", st["name"] + ".sb")
        text = job_script(c, st, steps)
        print(f"  job script {sb}", flush=True)
        open(sb, "w").write(text)
        return submit_wait(c, st, sb)
    if st["kind"] == "storm":                             # write_storm_job.py writes the job script
        run_local(c, st, steps, log)
        text = open(log).read()
        sbs = re.findall(r"wrote (\S+\.sb)", text)
        if not sbs:
            raise SystemExit("[storm] write_storm_job.py did not report a job script")
        return submit_wait(c, st, sbs[-1])


def done_ok(c, st):
    return os.path.isfile(c.r(".state", st["name"] + ".done")) and \
        all(os.path.exists(p) for p in st["done"](c))


def submit_driver(c: Ctx, a):
    res = c.m["slurm"]["driver"]
    flags = []
    for k in ("start", "until"):
        if getattr(a, k):
            flags += [f"--{'from' if k == 'start' else k}", getattr(a, k)]
    if a.only:
        flags += ["--only", *a.only]
    if a.force:
        flags.append("--force")
    if a.fresh_agents:
        flags.append("--fresh-agents")
    if a.allow_paid:
        flags.append("--allow-paid")
    cmd = [c.conda, "run", "--no-capture-output", "-n", c.m["envs"]["storm"], "python", "-u",
           os.path.join(REPO, "run_all.py"), "--card", c.card_path, "--methods", c.methods_path,
           "--run", c.run_id, *flags]
    sb = c.r("jobs", "driver.sb")
    text = "\n".join([
        "#!/bin/bash --login", f"#SBATCH --job-name={c.dataset}_{c.run_id}_driver",
        f"#SBATCH --time={res['time']}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={res['cpus']}", f"#SBATCH --mem={res['mem']}",
        f"#SBATCH --output={c.r('jobs', 'driver.out')}", "", "set -uo pipefail", "module purge || true",
        "unset PYTHONPATH", f"export OMP_NUM_THREADS={res['cpus']}",
        'echo "host=$(hostname) start=$(date -Is) job=$SLURM_JOB_ID"',
        f"cd {shlex.quote(REPO)}", " ".join(shlex.quote(x) for x in cmd), 'echo "end=$(date -Is)"', ""])
    os.makedirs(c.r("jobs"), exist_ok=True)
    open(sb, "w").write(text)
    r = subprocess.run([SBATCH, "--parsable", sb], capture_output=True, text=True)
    ids = re.findall(r"^(\d+)", r.stdout, re.M)
    if r.returncode != 0 or not ids:
        raise SystemExit(f"sbatch failed: {r.stderr.strip()} {r.stdout.strip()}")
    job = ids[0]
    print(f"driver submitted: job {job}\n  log  {c.r('jobs', 'driver.out')}\n  run  {c.run}")


# --------------------------------------------------------------------------- the graph
class RunState(TypedDict, total=False):
    """What travels along the graph: how the run went, nothing else."""
    ran: Annotated[list, operator.add]        # stages this invocation executed
    skipped: Annotated[list, operator.add]    # stages that were already done
    stopped: str                              # the paid stage the run stopped before


class StageFailed(Exception):
    """A stage failed; the message says which and where its log is."""


def stage_node(c: Ctx, a, st, selected, forced, rec, save):
    """One stage as a graph node: skip it if it is not selected or already done,
    stop before it if it is paid and paying was not allowed, otherwise run it,
    check its outputs and write its done marker."""
    name = st["name"]

    def node(state: RunState) -> dict:
        if name not in selected:
            return {}
        t0 = time.time()
        try:
            if name not in forced and done_ok(c, st):
                print(f"[{name}] done -- skipping", flush=True)
                return {"skipped": [name]}
            if st["kind"] == "paid" and not a.allow_paid:
                print(f"\n[{name}] PAID stage (Anthropic API). Not run without --allow-paid; "
                      f"stopping here. Every earlier stage is done.", flush=True)
                return {"stopped": name}
            print(f"\n[{name}] {st['note']}", flush=True)
            run_stage(c, st)
            missing = [x for x in st["done"](c) if not os.path.exists(x)]
            if missing:
                raise SystemExit(f"[{name}] finished but outputs are missing: {missing}")
        except SystemExit as e:
            rec["stages"].append(dict(stage=name, status="FAILED", error=str(e),
                                      seconds=round(time.time() - t0)))
            save()
            raise StageFailed(str(e)) from None
        json.dump(dict(stage=name, finished=now(), seconds=round(time.time() - t0)),
                  open(c.r(".state", name + ".done"), "w"))
        rec["stages"].append(dict(stage=name, status="ok", seconds=round(time.time() - t0)))
        save()
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
        return {"ran": [name]}

    return node


def build_graph(c: Ctx, a, stages, selected, forced, rec, save):
    """The pipeline as a LangGraph: stages in order, plus

      select_domains -> the first stage that needs no shortlist (the report)
                        when the domain panel shortlisted nothing
      a paid stage   -> END when it stopped the run (no --allow-paid)
    """
    names = [s["name"] for s in stages]
    g = StateGraph(RunState)
    for st in stages:
        g.add_node(st["name"], stage_node(c, a, st, selected, forced, rec, save))
    g.add_edge(START, names[0])

    gated = [s["name"] for s in stages if s["needs_shortlist"]]
    after_gated = names[names.index(gated[-1]) + 1]          # where a run with no shortlist resumes

    def shortlist_route(state: RunState) -> str:
        if c.no_shortlist():
            print(f"\nnothing shortlisted: {gated[0]} .. {gated[-1]} skipped", flush=True)
            return "nothing shortlisted"
        return "shortlist"

    def paid_route(state: RunState) -> str:
        return "stopped" if state.get("stopped") else "continue"

    for st, nxt in zip(stages, names[1:] + [END]):
        name = st["name"]
        if nxt in gated[:1]:                                  # the stage that makes the shortlist
            g.add_conditional_edges(name, shortlist_route,
                                    {"shortlist": nxt, "nothing shortlisted": after_gated})
        elif st["kind"] == "paid":
            g.add_conditional_edges(name, paid_route, {"continue": nxt, "stopped": END})
        else:
            g.add_edge(name, nxt)
    return g.compile()


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--card", required=True)
    p.add_argument("--methods", default=METHODS)
    p.add_argument("--run", required=True, help="run id: outputs go to runs/<dataset>/<run>/")
    p.add_argument("--list", action="store_true")
    p.add_argument("--graph", action="store_true", help="print the stage graph (Mermaid) and exit")
    p.add_argument("--submit", action="store_true", help="run this runner as a SLURM driver job")
    p.add_argument("--from", dest="start", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--force", action="store_true", help="re-run selected stages even if done")
    p.add_argument("--fresh-agents", action="store_true",
                   help="agent stages do not reuse finished calls (pays again)")
    p.add_argument("--allow-paid", action="store_true",
                   help="let agent stages call the API (costs money). Without it the run "
                        "stops before the first paid stage")
    ARGS = a = p.parse_args()
    c = Ctx(a.card, a.methods, a.run)
    stages = build_stages()
    names = [s["name"] for s in stages]
    for k in ("start", "until"):
        if getattr(a, k) and getattr(a, k) not in names:
            raise SystemExit(f"--{k} {getattr(a, k)}: one of {', '.join(names)}")
    if a.only and set(a.only) - set(names):
        raise SystemExit(f"--only {sorted(set(a.only) - set(names))}: one of {', '.join(names)}")

    if a.list:
        print(f"{c.dataset} / {c.run_id}  ->  {c.run}\n")
        for i, st in enumerate(stages, 1):
            mark = "done" if done_ok(c, st) else "    "
            print(f"  {mark} {i:2d}. {st['name']:19s} {st['kind']:6s} {st['note']}")
        return
    if a.graph:
        print(build_graph(c, a, stages, set(), set(), {}, lambda: None).get_graph().draw_mermaid())
        return
    if a.submit:
        return submit_driver(c, a)

    for d in ("logs", "jobs", ".state", "decisions"):
        os.makedirs(c.r(d), exist_ok=True)
    order = names[names.index(a.start):] if a.start else list(names)
    if a.until:
        order = order[:order.index(a.until) + 1] if a.until in order else order
    if a.only:
        order = [n for n in order if n in a.only]
    forced = set(order) if (a.force or a.start or a.only) else set()
    prov_path = c.r("provenance.json")
    prov = json.load(open(prov_path)) if os.path.isfile(prov_path) else dict(runs=[])
    rec = dict(started=now(), card=c.card_path, methods=c.methods_path, argv=sys.argv[1:], stages=[])
    prov["runs"].append(rec)

    def save():
        json.dump(prov, open(prov_path, "w"), indent=1)

    graph = build_graph(c, a, stages, set(order), forced, rec, save)
    try:
        graph.invoke({"ran": [], "skipped": []}, config={"recursion_limit": len(stages) + 10})
    except StageFailed as e:
        raise SystemExit(str(e))
    rec["finished"] = now()
    save()
    if os.path.isfile(c.r("report", "report.md")):
        print(f"\nreport: {c.r('report', 'report.md')}")


ARGS = None

if __name__ == "__main__":
    main()
