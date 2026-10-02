# Agentic target identification from spatial transcriptomics

An end-to-end pipeline that goes from Stereo-seq spatial transcriptomics of
diseased and control tissue to a ranked, evidence-audited list of candidate
therapeutic targets. Mathematical modelling produces spatial domains, gene
programmes and three independent axes of evidence per gene; panels of LLM agents
make the judgements between stages (which domains to test, which targets the
evidence supports) under mechanical rules, and every decision that passes from
one stage to the next is a file. The runner is a LangGraph state graph, and the
agents call the model through LangChain.

The repository is set up for a Stereo-seq dataset of human dorsolateral
prefrontal cortex in Alzheimer's disease (GEO `GSE269906`), described by the
data card `datasets/ad_pfc_stereoseq/card.yaml`. A different dataset is a new
data card.

## What the pipeline does

1. **Preparation.** Per-section nuclei (cellbin) and square tiles (bin110) are
   built from the GEF files; tiles are restricted to the genes all sections
   share and annotated with the card's metadata.
2. **Cell typing.** Nuclei are QC-filtered, integrated across sections (Harmony)
   and clustered (Leiden); three independent callers (CellTypist, an Allen
   reference correlation, the card's marker sets) vote each cluster onto the
   card's cell types.
3. **STORM.** A graph-regularised PARAFAC2 factorisation over all sections,
   `X_k ~ Q_k H D_k B^T`: `B` holds shared nonnegative gene programmes, `Q_k H`
   each tile's usage of every programme, and `D_k` each section's amplitude. The
   within-section graph is an expression kNN united with the tile lattice;
   sections are linked by optimal-transport couplings along the links the card
   gives; genes are linked by a STRING PPI graph.
4. **Spatial domains.** Leiden clustering of programme usage, with the
   resolution searched to return the number of domains the card specifies.
5. **Programmes.** Usage, amplitude and activity per domain and section, their
   case-control contrasts, curated over-representation of each programme's top
   genes, and optimal-transport tables between case and control sections.
6. **Domain panel (agents).** Five specialists each read one mathematical
   perspective of every domain (spatial, composition, factor, transport,
   design); a critic audits every cited item; a synthesis applies a stated rule.
   The output is the shortlist of domains and cell-type strata to test, plus any
   designed contrasts, carried forward mechanically.
7. **Differential expression.** NEBULA negative-binomial mixed models with a
   donor random intercept, per shortlisted domain and stratum.
8. **Network topology.** For each programme the panel upheld in a domain, a PPI
   network over that programme's top-loading genes, weighted by within-donor
   co-expression; genes are ranked by the degree residual of the leave-one-out
   sheaf Laplacian spectral shift (`FeatureSelection/Topology/README.md`).
9. **Target panel (agents).** Axis specialists read the three axes
   (dysregulation, programme, topology) in isolation; a direction agent grounds
   the direction of intervention in literature it must find by web search; a
   critic and a synthesis rank the targets. `select_targets.py` then recomputes
   every tier from the data axes alone.
10. **Report and plots.** The results of every stage are assembled into tables,
    a writer agent drafts the report from them (each number checked mechanically
    against the tables), and the run's outputs are plotted.

## Repository layout

```
run_all.py                 the end-to-end runner: data card -> report, a LangGraph
                           graph of 25 stages
paths.py                   REPO_ROOT (code) vs RUN_ROOT (one run)
datasets/<name>/card.yaml  the data card: the only description of a dataset the
                           pipeline receives (datasets/README.md)
config/methods.yaml        method defaults, conda env names and SLURM resources
envs/                      conda environment files

Utils/                     GEF conversion (build_cellbin, build_bin110), tile prep
                           (prep_bins), domain composition and plots, PPI graphs
                           (fetch_ppi)
CellBinAnalysis/           nucleus QC, HVG, PCA, Harmony, Leiden (celltype_nuclei);
                           label transfer and marker scoring (annotate_clusters)
Integration/STORM/         the STORM package (STORM/) and its wrappers: write_storm_job,
                           select_domain_resolution, finish_domains, pool_partition
Integration/OptimalTransport/  couplings and domain transport tables
FeatureSelection/Nebula/   NEBULA differential expression (deg_run)
FeatureSelection/Topology/ sheaf charges, co-expression, spectral ranking, degree
                           residual: one network per upheld programme per domain
FeatureSelection/FactorAnnotation/  factor programme activity and curated gene set enrichment of factors
agents/common/             the model-call seam (llm.py, LangChain), citation verification,
                           dataset-notes renderer, shared schema
agents/Panel1/             domain panel: packets, driver, schemas, selection, prompts
agents/Panel2/             target panel: packets, identity, driver, schemas, prompts,
                           select_targets (tiers from data lines only)
Report/                    assemble_report (deterministic tables), run_report (writer
                           agent), check_report, figures/ (plots)
```

## Requirements

- A SLURM cluster with conda. Heavy stages are submitted as jobs; the STORM fit
  and the transport tables use one GPU.
- An Anthropic key for the agent stages (`domain_panel`, `target_panel`,
  `report`). Everything else is free to run.
- Outbound HTTPS from the nodes that run the stages (STRING, Enrichr,
  CellTypist and Allen reference downloads, UniProt, PubMed, the LLM API).

## Environments

Two conda environments, one per group of stages; create both.

```
conda env create -f envs/st-graph-parafac2-gpu.yml
conda env create -f envs/deg-calib.yml
conda run -n deg-calib Rscript -e 'install.packages("nebula", repos="https://cloud.r-project.org")'
```

| env | stages |
|---|---|
| `st-graph-parafac2-gpu` | preparation, cell typing, STORM, domain clustering, programme activity, optimal transport, the agent drivers (`langchain-anthropic`, `pydantic`) and `run_all.py` itself (`langgraph`) |
| `deg-calib` | NEBULA differential expression (R + rpy2), the sheaf / topology stages (the only env with `gudhi`), and the plots |

## Running

```
C=datasets/ad_pfc_stereoseq/card.yaml
python run_all.py --card $C --run run1 --list      # the stages, and which are done
python run_all.py --card $C --run run1 --submit    # free stages only: stops before the first agent stage
python run_all.py --card $C --run run1 --submit --allow-paid   # everything, agents included
python run_all.py --card $C --run run1 --from deg --submit     # redo from a stage onward
python run_all.py --card $C --run run1 --only figures          # one stage
```

`run_all.py` itself needs `pyyaml` and `langgraph`: run it from
`st-graph-parafac2-gpu`.

`--submit` puts the runner itself in a SLURM job (`<run>/jobs/driver.out`).
Heavy stages become their own jobs, submitted from inside it and polled until
they finish; light stages run in the driver's allocation through
`conda run -n <env>`.

A stage is done when `<run>/.state/<stage>.done` exists and its outputs are
present; re-launching skips done stages and waits on a stage whose SLURM job is
still running rather than resubmitting it. `--from`, `--only` and `--force`
re-run explicitly.

The run is a LangGraph state graph (`build_graph` in `run_all.py`): one node per
stage in the order below, with two branches. After `select_domains` the run goes
straight to `report` if the domain panel shortlisted nothing, and a paid stage
ends the run when `--allow-paid` was not given. The graph state only records
how the run went; results travel as files, and the done markers on disk are what
make a re-launch resume.

### Stages

| # | stage | kind | env | what it does |
|---|---|---|---|---|
| 1 | `inputs` | local | storm | validate the card; link the card's sections; write samples, markers, links |
| 2 | `prep` | slurm | storm | gene intersection over the card's sections, empty-tile floor, metadata |
| 3 | `celltype` | slurm | storm | nucleus QC, Harmony, Leiden; three callers vote onto the card's cell types |
| 4 | `storm_graph` | slurm | storm | STORM graph: per-section kNN and lattice, couplings between sections (CPU) |
| 5 | `storm_fit` | slurm | storm | graph-regularised PARAFAC2 on the saved graph (GPU) |
| 6 | `domains` | slurm | storm | resolution searched for the card's domain count; maps, composition |
| 7 | `progact` | local | storm | programme usage, amplitude and activity; case-control contrasts per domain |
| 8 | `enrich` | local | storm | curated over-representation of each programme's top genes |
| 9 | `transport` | slurm | storm | optimal-transport tables between sections |
| 10 | `packets` | local | storm | deterministic evidence packets, one per domain and per adjacent pair pooled |
| 11 | `notes` | local | agents | dataset notes rendered from the card |
| 12 | `domain_panel` | paid | agents | five specialists, critic, synthesis: which domains and strata to test |
| 13 | `select_domains` | local | agents | the panel's shortlist, strata and designed contrasts, checked mechanically |
| 14 | `final_partition` | local | storm | adopted pools replace their members |
| 15 | `deg` | slurm | deg | NEBULA mixed model per contrast |
| 16 | `sheaf_ppi` | local | storm | STRING edges over the fit's genes |
| 17 | `sheaf_charge` | local | deg | charge = loading x upheld programme shift, per shortlisted domain |
| 18 | `sheaf_expr` | local | deg | donor-centred co-expression per charged domain |
| 19 | `sheaf_rank` | slurm | deg | leave-one-gene-out sheaf spectra |
| 20 | `sheaf_residual` | local | deg | degree residual per scale, intersected; graph position and pathway membership per gene |
| 21 | `target_packets` | local | agents | three axes per candidate gene; UniProt identity |
| 22 | `target_panel` | paid | agents | axis specialists, direction (literature), critic, synthesis |
| 23 | `select_targets` | local | agents | tiers recomputed from data lines only |
| 24 | `report` | paid | agents | tables assembled from every stage; the writer agent drafts the report |
| 25 | `figures` | local | deg | plots of the run's outputs |

### Agent stages

Put the key in `agents/common/.env` (git-ignored; see `.env.example`) or export
`ANTHROPIC_API_KEY`. The model, effort and search budgets are in the `agents`
block of `config/methods.yaml`. Every model call goes through one function,
`call()` in `agents/common/llm.py`, built on LangChain's `ChatAnthropic`; the
drivers themselves are plain functions over JSON packets and files.

- Agent stages spend API credit, so they run only with `--allow-paid`; without
  it the runner stops before the first of them, with every free stage complete.
- What an agent receives is its system prompt (`agents/Panel*/prompts/*.md`), the
  dataset notes rendered from the data card, and its packet. Every request is
  written to `<agent>.request.json` and `<agent>.prompt.md` in the stage's output
  directory before the call; those files are generated, so edit the prompts, not
  the rendered output.
- Drivers are resumable: re-running a stage pays only for calls that had not
  finished (`--fresh-agents` to pay again). Every call's usage is appended to
  `usage.jsonl` in the agent's output directory as it returns.
- Every literature citation an agent makes is machine-verified
  (`agents/common/citations.py`): a web search in the run must have returned it,
  and a PMID must resolve in PubMed to the same paper. The verification table is
  written next to the panel's output and goes into the report.

`agents/README.md` describes the panels, their prompts and their contracts.

## Using another dataset

Write `datasets/<name>/card.yaml` (see `datasets/README.md`), convert the
sections to `<section>_cellbin.h5ad` and `<section>_bin<size>.h5ad`, and run
`run_all.py` with the new card. Method choices live in `config/methods.yaml`
and are shared by every dataset; pass a different file with `--methods` to
change them. The conversion scripts in `Utils/` find each section's GEF by the
chip id the card gives.
