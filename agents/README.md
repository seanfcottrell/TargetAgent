# Agent panels

Two panels, one per stage, plus what they share:

| dir | panel | driver |
|---|---|---|
| `common/` | shared: `llm.py` (THE model-call seam, on LangChain's `ChatAnthropic`; usage ledger, .env), `citations.py` (machine verification), `render_notes.py`, `schemas_common.py`, `prompts/_common_target.md`, `.env` | -- |
| `Panel1/` | domain panel: which spatial domains and strata to test | `run_agents.py` |
| `Panel2/` | target panel: three axes, direction, critic, synthesis; then `select_targets.py` recomputes tiers from DATA lines only (literature grounds claims and sets direction, never eligibility) | `run_target_agents.py` |

The report writer is a third role and lives in `../Report/`.

## Panel1: multi-agent selection of disease-relevant spatial domains

Five specialist agents, each reading ONE mathematical perspective of the STORM
analysis, score every spatial domain on relevance (is disease acting here) and
suitability (can a case-control contrast be estimated cleanly); a critic labels every
cited evidence item independent / redundant / explained-away; a synthesis
applies a stated rule and ranks. The output is a shortlist of domains and
cell-type strata, with pre-registered falsifiable predictions, for the DEG step.

| file | role |
|---|---|
| `Panel1/build_packets.py` | deterministic: final tables -> `packets/cohort.json` (samples, design, confounds, programme definitions, semantics and curated enrichment) + `packets/domain_<d>.json` (identity, spatial, composition, factors, transport, design blocks; every number with its source field). No external reference partition and no permutation null are used anywhere: each metric carries a reference that is interpretable on its own - the share of tiles each section would contribute if the domain were spread evenly, the stage fractions expected if uniform, and the within-stage versus cross-stage transport baselines measured in the cohort. Cohort facts (stage counts, depth and QC ranges, shared chip lots) are derived from the fit, never hardcoded. |
| `../FeatureSelection/FactorAnnotation/enrich_programmes.py` | hypergeometric over-representation of each programme's top-50 genes against GO BP, Reactome, KEGG, PanglaoDB cell types and DisGeNET (background = the fit's HVGs; libraries cached in `reference/enrichr/`) -> `program_activity/program_enrichment.{csv,json}`; rerun before `build_packets.py` after any refit |
| `Panel1/schemas.py` (+ `common/schemas_common.py`: `Strict`, `Citation`) | pydantic contracts: `AgentReport`, `CriticReport`, `SynthesisReport` (structured JSON output) |
| `Panel1/prompts/_common.md` | GENERIC system prompt shared by every agent: the panel's task, the semantics of the mathematical objects (domains, usage/amplitude/activity, composition, transport, nulls), and the output contract. No dataset facts. |
| `Panel1/prompts/{spatial,composition,factor,transport,design,critic,synthesis}.md` | GENERIC per-role system prompts: what that perspective's mathematics licenses it to say, its mandatory confound checks, how it scores. No dataset facts. |
| `common/render_notes.py` | the dataset notes every agent gets, RENDERED from the data card: description, cohort, how the objects were made, prior cell types and domain count, policies. No findings, no statistics. |
| `Panel1/select_domains.py` | the domain panel's synthesis -> `decisions/domains.json` + `decisions/contrasts.json`, mechanically (shortlisted domains; their named strata that are prior cell types; the designed contrasts the synthesis adopted, each checked against `contrast_proposals.json` and the critic's `contrast_verdicts`) |
| `common/citations.py` | machine verification of every literature citation: the driver logs each web search (query, result URLs and titles); a citation must match a result of the run's searches, and a PMID must resolve in PubMed to the same paper. Statuses verified / verified_session (support) vs exists_not_searched / untraceable / pmid_mismatch / pmid_not_found. The critics receive the table as `citation_verification`. |
| `../Report/run_report.py` + `../Report/prompts/report.md` + `../Report/schemas_report.py` | the report writer: `report/report_data.json` -> `report/report.md`, every number cited to its field, no web search |
| `Panel1/run_agents.py` | driver; every call goes through `common/llm.py` `call()`: `claude-opus-5`, adaptive thinking, `output_config.effort`, structured output via `output_config.format`, cached preamble, server-side refusal fallbacks; the factor agent gets the server-side `web_search` tool (`--web-search`, capped by `--max-searches`) to check its mechanism hypotheses against the literature; saves every request, raw response, parsed JSON and token usage under `--out` |

Prompt layout: system = generic role + semantics + contract (cached, reusable across datasets); user = instructions + dataset notes + cohort + null reference + the unit packets. Evidence isolation: each specialist receives only its own packet blocks plus `identity` (size only) and `design` and the cohort; the critic receives everything plus all
five reports; the synthesis receives the critic's verdicts and the reports.

Rule (synthesis): shortlist = >= 2 independent surviving lines of relevance
from different perspectives AND suitability passes AND no unresolved decisive
confound; hold = relevance evidence but suitability fails, or exactly one line;
exclude otherwise.

Run:
```
export ANTHROPIC_API_KEY=...            # or put it in agents/common/.env
python run_all.py --card datasets/<name>/card.yaml --run <run id> --only packets notes      # packets and notes
python run_all.py --card datasets/<name>/card.yaml --run <run id> --only domain_panel --allow-paid
```
`usage.csv` in the output directory records the spend of every call. Outputs:
`assessments.csv` (every specialist score), `critic.json`, `ranking.csv`, `report.md`.

To review the panel's decision before any DEG is run, stop the runner after it
with `--until select_domains` and read `report.md`: its predictions are the
hypotheses the DEG results are read against.
