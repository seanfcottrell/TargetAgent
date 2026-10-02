# Topological importance: the degree residual of a charged sheaf Laplacian

One of the three axes of evidence behind target selection (with differential
expression and programme loading). The question it asks: within the genes a
domain's moving programmes implicate, which are structurally central to the
module rather than merely strongly loaded.

The unit is **one programme in one domain**: a domain in which the domain panel
upheld two programmes gets two networks and two residual sets, never one network
over their sum.

## Method

1. **Charge.** For a programme `r` the panel upheld in domain `d`,
   `charge[g] = B[g, r] x shift[d, r]`, where `B` is the STORM gene loading and
   the shift is the case-control Cohen's d of the programme's centred usage in
   that domain. The nodes of the network are the top genes by |charge|
   (`sheaf.top_nodes` in `config/methods.yaml`), i.e. that programme's top
   loaders. Which programmes enter is derived mechanically from the domain
   panel's `critic.json`: movement citations the critic upheld as independent.
2. **Edges.** STRING interactions above `sheaf.string_score` give which edges
   exist; within-donor co-expression over the domain's nuclei gives their
   weight, `1 - |rho|`. Expression is CP10K on each nucleus's full library,
   log1p, then centred within each section, so Pearson on the pooled matrix is
   the within-donor correlation. The weights are rank-transformed
   (`--weight-transform rank`): per-nucleus co-expression is small, so raw
   `1 - |rho|` occupies a narrow band and the filtration would carry no scale
   information.
3. **Spectral shift.** For each gene, the leave-one-gene-out shift of the sheaf
   Laplacian spectra (L0 and L1) across `sheaf.n_scales` filtration scales
   (`PSL.py`, `SignificanceRankingPipeline.py`).
4. **Degree residual.** A leave-one-out spectral shift is dominated by the
   removed node's incident edge mass, so ranking by the shift alone largely
   reproduces weighted degree. The statistic is therefore the residual: at each
   scale the shift is regressed on that scale's weighted degree, the residuals
   are ranked, and the per-scale top-`sheaf.topn` lists are intersected. A gene
   absent from a scale is excluded from it rather than scored as zero.

## Files

| file | role |
|---|---|
| `../../Utils/fetch_ppi.py` | STRING edges over the fit's genes -> `string400.csv` (gene1, gene2, combined_score). Cached under a hash of the gene set, so a different gene set cannot silently reuse it. |
| `units.py` | The unit is one programme in one domain. `tag(d, F)` -> `domain<d>_F<r>`, `networks(charges_dir)` lists the (domain, programme) networks from `provenance.json`, `load_expr` slices one network's rows out of the domain matrix. |
| `build_charge.py` | Programme subset per domain from the domain panel's `critic.json`. Writes the combined charge (`domain<d>_charge.csv`, the sum over upheld programmes with per-programme contributions: the programme axis's quantity) and one charge table per programme (`domain<d>_F<r>_charge.csv`), which is what the sheaf runs on. Every accepted and rejected citation is logged in `charges/provenance.json`. |
| `build_expr.py` | Per-domain nucleus matrices over the union of the domain's networks' node genes, normalised and centred within each section; plus one ordered node list per network (`domain<d>_F<r>_genes.csv`). |
| `run_sheaf.py` | Per programme network: leave-one-gene-out spectral shift across the filtration scales. |
| `annotate_sheaf.py` | Per network: each gene's position in the graph (degree, weighted degree, betweenness, clustering) next to its spectral rank, and its membership of the programme's enriched pathways. This is what the target packets report on the topology axis. |
| `residual_rank.py` | **The axis output.** Per network and per scale, regress the shift on weighted degree, rank residuals, intersect the per-scale top lists -> `out_rank/domain<d>_F<r>_residual_per_scale.csv`, `out_rank/residual_per_scale.json` = `{domain: {programme: {...}}}`. |
| `PSL.py`, `SignificanceRankingPipeline.py` | The persistent sheaf Laplacian and the graph builder / ranking pipeline the scripts above call. |

`run_all.py` runs these as the stages `sheaf_ppi`, `sheaf_charge`, `sheaf_expr`,
`sheaf_rank` and `sheaf_residual`, with the parameters in the `sheaf` block of
`config/methods.yaml`.

## Things to get right

* **`--weight-transform` must match** across `run_sheaf.py`, `annotate_sheaf.py`
  and `residual_rank.py`, or the centralities describe a different graph than
  the spectral shift does.
* **Pin BLAS threads** (`OMP_NUM_THREADS=1` and friends, as `run_all.py`'s
  `sheaf_rank` job does). Unpinned, every pool worker spawns its own eigensolver
  threads and the job oversubscribes its cores.
* **Environment is `deg-calib`**, the one carrying gudhi.
