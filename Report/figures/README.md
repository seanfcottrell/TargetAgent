# Plots

`make_figures.py` draws plots of one run's outputs from the files that run
wrote, located through `<run>/report/manifest.json`. Nothing is typed in: every
number, gene name and threshold comes from the run, so the same script works on
any dataset and a rerun produces updated plots.

```
conda run -n deg-calib python make_figures.py --run ../../runs/<dataset>/<run>
conda run -n deg-calib python make_figures.py --run <run> --only domains programmes
conda run -n deg-calib python make_figures.py --run <run> --only usage --programmes F1 F3
```

Output goes to `<run>/report/figures/` (or `--out`) as PNG at 600 dpi and PDF
with embedded Type-42 fonts. Every plot is its own file; a legend for an
encoding shared by sibling plots is written once as its own file (`*_legend`).
`run_all.py` runs this as its last stage (`figures`), which is free and
repeatable. A group whose inputs the run has not produced yet is skipped.

**Environment: `deg-calib`.** The network plot imports `build_gcn` from
`FeatureSelection/Topology`, so the picture is built by the same code as the
statistic, and that module imports gudhi.

## What gets drawn

| group | files | what it shows | read from |
|---|---|---|---|
| `qc` | `qc_genes_per_nucleus`, `qc_transcripts_per_nucleus`, `qc_nuclei_retained` | detected genes and transcripts per QC-passed nucleus by condition (log10); nuclei segmented and retained per section | `celltypes_qc/cells.csv.gz`, `celltypes_qc/composition_by_section.csv` |
| `domains` | `domains_map_<section>` + `domains_map_legend`, `domains_composition` | the spatial domains drawn in each section, one colour per domain; cell-type composition of each domain | `bin_meta.csv`, `bin_clusters_*.csv`, `composition/` |
| `programmes` | `programmes_loadings`, `programmes_pathways_<programme>`, `programmes_amplitude` | the factor matrices: gene loadings B (top genes of each programme against every programme, each gene scaled to its largest loading); enriched terms per programme; per-section amplitude D_k | `B_gene_loadings.npy`, `genes.csv`, `program_top_genes.csv`, `program_enrichment.csv`, `w_by_sample.csv` |
| `usage` | `usage_<programme>_<section>` + `usage_<programme>_colorbar` | the columns of Q_k H drawn on the tissue: how much each tile uses each programme. One colour scale per programme, fixed across sections, stretched between the 5th and 95th percentile | `shared_coords_QH.npy`, `bin_meta.csv` |
| `deg` | `deg_volcano_<contrast>` + `deg_volcano_legend` | one volcano per differential-expression contrast, hits coloured and named, guides at the hit thresholds | `deg_out/deg_*.csv`, `deg_out/summary.csv` |
| `networks` | `network_domain<d>_<programme>`, `residual_domain<d>_<programme>` + `residual_legend` | each PPI network the sheaf ranked (one per programme per domain): layout by co-expression distance, node size by degree, node colour by charge, degree-residual set ringed and named; and its leave-one-out spectral shift against weighted degree | `sheaf/charges/`, `sheaf/expr/`, `sheaf/string400.csv`, `sheaf/out_rank/` |

`--programmes` restricts the `programmes` and `usage` groups to the programmes
named; by default every programme is drawn (the `usage` group then writes one
file per programme per section).

## Style

Set once in `style.py` and used by every plot:

- **One typeface, Times.** `Times New Roman` is used where it is installed;
  otherwise the plots fall back to `Nimbus Roman` or `STIXGeneral`, which are
  Times-metric.
- **One text size** (`FS`, 9 pt) and **one text colour**, black. Plots are drawn
  at the size they are meant to be placed, so scaling a plot scales its type.
- **Title Case**, through `cap()`, which leaves gene symbols, accessions and
  units alone.
- **No text over a mark and no titles**, beyond the one identifier a plot needs
  to be told from its siblings (a section, a contrast, a domain).
- **Colour carries data, never words.** Eight categorical slots in a fixed
  order, one sequential ramp and one warm/cool diverging pair; the case group is
  warm and the control group cool, whatever the data card names them.

## Adding a plot

Add it to the group function it belongs to, or write a new group. Build the
figure with `newfig(w, h)`, read everything through the `Run` object
(`run.fit(...)`, `run.path("deg", ...)`), call `save(fig, out, name)` once per
plot, and register a new group in `FIGURES` and `NEEDS` at the foot of the file.
If the encoding repeats across sibling plots, emit `legend_file(...)` rather
than a legend in every one. Keep dataset facts out of the code: a gene, a domain
number or a disease name written into this directory is the same bug as one
written into a prompt.
