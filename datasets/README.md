# Data cards

A data card is the only description of a dataset the pipeline receives. A new
dataset is a new `datasets/<name>/card.yaml`; no code changes.

```
python run_all.py --card datasets/<name>/card.yaml --run <run id> --list
python run_all.py --card datasets/<name>/card.yaml --run <run id> --submit
```

Outputs go to `runs/<name>/<run id>/`; the written report is
`runs/<name>/<run id>/report/report.md`.

## What a card holds

| block | contents |
|---|---|
| `description` | tissue, technology, what was measured, disease, design; `presentation` (`focal` or `diffuse`: how the disease is laid out in tissue, which decides whether comparisons within diseased sections count as disease contrasts) and a one-line `presentation_note` |
| `condition` | the metadata column holding condition, the control label, the names of the two groups, the stage order |
| `samples` | one row per section: chip, stage, batch identifiers (chip lot, GEF writer), demographics, pathology scores. **Sections not listed are not part of the dataset** -- data dropped before analysis is simply not listed, and nothing downstream ever learns it existed. |
| `inputs` | where the converted per-section files are (`build_cellbin.py`, `build_bin110.py` output) |
| `storm_links` | the cross-section links of the STORM graph, chosen by the user (chip-lot matched here) |
| `priors` | the target number of spatial domains (the clustering resolution is searched to hit it); the expected cell types with marker genes; the map from reference-model labels onto those names |
| `policies` | rules every agent is given (e.g. do not look up the source publication) |

## What a card must not hold

Anything learned by analysing the data: readings of programmes or domains,
which genes are nuisances, statistics, which domains matter. Agents receive the
card only through `agents/common/render_notes.py`, which renders the description,
cohort, priors and policies and nothing else; QC statistics are computed by the
pipeline and reach the agents inside packets.

Method choices (ranks, thresholds, floors, the model) are not dataset facts
either: they live in `config/methods.yaml`, shared by every dataset.
