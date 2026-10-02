"""
Shared helpers for the domain-track scripts (plot_domains.py,
compare_domains.py, compose_domains.py, STORM/sweep_resolution.py).

Kept deliberately tiny: reading a run directory correctly (the control stage
label is the literal string "NA", so pandas must not parse it as missing) and
the k-nearest-neighbour label-agreement score that every script reports.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

META_DTYPES = {"section": str, "stage": str, "lot": str}


def read_meta(run: str, clusters: Optional[str] = None) -> pd.DataFrame:
    """
    bin_meta.csv of a STORM or SCC run, optionally joined with a cluster file
    (default bin_clusters.csv) as column 'cl' (string). Rows align by position,
    which is how storm_chain_run.py and scc_baseline_bin110.py write them.
    """
    meta = pd.read_csv(os.path.join(run, "bin_meta.csv"),
                       keep_default_na=False, dtype=META_DTYPES)
    if clusters is not None:
        path = clusters if os.path.isabs(clusters) else os.path.join(run, clusters)
        lab = pd.read_csv(path)["cluster"].astype(str).to_numpy()
        if len(lab) != len(meta):
            raise ValueError(f"{path}: {len(lab)} labels for {len(meta)} bins")
        meta = meta.assign(cl=lab)
    return meta


def spatial_coherence(meta: pd.DataFrame, labels: Optional[np.ndarray] = None,
                      k: int = 8) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Per bin: fraction of its k nearest spatial neighbours (within the same
    section) that share its label. Returns (per-bin array, per-label table with
    columns spatial_coherence and n_bins).

    Descriptive only: a spatially regularised fit will score high partly
    because of its prior. The informative direction is LOW coherence.
    """
    from scipy.spatial import cKDTree
    lab = meta["cl"].to_numpy() if labels is None else np.asarray(labels).astype(str)
    coh = np.full(len(meta), np.nan)
    for sec, idx in meta.groupby("section").indices.items():
        xy = meta.iloc[idx][["x", "y"]].to_numpy(dtype=float)
        kk = min(k + 1, len(idx))
        if kk < 2:
            coh[idx] = 1.0
            continue
        _, nn = cKDTree(xy).query(xy, k=kk)
        L = lab[idx]
        coh[idx] = (L[nn[:, 1:]] == L[:, None]).mean(1)
    tbl = (pd.DataFrame(dict(cl=lab, coh=coh)).groupby("cl").coh
           .agg(["mean", "size"])
           .rename(columns={"mean": "spatial_coherence", "size": "n_bins"})
           .sort_values("spatial_coherence", ascending=False))
    return coh, tbl


def stage_purity(meta: pd.DataFrame, labels: Optional[np.ndarray] = None
                 ) -> pd.DataFrame:
    """Per label: dominant stage, its fraction, and max fraction from one section."""
    lab = meta["cl"].to_numpy() if labels is None else np.asarray(labels).astype(str)
    m = meta.assign(cl=lab)
    rows = []
    for cl, sub in m.groupby("cl"):
        st = sub.stage.value_counts(normalize=True)
        sec = sub.section.value_counts(normalize=True)
        rows.append(dict(cl=cl, n_bins=len(sub), dominant_stage=st.index[0],
                         stage_purity=float(st.iloc[0]),
                         max_section=float(sec.iloc[0]),
                         n_sections=int(sub.section.nunique())))
    return pd.DataFrame(rows).set_index("cl")
