#!/usr/bin/env python
"""
Square-bin h5ads -> STORM-ready inputs.

  1. intersects genes across the card's sections (STORM's shared gene mode
     needs identical columns; union-with-zeros would encode "absent from this
     chip's annotation" as "not expressed")
  2. attaches the per-section metadata of the data card
  3. drops empty tiles (below --min-counts), with a ledger entry per decision
  4. attaches nuclear density and size from the matching *.cellbin.gef

No depth scaling, gene-count filter or mitochondrial filter is applied.

    python prep_bins.py --bindir <run>/bin110 --bin-size 110 --gefdir <run>/gef \
        --samples <run>/inputs/samples.csv --outdir <run>/prepped
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad_mod
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import rpath  # noqa: E402

STAGE_ORDER = {"NA": 0, "moderate": 1, "severe": 2}

MIN_COUNTS = 50      # empty-bin floor only; costs 0.4-1.6% of bins
LEDGER: list[dict] = []


def log(step, action, mechanism, evidence, affected=None):
    """Every removal of variation records WHY. Steps without one didn't happen."""
    LEDGER.append(dict(step=step, action=action, technical_mechanism=mechanism,
                       evidence=evidence, affected=affected))


# --------------------------------------------------------------- nuclei
def _field(dt, *cands):
    names = {n.lower(): n for n in (dt.names or ())}
    for c in cands:
        if c.lower() in names:
            return names[c.lower()]
    return None


def nuclear_features(cellbin_path: Path, coords: np.ndarray, bin_size: int):
    """
    Per-bin nuclear density and size from the ssDNA-derived cell segmentation.

    Independent of expression, so it can corroborate clusters without the
    circularity of validating expression clusters with expression markers.
    Nuclear density is the classic cytoarchitectural read on cortical layers
    (granular layer IV dense, layer I nearly acellular, white matter distinct),
    and nuclear area separates pyramidal neurons from glia.
    """
    import h5py
    with h5py.File(cellbin_path, "r") as f:
        key = "cellBin/cell" if "cellBin/cell" in f else None
        if key is None:
            for k in ("cellBin/cellBin", "cellbin/cell"):
                if k in f:
                    key = k
                    break
        if key is None:
            return None
        cell = f[key][:]

    xf, yf = _field(cell.dtype, "x"), _field(cell.dtype, "y")
    af = _field(cell.dtype, "area")
    if xf is None or yf is None:
        return None

    # Join on the (bx, by) PAIR directly. Reconstructing the linear bin key
    # would require the nby used at binning time, which is not recoverable
    # from coordinates alone -- and a wrong nby silently matches nothing.
    bins = pd.DataFrame({
        "bx": ((coords[:, 0] - bin_size / 2.0) / bin_size).round().astype(np.int64),
        "by": ((coords[:, 1] - bin_size / 2.0) / bin_size).round().astype(np.int64),
        "row": np.arange(coords.shape[0], dtype=np.int64)})
    cells = pd.DataFrame({
        "bx": (cell[xf].astype(np.int64) // bin_size),
        "by": (cell[yf].astype(np.int64) // bin_size),
        "area": (cell[af].astype(np.float64) if af is not None
                 else np.ones(cell.shape[0]))})
    j = cells.merge(bins, on=["bx", "by"], how="inner")
    if j.empty:
        return None

    n = coords.shape[0]
    dens = np.bincount(j["row"].to_numpy(), minlength=n).astype(np.float32)
    out = dict(n_nuclei=dens)
    if af is not None:
        w = j["area"].to_numpy()
        r = j["row"].to_numpy()
        s = np.bincount(r, weights=w, minlength=n)
        s2 = np.bincount(r, weights=w ** 2, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(dens > 0, s / np.maximum(dens, 1), np.nan)
            var = np.where(dens > 1, s2 / np.maximum(dens, 1) - mean ** 2, np.nan)
        out["nuclear_area_mean"] = mean.astype(np.float32)
        out["nuclear_area_sd"] = np.sqrt(np.maximum(var, 0)).astype(np.float32)
    return out


# ----------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bindir", required=True)
    p.add_argument("--gefdir", default=rpath("gse269906"))
    p.add_argument("--samples", required=True,
                   help="per-section metadata CSV written from the data card by run_all.py "
                        "(section, chip, stage, chip_lot, gef_writer, sex, age, apoe, thal, "
                        "cerad, braak). Only these sections are prepped, and their stage, "
                        "chip lot and writer are taken from it.")
    p.add_argument("--outdir", default=rpath("prepped"))
    p.add_argument("--bin-size", type=int, default=110)
    p.add_argument("--min-counts", type=int, default=MIN_COUNTS)
    a = p.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    card = pd.read_csv(a.samples, dtype=str, keep_default_na=False).set_index("section")
    files = sorted(Path(a.bindir).glob(f"*_bin{a.bin_size}.h5ad"))
    secs = [f.name.split("_bin")[0] for f in files]
    missing = sorted(set(card.index) - set(secs))
    if missing:
        sys.exit(f"card sections with no *_bin{a.bin_size}.h5ad in {a.bindir}: {missing}")
    keep = [(f, s) for f, s in zip(files, secs) if s in card.index]
    if not keep:
        sys.exit(f"no matching h5ads in {a.bindir}")
    print(f"samples: {[s for _, s in keep]}")

    # --- metadata -------------------------------------------------------
    num = lambda v: pd.to_numeric(v, errors="coerce")
    md = pd.DataFrame({"Thal": num(card.thal), "CERAD": num(card.cerad),
                       "Braak": num(card.braak), "Age": card.age,
                       "ApoE Genotype": card.apoe.replace({"": "unknown"}),
                       "Sex": card.sex}).set_index(card.chip)
    md["age_mid"] = md["Age"].astype(str).str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2)
    known = md["ApoE Genotype"].notna() & (md["ApoE Genotype"].astype(str) != "unknown")
    md["apoe_e4"] = md["ApoE Genotype"].astype(str).str.count("E4").where(known)

    # --- pass 1: gene intersection --------------------------------------
    gsets = []
    for f, s in keep:
        A = ad_mod.read_h5ad(f, backed="r")
        gsets.append(set(A.var_names))
        A.file.close()
    shared = sorted(set.intersection(*gsets))
    print(f"gene intersection: {len(shared)} "
          f"(per-sample {min(len(g) for g in gsets)}-{max(len(g) for g in gsets)})")
    log("genes", f"intersect to {len(shared)}",
        "STORM's shared gene mode requires identical columns across samples",
        f"per-sample annotations differ ({min(len(g) for g in gsets)}"
        f"-{max(len(g) for g in gsets)} genes)")

    # --- pass 2: subset, QC, annotate ------------------------------------
    rows = []
    for f, s in keep:
        A = ad_mod.read_h5ad(f)
        A = A[:, shared].copy()
        chip = card.loc[s, "chip"]
        # the card is the source of truth for condition and batch labels
        A.obs["stage"] = card.loc[s, "stage"]
        A.obs["chip"] = chip
        A.obs["chip_lot"] = card.loc[s, "chip_lot"]
        A.obs["gef_writer"] = card.loc[s, "gef_writer"]

        c = np.asarray(A.X.sum(1)).ravel()
        n0 = A.n_obs
        mask = c >= a.min_counts
        A = A[mask].copy()
        log(f"{s}: bins", f"drop {n0 - A.n_obs} of {n0} bins",
            f"empty/near-empty bins below {a.min_counts} counts carry no "
            "information and are off-tissue or edge artifacts",
            f"{(n0 - A.n_obs) / n0:.2%} removed")

        if chip in md.index:
            r = md.loc[chip]
            for k, v in dict(thal=r["Thal"], cerad=r["CERAD"], braak=r["Braak"],
                             age_mid=r["age_mid"], apoe=str(r["ApoE Genotype"]),
                             apoe_e4=r["apoe_e4"], sex=r["Sex"]).items():
                A.obs[k] = v
        A.obs["stage_ord"] = STAGE_ORDER.get(str(A.obs["stage"].iloc[0]), -1)

        nf = None
        cb = list(Path(a.gefdir).glob(f"*_{chip}.cellbin.gef"))
        if cb:
            try:
                nf = nuclear_features(cb[0], np.asarray(A.obsm["spatial"], float),
                                      a.bin_size)
            except Exception as ex:
                print(f"  {s}: nuclei failed ({type(ex).__name__}: {ex})")
        if nf:
            for k, v in nf.items():
                A.obs[k] = v
            A.uns["has_nuclear_features"] = True

        A.write_h5ad(out / f"{s}_prepped.h5ad", compression="gzip")
        cc = np.asarray(A.X.sum(1)).ravel()
        rows.append(dict(
            section=s, chip=chip, stage=str(A.obs["stage"].iloc[0]),
            lot=str(A.obs["chip_lot"].iloc[0]),
            writer=str(A.obs["gef_writer"].iloc[0]),
            n_bins=int(A.n_obs), dropped=int(n0 - A.n_obs),
            med_counts=float(np.median(cc)),
            braak=float(A.obs["braak"].iloc[0]) if "braak" in A.obs else np.nan,
            nuclei_med=(float(np.median(A.obs["n_nuclei"]))
                        if "n_nuclei" in A.obs else np.nan),
            nuc_area_med=(float(np.nanmedian(A.obs["nuclear_area_mean"]))
                          if "nuclear_area_mean" in A.obs else np.nan)))
        print(f"  {s:>6s} {rows[-1]['stage']:>8s} n={A.n_obs:,} "
              f"(-{rows[-1]['dropped']}) med_counts={rows[-1]['med_counts']:.0f} "
              f"nuclei/bin={rows[-1]['nuclei_med']:.1f}")
        del A

    df = pd.DataFrame(rows)
    df.to_csv(out / "prep_summary.csv", index=False)
    (out / "qc_ledger.json").write_text(json.dumps(LEDGER, indent=2))

    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False))
    print(f"\nledger: {out/'qc_ledger.json'} ({len(LEDGER)} entries)")
    print("NOT applied: depth scaling, gene-count filter, mitochondrial filter")


if __name__ == "__main__":
    main()
