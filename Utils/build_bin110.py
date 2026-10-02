#!/usr/bin/env python
"""
tissue.gef (bin1) -> square-bin h5ads, default bin110, for the sections of a data card.

    python build_bin110.py --card ../datasets/<name>/card.yaml \
        --gefdir ../data/<name>/<gef dir> --outdir ../data/<name>/bin110

Output is <SECTION>_bin<size>.h5ad, which is what prep_bins.py globs. The GEF
holds only single-DNB (bin1) records, so the tiles are aggregated here; at the
0.5 um DNB pitch bin110 is 55 um. Each section's GEF is found by its chip id
(*_<chip>.tissue.gef), and the section's stage and batch identifiers are
written into obs from the card. Format conversion only: no QC, no selection.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import load_yaml  # noqa: E402

def card_samples(card_path) -> dict:
    """{section: its sample row} from the data card."""
    return {str(s["section"]): s for s in load_yaml(card_path)["samples"]}


def annotate(A: ad.AnnData, sample: dict) -> ad.AnnData:
    """The card's labels for this section, which prep_bins.py and the cell typing read."""
    A.obs["section"] = str(sample["section"])
    A.obs["stage"] = str(sample["stage"])
    A.obs["chip"] = str(sample["chip"])
    A.obs["chip_lot"] = str(sample["chip_lot"])
    A.obs["gef_writer"] = str(sample["gef_writer"])
    return A


def _field(dt, *cands):
    names = {n.lower(): n for n in (dt.names or ())}
    for c in cands:
        if c.lower() in names:
            return names[c.lower()]
    return None


def build_one(gef: Path, bin_size: int) -> ad.AnnData:
    with h5py.File(gef, "r") as f:
        if "geneExp/bin1" not in f:
            raise KeyError(f"{gef.name}: no geneExp/bin1 "
                           f"(found: {list(f.get('geneExp', {}))})")
        grp = f["geneExp/bin1"]
        gene_ds, expr_ds = grp["gene"], grp["expression"]

        gdt = gene_ds.dtype
        gname_f = _field(gdt, "gene", "geneName", "geneID")
        goff_f = _field(gdt, "offset")
        gcnt_f = _field(gdt, "count", "cellCount", "geneCount")
        if None in (gname_f, goff_f, gcnt_f):
            raise KeyError(f"unexpected gene fields: {gdt.names}")

        g = gene_ds[:]
        gnames = g[gname_f]
        if gnames.dtype.kind == "S":
            gnames = np.char.decode(gnames, "utf-8", "replace")
        offs = g[goff_f].astype(np.int64)
        cnts = g[gcnt_f].astype(np.int64)

        edt = expr_ds.dtype
        x_f, y_f = _field(edt, "x"), _field(edt, "y")
        c_f = _field(edt, "count", "umi", "midcnt", "mid_count")
        if None in (x_f, y_f, c_f):
            raise KeyError(f"unexpected expression fields: {edt.names}")

        n_rec = expr_ds.shape[0]
        e = expr_ds[:]
        x = e[x_f].astype(np.int64)
        y = e[y_f].astype(np.int64)
        cnt = e[c_f].astype(np.float32)
        del e

        # Absolute chip coordinates: some writers store x/y relative to the
        # tissue bounding box and keep the origin in attrs. Add it back when
        # present so the binning grid is anchored the same way for every
        # section (and print it, because a silent origin shift would move
        # every tile by a fraction of a bin).
        ox = int(f.attrs.get("offsetX", grp.attrs.get("offsetX", 0)) or 0)
        oy = int(f.attrs.get("offsetY", grp.attrs.get("offsetY", 0)) or 0)
        if ox or oy:
            x += ox
            y += oy

    gene_id = np.repeat(np.arange(len(gnames), dtype=np.int32), cnts)
    if gene_id.shape[0] != n_rec:
        raise ValueError(f"gene offsets cover {gene_id.shape[0]} records but "
                         f"expression has {n_rec}")

    bx = x // bin_size
    by = y // bin_size
    del x, y
    nbx, nby = int(bx.max()) + 1, int(by.max()) + 1
    tile = bx.astype(np.int64) * nby + by.astype(np.int64)
    del bx, by

    # coo -> csr sums duplicate (tile, gene) pairs, which IS the aggregation
    M = sp.coo_matrix((cnt, (tile, gene_id)),
                      shape=(nbx * nby, len(gnames))).tocsr()
    del cnt, tile, gene_id

    keep = np.asarray((M > 0).sum(1)).ravel() > 0
    M = M[keep]
    idx = np.nonzero(keep)[0]
    tx, ty = idx // nby, idx % nby

    A = ad.AnnData(X=M.astype(np.float32))
    A.var_names = pd.Index(gnames)
    A.var_names_make_unique()
    A.obs_names = pd.Index([f"{i}_{j}" for i, j in zip(tx, ty)])
    # tile CENTRE in DNB units, so coordinates stay on the same scale as bin50
    A.obsm["spatial"] = np.c_[tx * bin_size + bin_size // 2,
                              ty * bin_size + bin_size // 2].astype(np.float32)
    A.obs["bx"], A.obs["by"] = tx, ty
    A.obs["n_counts"] = np.asarray(M.sum(1)).ravel()
    A.obs["n_genes"] = np.asarray((M > 0).sum(1)).ravel()
    A.uns["bin_size"] = int(bin_size)
    A.uns["offset"] = [ox, oy]
    A.uns["grid"] = [nbx, nby]
    return A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True, help="the data card: its sections are converted")
    p.add_argument("--gefdir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--bin-size", type=int, default=110)
    p.add_argument("--only", nargs="*", default=None, help="a subset of the card's sections")
    a = p.parse_args()

    samples = card_samples(a.card)
    gefdir, out = Path(a.gefdir), Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for s in (a.only or sorted(samples)):
        dst = out / f"{s}_bin{a.bin_size}.h5ad"
        chip = samples[s]["chip"]
        hits = sorted(gefdir.glob(f"*_{chip}.tissue.gef"))
        if not hits:
            print(f"[{s}] no tissue.gef for chip {chip} -- skipped")
            continue
        if dst.exists():
            print(f"[{s}] {dst.name} exists -- skipped")
            continue
        t0 = time.time()
        A = annotate(build_one(hits[0], a.bin_size), samples[s])
        A.write_h5ad(dst, compression="gzip")
        print(f"[{s}] {A.n_obs:,} tiles x {A.n_vars:,} genes  "
              f"median counts={np.median(A.obs['n_counts']):.0f}  "
              f"grid={A.uns['grid']}  offset={A.uns['offset']}  "
              f"({time.time()-t0:.0f}s) -> {dst.name}", flush=True)


if __name__ == "__main__":
    main()
