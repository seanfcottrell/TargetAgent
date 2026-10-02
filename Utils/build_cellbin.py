#!/usr/bin/env python
"""
cellbin.gef -> per-section nucleus x gene h5ads, for the sections of a data card.

    python build_cellbin.py --card ../datasets/<name>/card.yaml \
        --gefdir ../data/<name>/<gef dir> --outdir ../data/<name>/cellbin

Uses h5py only. cellbin GEFs come in two indexings and the field names vary by
writer version:

  gene-major : cellBin/gene    (geneName, offset, count)
               cellBin/geneExp (cellID, count)
  cell-major : cellBin/cell    (id, x, y, offset, count, area, ...)
               cellBin/cellExp (geneID, count)

Either reconstructs the same matrix. The layout is detected, the index ranges
are cross-checked against the expression table, and an unrecognised structure
raises rather than being guessed at. Format conversion only: every segmented
nucleus is kept, including zero-count ones; QC happens in the cell-typing stage.
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

from build_bin110 import card_samples, annotate


def _f(dt, *cands):
    names = {n.lower(): n for n in (dt.names or ())}
    for c in cands:
        if c.lower() in names:
            return names[c.lower()]
    return None


def _describe(g, prefix="cellBin"):
    out = []
    for k in g:
        n = g[k]
        if isinstance(n, h5py.Dataset):
            out.append(f"    {prefix}/{k}: shape={n.shape} "
                       f"fields={n.dtype.names or n.dtype}")
    return "\n".join(out)


def build_one(gef: Path) -> ad.AnnData:
    with h5py.File(gef, "r") as f:
        if "cellBin" not in f:
            raise KeyError(f"{gef.name}: no cellBin group; found {list(f)}")
        cb = f["cellBin"]
        struct = _describe(cb)

        if "cell" not in cb:
            raise KeyError(f"{gef.name}: no cellBin/cell\n{struct}")
        cell = cb["cell"][:]
        cdt = cell.dtype
        x_f, y_f = _f(cdt, "x"), _f(cdt, "y")
        if None in (x_f, y_f):
            raise KeyError(f"{gef.name}: cell has no x/y\n{struct}")
        area_f = _f(cdt, "area")
        n_cells = cell.shape[0]

        # ---- gene-major first: cellBin/gene + cellBin/geneExp ----------
        if "gene" in cb and "geneExp" in cb:
            gdt = cb["gene"].dtype
            gname_f = _f(gdt, "geneName", "gene", "geneID")
            goff_f, gcnt_f = _f(gdt, "offset"), _f(gdt, "count", "cellCount")
            edt = cb["geneExp"].dtype
            cid_f = _f(edt, "cellID", "cellid", "cell")
            cnt_f = _f(edt, "count", "midcnt", "umi")
            if None in (gname_f, goff_f, gcnt_f, cid_f, cnt_f):
                raise KeyError(f"{gef.name}: gene-major fields not recognised\n"
                               f"    gene={gdt.names}\n    geneExp={edt.names}\n"
                               f"{struct}")
            g = cb["gene"][:]
            names = g[gname_f]
            if names.dtype.kind == "S":
                names = np.char.decode(names, "utf-8", "replace")
            offs, cnts = g[goff_f].astype(np.int64), g[gcnt_f].astype(np.int64)
            e = cb["geneExp"][:]
            rows = e[cid_f].astype(np.int64)
            vals = e[cnt_f].astype(np.float32)
            cols = np.repeat(np.arange(len(names), dtype=np.int32), cnts)
            mode = "gene-major"

        # ---- else cell-major: cellBin/cell + cellBin/cellExp -----------
        elif "cellExp" in cb:
            off_f, cnt_f0 = _f(cdt, "offset"), _f(cdt, "count", "geneCount")
            edt = cb["cellExp"].dtype
            gid_f = _f(edt, "geneID", "geneid", "gene")
            cnt_f = _f(edt, "count", "midcnt", "umi")
            if None in (off_f, cnt_f0, gid_f, cnt_f):
                raise KeyError(f"{gef.name}: cell-major fields not recognised\n"
                               f"    cell={cdt.names}\n    cellExp={edt.names}\n"
                               f"{struct}")
            cnts = cell[cnt_f0].astype(np.int64)
            e = cb["cellExp"][:]
            cols = e[gid_f].astype(np.int64)
            vals = e[cnt_f].astype(np.float32)
            rows = np.repeat(np.arange(n_cells, dtype=np.int64), cnts)
            gl = cb.get("geneList", cb.get("gene"))
            if gl is None:
                raise KeyError(f"{gef.name}: cell-major but no gene names\n"
                               f"{struct}")
            gg = gl[:]
            names = gg if gg.dtype.names is None else gg[_f(gg.dtype,
                                                            "geneName", "gene")]
            if names.dtype.kind == "S":
                names = np.char.decode(names, "utf-8", "replace")
            mode = "cell-major"
        else:
            raise KeyError(f"{gef.name}: neither gene-major nor cell-major "
                           f"layout recognised\n{struct}")

        if len(rows) != len(cols) or len(rows) != len(vals):
            raise ValueError(f"{gef.name}: index/value length mismatch "
                             f"{len(rows)}/{len(cols)}/{len(vals)}")
        if int(cnts.sum()) != len(vals):
            raise ValueError(f"{gef.name}: offset/count table covers "
                             f"{int(cnts.sum())} records but the expression "
                             f"table has {len(vals)} -- refusing to guess")
        if rows.max() >= n_cells:
            raise ValueError(f"{gef.name}: cellID {rows.max()} exceeds "
                             f"{n_cells} cells")

        cx = cell[x_f].astype(np.float32)
        cy = cell[y_f].astype(np.float32)
        carea = (cell[area_f].astype(np.float32) if area_f
                 else np.full(n_cells, np.nan, np.float32))

    M = sp.coo_matrix((vals, (rows, cols)),
                      shape=(n_cells, len(names))).tocsr()

    A = ad.AnnData(X=M.astype(np.float32))
    A.var_names = pd.Index(names)
    A.var_names_make_unique()
    A.obs_names = pd.Index(np.arange(n_cells).astype(str))
    A.obsm["spatial"] = np.c_[cx, cy]
    A.obs["x"], A.obs["y"], A.obs["area"] = cx, cy, carea
    A.obs["n_counts"] = np.asarray(M.sum(1)).ravel()
    A.obs["n_genes"] = np.asarray((M > 0).sum(1)).ravel()
    A.uns["gef_layout"] = mode
    return A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True, help="the data card: its sections are converted")
    p.add_argument("--gefdir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--only", nargs="*", default=None, help="a subset of the card's sections")
    a = p.parse_args()

    samples = card_samples(a.card)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for s in (a.only or sorted(samples)):
        chip = samples[s]["chip"]
        hits = sorted(Path(a.gefdir).glob(f"*_{chip}.cellbin.gef"))
        if not hits:
            print(f"[{s}] no cellbin.gef for chip {chip} -- skipped")
            continue
        dst = out / f"{s}_cellbin.h5ad"
        if dst.exists():
            print(f"[{s}] exists -- skipped")
            continue
        t0 = time.time()
        A = annotate(build_one(hits[0]), samples[s])
        A.write_h5ad(dst, compression="gzip")
        z = int((A.obs["n_counts"] == 0).sum())
        print(f"[{s}] {A.n_obs:,} cells x {A.n_vars:,} genes  "
              f"[{A.uns['gef_layout']}]  "
              f"median counts={np.median(A.obs['n_counts']):.0f}  "
              f"zero-count={z:,} ({z/A.n_obs:.1%})  "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
