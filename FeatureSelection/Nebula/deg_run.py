from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np, pandas as pd
import anndata as ad
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import RUN_ROOT as BASE  # noqa: E402  the run directory (celltypes_qc/, cellbin/, prepped/)

NA = dict(keep_default_na=False)
LN2 = np.log(2.0)
FIN = CLUSTERS = None      # the fitted run and its partition file (set in main)
SECS, AD = [], []          # every donor section of the cohort; the case sections


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def load_stratum(domain, ctype, bin_size=110, return_types=False):
    cells = pd.read_csv(os.path.join(BASE, "celltypes_qc", "cells.csv.gz"), dtype={"section": str}, **NA)
    cells["type_final"] = cells.type_final.replace("", "Unk")
    cells["bx"] = (cells.x.astype(float) // bin_size).astype(np.int64)
    cells["by"] = (cells.y.astype(float) // bin_size).astype(np.int64)
    meta = pd.read_csv(os.path.join(FIN, "bin_meta.csv"), dtype={"section": str}, **NA)
    meta["cl"] = pd.read_csv(os.path.join(FIN, CLUSTERS), dtype={"cluster": str}, **NA).cluster.values
    meta["bx"] = np.round((meta.x.astype(float) - bin_size / 2.0) / bin_size).astype(np.int64)
    meta["by"] = np.round((meta.y.astype(float) - bin_size / 2.0) / bin_size).astype(np.int64)
    J = cells.merge(meta[["section", "bx", "by", "cl"]], on=["section", "bx", "by"], how="left")
    sel = (J.cl == str(domain))
    sel &= (J.type_final != "Unk") if ctype == "all" else (J.type_final == ctype)
    keep = J[sel]
    print(f"stratum domain {domain} x {ctype}: {len(keep):,} nuclei", flush=True)
    print(keep.groupby("section").size().reindex(SECS).fillna(0).astype(int).to_string(), flush=True)
    # the per-section cellbin files do NOT share a gene space: intersect first,
    # keeping the first section's (alphabetical) order
    vs = {}
    for s in SECS:
        A = ad.read_h5ad(os.path.join(BASE, "cellbin", f"{s}_cellbin.h5ad"), backed="r")
        vs[s] = pd.Index(A.var_names.astype(str))
        A.file.close()
    common = vs[SECS[0]]
    for s in SECS[1:]:
        common = common[common.isin(set(vs[s]))]
    print(f"gene space: {len(common):,} genes common to all {len(SECS)} sections "
          f"(per-section {min(len(v) for v in vs.values()):,}-{max(len(v) for v in vs.values()):,})", flush=True)

    X, donors, types = [], [], []
    for s in SECS:
        sub_k = keep[keep.section == s]
        ids = set(sub_k.cell_id.astype(str))
        tmap = dict(zip(sub_k.cell_id.astype(str), sub_k.type_final.astype(str)))
        if not ids:
            continue
        # read in memory (not backed): across anndata versions a backed view's
        # .X can be a wrapper that neither issparse() nor np.asarray() handles,
        # and scipy >= 1.17 then rejects the object dtype. Columns are selected
        # positionally so every block shares the gene order.
        A = ad.read_h5ad(os.path.join(BASE, "cellbin", f"{s}_cellbin.h5ad"))
        m = np.fromiter((n in ids for n in A.obs_names.astype(str)), bool, A.n_obs)
        pos = pd.Index(A.var_names.astype(str)).get_indexer(common)
        Xs = A.X
        if hasattr(Xs, "to_memory"):
            Xs = Xs.to_memory()
        Xs = Xs.tocsr() if sparse.issparse(Xs) else sparse.csr_matrix(np.asarray(Xs, dtype=np.float32))
        X.append(Xs[m][:, pos].astype(np.float32))
        donors += [s] * int(m.sum())
        types += [tmap[n] for n in A.obs_names.astype(str)[m]]
        print(f"  {s}: {int(m.sum())} nuclei", flush=True)
        del A, Xs
    out = (sparse.vstack(X).tocsr(), np.array(donors), pd.Index(common))
    return out + (np.array(types),) if return_types else out


class Stratum:
    """Dense counts of one stratum, its donors, and the per-nucleus library size."""

    def __init__(self, C, donors):
        self.Cd = np.asarray(C.todense(), dtype=np.int32)
        self.donors = donors
        self.lib = self.Cd.sum(1).astype(np.float64)
        self.lib[self.lib == 0] = 1.0


def nebula_rule(Cd, donors, grp_a, lib, genes, ncore=4, types=None):
    """NEBULA NBGMM. Cd is nuclei x genes (int); NEBULA wants genes x cells with
    cells grouped by donor, which the loader guarantees (whole sections stacked).
    Returns (padj, log2FC) or None when NEBULA is unavailable."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except Exception as e:                                  # rpy2 missing
        print(f"    [nebula] rpy2 unavailable ({type(e).__name__}); skipped", flush=True)
        return None
    try:
        neb = importr("nebula")
        mat = importr("Matrix")
    except Exception as e:
        print(f"    [nebula] R package missing ({e}); skipped", flush=True)
        return None

    from scipy import sparse
    M = sparse.csc_matrix(Cd.T)                             # genes x cells
    try:
        R = ro.r
        R.assign("i_", ro.IntVector(M.indices.astype(int)))
        R.assign("p_", ro.IntVector(M.indptr.astype(int)))
        R.assign("x_", ro.FloatVector(M.data.astype(float)))
        R.assign("dims_", ro.IntVector(list(M.shape)))
        R("cnt <- Matrix::sparseMatrix(i = i_, p = p_, x = x_, dims = dims_, index1 = FALSE)")
        grp = np.where(np.isin(donors, list(grp_a)), "dis", "ref")
        R.assign("id_", ro.StrVector(list(donors)))
        R.assign("grp_", ro.StrVector(list(grp)))
        R.assign("lib_", ro.FloatVector(np.asarray(lib, float)))
        R('grp_ <- factor(grp_, levels = c("ref", "dis"))')
        if types is None:
            R("pred_ <- model.matrix(~ grp_)")
        else:                      # composition-adjusted: cell type as a covariate
            R.assign("ct_", ro.StrVector(list(types)))
            R("ct_ <- factor(ct_)")
            R("pred_ <- model.matrix(~ grp_ + ct_)")
        t0 = time.time()
        R(f'fit_ <- nebula::nebula(count = cnt, id = id_, pred = pred_, '
          f'offset = lib_, model = "NBGMM", cpc = 0, ncore = {int(ncore)})')
        S = R("fit_$summary")
        cols = list(R("colnames(fit_$summary)"))
        got = {c: np.asarray(S.rx2(c), float) for c in cols if c.startswith(("logFC_", "p_"))}
        lfc_col = next(c for c in got if c.startswith("logFC_") and c.endswith("grp_dis"))
        p_col = next(c for c in got if c.startswith("p_") and c.endswith("grp_dis"))
        # NEBULA can drop genes it cannot fit; align by its gene index
        idx = np.asarray(R("fit_$summary$gene_id"), int) - 1 if "gene_id" in cols else np.arange(len(got[p_col]))
        p = np.ones(Cd.shape[1]); l2 = np.zeros(Cd.shape[1])
        p[idx] = got[p_col]
        l2[idx] = got[lfc_col] / LN2                        # NEBULA reports natural-log FC
        print(f"    [nebula] {len(idx)}/{Cd.shape[1]} genes fitted in {time.time()-t0:.0f}s", flush=True)
        return bh(np.nan_to_num(p, nan=1.0)), l2
    finally:
        pass


def nebula_design(Cd, ids, lib, factors, numeric, formula, coef, ncore=4):
    """NEBULA NBGMM with a general fixed-effect design (donor random intercept,
    library-size offset). factors: {name: (values, levels)} become R factors with
    the first level as reference; numeric: {name: values}; formula is an R formula
    over those names; coef is the model-matrix column tested. Nuclei of one donor
    must be contiguous in Cd. Returns (padj, log2FC) or None when NEBULA is
    unavailable. Names are kept plain (no ':' interaction columns: an interaction
    enters as a numeric product column) so NEBULA's summary names are unambiguous."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        importr("nebula"); importr("Matrix")
    except Exception as e:
        print(f"    [nebula] unavailable ({type(e).__name__}: {e}); skipped", flush=True)
        return None
    from scipy import sparse
    M = sparse.csc_matrix(Cd.T)                             # genes x cells
    R = ro.r
    R.assign("i_", ro.IntVector(M.indices.astype(int)))
    R.assign("p_", ro.IntVector(M.indptr.astype(int)))
    R.assign("x_", ro.FloatVector(M.data.astype(float)))
    R.assign("dims_", ro.IntVector(list(M.shape)))
    R("cnt <- Matrix::sparseMatrix(i = i_, p = p_, x = x_, dims = dims_, index1 = FALSE)")
    R.assign("id_", ro.StrVector(list(ids)))
    R.assign("lib_", ro.FloatVector(np.asarray(lib, float)))
    cols = []
    for name, (vals, levels) in factors.items():
        R.assign(name, ro.StrVector(list(vals)))
        R.assign(f"{name}_lv", ro.StrVector(list(levels)))
        R(f"{name} <- factor({name}, levels = {name}_lv)")
        cols.append(name)
    for name, vals in numeric.items():
        R.assign(name, ro.FloatVector(np.asarray(vals, float)))
        cols.append(name)
    R(f"df_ <- data.frame({', '.join(cols)})")
    R(f"pred_ <- model.matrix({formula}, data = df_)")
    have = list(R("colnames(pred_)"))
    if coef not in have:
        raise ValueError(f"coefficient {coef} not in the design {have}")
    t0 = time.time()
    R(f'fit_ <- nebula::nebula(count = cnt, id = id_, pred = pred_, '
      f'offset = lib_, model = "NBGMM", cpc = 0, ncore = {int(ncore)})')
    S = R("fit_$summary")
    names = list(R("colnames(fit_$summary)"))
    lfc = np.asarray(S.rx2(f"logFC_{coef}"), float)
    pv = np.asarray(S.rx2(f"p_{coef}"), float)
    idx = np.asarray(R("fit_$summary$gene_id"), int) - 1 if "gene_id" in names else np.arange(len(pv))
    p = np.ones(Cd.shape[1]); l2 = np.zeros(Cd.shape[1])
    p[idx] = pv
    l2[idx] = lfc / LN2                                     # NEBULA reports natural-log FC
    print(f"    [nebula] {len(idx)}/{Cd.shape[1]} genes fitted in {time.time()-t0:.0f}s ({formula}; test {coef})", flush=True)
    return bh(np.nan_to_num(p, nan=1.0)), l2


DEPTH = None      # donor -> centred log median counts per bin, when --depth-covariate


def fit(S, case, types, ncore):
    if DEPTH is None:                  # the original design, kept for exact reproducibility
        q, l2 = nebula_rule(S.Cd, S.donors, case, S.lib, None, ncore=ncore, types=types)
        return pd.DataFrame({"padj": q, "log2FC": l2})
    grp = np.where(np.isin(S.donors, list(case)), "case", "control")
    factors = {"grp": (grp, ["control", "case"])}
    formula = "~ grp + depth"
    if types is not None:
        factors["ct"] = (types, sorted(set(types)))
        formula += " + ct"
    q, l2 = nebula_design(S.Cd, S.donors, S.lib, factors,
                          {"depth": [DEPTH[d] for d in S.donors]}, formula, "grpcase", ncore=ncore)
    return pd.DataFrame({"padj": q, "log2FC": l2})


def run_contrast(domain, ctype, a):
    covar = (ctype == "all")
    out = load_stratum(domain, ctype, return_types=covar)
    C, donors, genes = out[0], out[1], out[2]
    types = out[3] if covar else None
    det = np.asarray((C > 0).mean(0)).ravel()
    keep = det >= a.min_frac                     # genes only; types are per nucleus
    C, genes = C[:, keep], genes[keep]
    print(f"  {C.shape[0]:,} nuclei x {C.shape[1]:,} genes", flush=True)
    S = Stratum(C, donors)
    per_donor = pd.Series(donors).value_counts().reindex(SECS).fillna(0).astype(int)

    res = fit(S, a.case, types, a.ncore)
    res.insert(0, "gene", genes)
    res["hit"] = (res.padj < 0.05) & (res.log2FC.abs() > 1)
    n_hit = int(res.hit.sum())
    row = dict(domain=domain, cell_type=ctype, n_nuclei=int(C.shape[0]), n_genes=int(C.shape[1]),
               min_nuclei_per_donor=int(per_donor.min()), n_hits=n_hit,
               n_fdr10=int(((res.padj < 0.10) & (res.log2FC.abs() > 1)).sum()))

    os.makedirs(a.out, exist_ok=True)
    res.sort_values("padj").to_csv(os.path.join(a.out, f"deg_domain{domain}_{ctype}.csv"), index=False)
    print(f"  -> {n_hit} hits at FDR 0.05 & |log2FC|>1", flush=True)
    return row


def run_designed(c, a, cache):
    """A contrast the domain panel designed (select_domains.py): nuclei of side_a
    against side_b, each side = its domains x its group ('case', 'control', or
    'both' in an interaction), one cell type (or 'all', with cell type in the
    model). Donor random intercept throughout, so a donor contributing to both
    sides is its own control (within-group and interaction designs), and a
    between-group design is a between-donor comparison as in the default.
      difference:  ~ side (+ cell type)                 test side_a vs side_b
      interaction: ~ side + group + side x group (+ ct) test the side x group term"""
    ctype = c["type"]; covar = ctype == "all"
    case = set(AD)
    C_, donors_, side_, grp_, types_ = [], [], [], [], []
    genes = None
    for side in ("side_a", "side_b"):
        sp = c[side]
        for d in sp["domains"]:
            key = (str(d), ctype)
            if key not in cache:
                cache[key] = load_stratum(str(d), ctype, return_types=covar)
            out = cache[key]
            Cm, dn, genes = out[0], np.asarray(out[1]), out[2]
            g = np.where(np.isin(dn, list(case)), "case", "control")
            keep = np.ones(len(dn), bool) if sp["group"] == "both" else (g == sp["group"])
            C_.append(Cm[keep]); donors_.append(dn[keep]); grp_.append(g[keep])
            side_.append(np.full(int(keep.sum()), "A" if side == "side_a" else "B"))
            if covar:
                types_.append(np.asarray(out[3])[keep])
    C = sparse.vstack(C_).tocsr()
    donors, side, grp = np.concatenate(donors_), np.concatenate(side_), np.concatenate(grp_)
    types = np.concatenate(types_) if covar else None
    o = np.argsort(donors, kind="stable")                # NEBULA: a donor's nuclei contiguous
    C, donors, side, grp = C[o], donors[o], side[o], grp[o]
    if covar:
        types = types[o]
    det = np.asarray((C > 0).mean(0)).ravel()
    keep = det >= a.min_frac
    C, genes = C[:, keep], genes[keep]
    Cd = np.asarray(C.todense(), dtype=np.int32)
    lib = Cd.sum(1).astype(np.float64); lib[lib == 0] = 1.0   # as Stratum does for the default
    tab = pd.crosstab(pd.Series(donors, name="donor"), pd.Series(side, name="side"))
    tab = tab.reindex(columns=["A", "B"], fill_value=0)
    n_a, n_b = int((tab.A > 0).sum()), int((tab.B > 0).sum())
    print(f"  {Cd.shape[0]:,} nuclei x {Cd.shape[1]:,} genes; donors with nuclei on side_a {n_a}, side_b {n_b}", flush=True)
    print(tab.to_string(), flush=True)
    factors = {"side": (side, ["B", "A"])}
    numeric = {}
    if c["test"] == "interaction":
        both = tab[(tab.A > 0) & (tab.B > 0)].index
        gof = dict(zip(donors, grp))
        n_case = sum(gof[x] == "case" for x in both); n_ctrl = len(both) - n_case
        if min(n_case, n_ctrl) < 2:
            raise ValueError(f"interaction needs >= 2 donors per group with nuclei on both sides "
                             f"(case {n_case}, control {n_ctrl})")
        factors["grp"] = (grp, ["control", "case"])
        numeric["inter"] = ((side == "A") & (grp == "case")).astype(float)
        formula, coef = "~ side + grp + inter", "inter"
    else:
        if min(n_a, n_b) < 2:
            raise ValueError(f"each side needs nuclei from >= 2 donors (side_a {n_a}, side_b {n_b})")
        formula, coef = "~ side", "sideA"
    if covar:
        factors["ct"] = (types, sorted(set(types)))
        formula += " + ct"
    if DEPTH is not None:
        numeric["depth"] = [DEPTH[d] for d in donors]
        formula += " + depth"
    q, l2 = nebula_design(Cd, donors, lib, factors, numeric, formula, coef, ncore=a.ncore)
    res = pd.DataFrame({"gene": genes, "padj": q, "log2FC": l2})
    res["hit"] = (res.padj < 0.05) & (res.log2FC.abs() > 1)
    res.sort_values("padj").to_csv(os.path.join(a.out, f"deg_contrast_{c['id']}.csv"), index=False)
    n_hit = int(res.hit.sum())
    print(f"  -> {n_hit} hits at FDR 0.05 & |log2FC|>1", flush=True)
    desc = lambda sp: f"{sp['group']}:{'+'.join(map(str, sp['domains']))}"
    return dict(contrast_id=c["id"], test=c["test"], side_a=desc(c["side_a"]), side_b=desc(c["side_b"]),
                cell_type=ctype, disease_informative=bool(c.get("disease_informative", True)),
                n_nuclei=int(Cd.shape[0]), n_genes=int(Cd.shape[1]),
                donors_side_a=n_a, donors_side_b=n_b,
                min_nuclei_per_donor=int(pd.Series(donors).value_counts().min()), n_hits=n_hit,
                n_fdr10=int(((res.padj < 0.10) & (res.log2FC.abs() > 1)).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contrasts", default=None, help="JSON list of {domain, type}")
    p.add_argument("--domain"); p.add_argument("--type", default="all")
    p.add_argument("--run", required=True); p.add_argument("--clusters", required=True)
    p.add_argument("--sections", nargs="+", required=True, help="every donor section in the cohort")
    p.add_argument("--case", nargs="+", required=True, help="the case sections (the rest are controls)")
    p.add_argument("--min-frac", type=float, default=0.05)
    p.add_argument("--depth-covariate", action="store_true",
                   help="per-section depth (centred log median counts per bin, prepped/prep_summary.csv) "
                        "as a fixed covariate in every contrast; the per-nucleus library-size offset stays")
    p.add_argument("--ncore", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    global FIN, CLUSTERS, SECS, AD, DEPTH
    FIN, CLUSTERS = a.run, a.clusters
    SECS = list(a.sections)                       # the cohort, in the order given
    AD = [s for s in SECS if s in set(a.case)]    # the case sections
    if a.depth_covariate:
        ps = pd.read_csv(os.path.join(BASE, "prepped", "prep_summary.csv"), dtype={"section": str},
                         keep_default_na=False).set_index("section")
        ld = np.log(ps.loc[SECS, "med_counts"].astype(float))
        DEPTH = (ld - ld.mean()).round(6).to_dict()
        print(f"depth covariate (centred log median counts per bin): {DEPTH}", flush=True)
    print(f"partition {a.run} / {a.clusters}; cohort {len(SECS)} donors, case {AD}", flush=True)

    todo = json.load(open(a.contrasts)) if a.contrasts else [{"domain": a.domain, "type": a.type}]
    rows, failed = [], []
    os.makedirs(a.out, exist_ok=True)
    designed = [c for c in todo if "test" in c]
    json.dump(designed, open(os.path.join(a.out, "designed_contrasts.json"), "w"), indent=1)
    cache = {}
    for c in todo:
        t0 = time.time()
        if "test" in c:
            label = dict(contrast_id=c["id"], test=c["test"], cell_type=c["type"])
            print(f"\n=== {c['id']} ({c['test']}): {c['side_a']} vs {c['side_b']} x {c['type']}", flush=True)
        else:
            label = dict(domain=str(c["domain"]), cell_type=c["type"], test="default")
            print(f"\n=== domain {c['domain']} x {c['type']}", flush=True)
        try:
            if "test" in c:
                rows.append(run_designed(c, a, cache))
            else:
                rows.append(dict(**run_contrast(str(c["domain"]), c["type"], a), test="default"))
        except Exception as e:
            # A stratum an agent recommended may be too thin to fit. Record it and
            # go on: one unfittable stratum must not discard the others.
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(dict(**label, error=f"{type(e).__name__}: {e}"))
            rows.append(dict(**label, n_hits=None, error=f"{type(e).__name__}: {e}"))
        rows[-1]["seconds"] = round(time.time() - t0, 1)
        rows[-1]["depth_covariate"] = DEPTH is not None
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    if failed and len(failed) == len(todo):
        sys.exit(f"every contrast failed; first error: {failed[0]['error']}")
    print("\n===== summary =====")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
