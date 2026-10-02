#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
storm_chain_run.py -- prepped h5ads -> ChainGraph -> STORM -> clusters -> triage
================================================================================

One job, one fit. The expensive step (fit_STORM's ADMM with the Jacobi-
preconditioned CG Sylvester solve) runs EXACTLY ONCE; everything after it
operates on frozen shared coordinates and is cheap.

    prepped/*.h5ad
        |
        +-- HVG (batch-aware) + normalise
        +-- ChainGraph      -> L_s   (unbalanced FGW, lot-matched bridges)
        +-- PPI (STRING)    -> L_g   (external; orthogonal to L_s)
        +-- rank            <- split-half congruence, UNREGULARISED PARAFAC2
        |
        +== fit_STORM ==============================  <- the only expensive step
        |
        +-- resolution sweep -> ONE partition at the knee
        +-- cluster triage   -> testable / stage-specific / underpowered
        +-- run record

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
No differential expression. The triage table says which contrast each cluster
supports; running it is a separate step.

    python storm_chain_run.py --prepped ../prepped --out ../storm_out
"""
from __future__ import annotations

import argparse, json, os, sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# gene selection + normalisation
# ===========================================================================

DEFAULT_GENE_ANNOTATION = (Path(__file__).resolve().parents[2]
                           / "reference" / "hgnc_complete_set.txt")


def protein_coding_mask(var_names, annotation) -> np.ndarray:
    """
    Boolean mask over var_names: HGNC locus_group == "protein-coding gene",
    matched on the approved symbol, then on previous and alias symbols (an
    approved symbol always wins over another gene's alias). Clone-named loci
    (AC/AL/AP...), pseudogenes and non-coding RNAs are out of the pool.
    """
    hg = pd.read_csv(annotation, sep="\t", dtype=str,
                     usecols=["symbol", "locus_group", "prev_symbol",
                              "alias_symbol"])
    group = dict(zip(hg.symbol, hg.locus_group))
    for col in ("prev_symbol", "alias_symbol"):
        for syms, grp in zip(hg[col], hg.locus_group):
            if isinstance(syms, str):
                for s in syms.split("|"):
                    group.setdefault(s, grp)
    return np.array([group.get(str(g)) == "protein-coding gene"
                     for g in var_names])


def select_hvgs(adatas, n_top: int, min_frac_samples: float = 0.5,
                seed: int = 0, pool: Optional[np.ndarray] = None,
                tiebreak: str = "median_rank") -> np.ndarray:
    """
    Batch-aware HVG selection: variable in a MAJORITY of slices, not variable
    in the pooled data.

    Selecting on the concatenation picks up genes that vary BECAUSE of batch
    or stage, and those then dominate both the FGW linear cost and the gene
    graph. Requiring replication across slices is the same edge-replication
    logic used for the co-expression graph, applied to gene selection.

    Per slice, seurat_v3 flags its top 2 x n_top genes; a gene is kept if
    flagged in >= min_frac_samples of the slices, and the n_top with the most
    votes are returned. pool (boolean mask over var_names, e.g. from
    protein_coding_mask) restricts the candidates BEFORE ranking, so the
    mean-variance trend is fitted on the pool only. tiebreak orders genes with
    equal vote counts at the n_top cut: "median_rank" = median seurat_v3 rank
    across slices (lower = more variable); "order" = first-seen order, i.e.
    alphabetical (the behaviour before 2026-09-11; reproduces older runs).
    """
    import scanpy as sc
    from collections import Counter
    cnt = Counter()
    ranks = []
    for A in adatas:
        B = A[:, pool].copy() if pool is not None else A.copy()
        try:
            sc.pp.highly_variable_genes(B, flavor="seurat_v3",
                                        n_top_genes=min(n_top * 2, B.n_vars))
            score = B.var["variances_norm"]
        except Exception:
            sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B)
            sc.pp.highly_variable_genes(B, n_top_genes=min(n_top * 2, B.n_vars))
            score = B.var["dispersions_norm"]
        cnt.update(B.var_names[B.var["highly_variable"].values].tolist())
        ranks.append(score.rank(ascending=False, method="first"))
        del B
    need = int(np.ceil(min_frac_samples * len(adatas)))
    keep = [g for g, c in cnt.items() if c >= need]
    if len(keep) > n_top:
        if tiebreak == "median_rank":
            med = pd.concat(ranks, axis=1).median(axis=1)
            keep = sorted(keep, key=lambda g: (-cnt[g], med[g]))[:n_top]
        elif tiebreak == "order":
            keep = [g for g, _ in sorted(((g, cnt[g]) for g in keep),
                                         key=lambda t: -t[1])[:n_top]]
        else:
            raise ValueError(f"unknown tiebreak {tiebreak!r}")
    idx = pd.Index(adatas[0].var_names)
    return np.sort(idx.get_indexer(pd.Index(sorted(keep))))


# ===========================================================================
# gene graph: replicated co-expression, cross-fitted
# ===========================================================================

def fetch_string(genes: pd.Index, cache_dir: str, version: str = "12.0",
                 species: int = 9606, min_score: float = 600.0,
                 chunk: int = 900, timeout: int = 120) -> pd.DataFrame:
    """
    Pull a STRING network for `genes` over HTTPS, caching the result.

    CHUNKING IS NOT OPTIONAL AND IS NOT SAFE NAIVELY. The /network endpoint
    caps identifiers per request, and it returns only edges BETWEEN the
    identifiers you sent -- so splitting 2000 genes into two calls silently
    loses every edge that spans the two halves. We therefore query each chunk
    AND every PAIR of chunks, so every gene pair appears in at least one
    request. With 2000 genes at 900/chunk that is 3 chunks -> 6 calls.

    The resolved edge table is cached under `cache_dir` keyed by version,
    species, score cut and a hash of the gene set, and the hash is written
    into the run record. An unpinned API call would make the run
    irreproducible: STRING versions change, and a rerun months later would
    silently get a different graph.

    Raises on failure. Falling back to an empty graph would leave the gene
    mode unregularised, which is a DIFFERENT MODEL, not a degraded one.
    """
    import hashlib, io, urllib.error, urllib.parse, urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    up = sorted({str(g).upper() for g in genes})
    h = hashlib.sha1(("|".join(up)).encode()).hexdigest()[:12]
    cache = os.path.join(
        cache_dir, f"string_v{version}_sp{species}_s{int(min_score)}_{h}.tsv")
    if os.path.isfile(cache):
        print(f"[ppi] cache hit {os.path.basename(cache)}", flush=True)
        return pd.read_csv(cache, sep="\t")

    base = f"https://version-{version.replace('.', '-')}.string-db.org/api"
    chunks = [up[i:i + chunk] for i in range(0, len(up), chunk)]
    pairs = ([(i,) for i in range(len(chunks))]
             + [(i, j) for i in range(len(chunks))
                for j in range(i + 1, len(chunks))])
    print(f"[ppi] STRING v{version}: {len(up)} genes, {len(chunks)} chunks, "
          f"{len(pairs)} requests", flush=True)

    frames = []
    for n, pr in enumerate(pairs, 1):
        ids = [g for c in pr for g in chunks[c]]
        body = urllib.parse.urlencode({
            "identifiers": "\r".join(ids), "species": species,
            "required_score": int(min_score), "caller_identity": "storm_chain",
        }).encode()
        url = f"{base}/tsv/network"
        # The public API intermittently answers 502/503/504 (seen: a 504 on a
        # 1000-gene request). Those are transient, so retry with backoff before
        # giving up; a 4xx or a parse error is not transient and fails at once.
        txt, last = None, None
        for attempt, wait in enumerate((0, 15, 45, 90), 1):
            if wait:
                print(f"[ppi]   {n}/{len(pairs)}: retry {attempt} in {wait}s "
                      f"({last})", flush=True)
                time.sleep(wait)
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(url, data=body),
                        timeout=timeout) as r:
                    txt = r.read().decode("utf-8", "replace")
                break
            except urllib.error.HTTPError as ex:
                last = f"HTTPError {ex.code}"
                if ex.code < 500:
                    break
            except Exception as ex:               # URLError, socket timeout
                last = f"{type(ex).__name__}: {ex}"
        if txt is None:
            raise RuntimeError(
                f"STRING request {n}/{len(pairs)} failed after retries "
                f"({last}).\nIf the node has no outbound HTTPS, download\n"
                f"  https://stringdb-downloads.org/download/"
                f"protein.links.v{version}/{species}.protein.links.v{version}.txt.gz\n"
                f"plus protein.info, and pass --ppi <path> instead -- or pass "
                f"a cached ../ppi_cache/*.tsv as --ppi.")
        if txt.strip().startswith("Error") or "ErrorMessage" in txt[:200]:
            raise RuntimeError(f"STRING returned an error: {txt[:300]}")
        d = pd.read_csv(io.StringIO(txt), sep="\t")
        if len(d):
            frames.append(d)
        print(f"[ppi]   {n}/{len(pairs)}: {len(d)} edges", flush=True)

    if not frames:
        raise RuntimeError(
            "STRING returned no edges for any chunk. Check that the gene "
            "symbols are human HGNC symbols and that species=9606 is right.")
    out = pd.concat(frames, ignore_index=True)
    # preferredName_A/B are the symbols STRING mapped our input to
    keep = [c for c in ("preferredName_A", "preferredName_B", "score")
            if c in out.columns]
    if len(keep) == 3:
        out = out[keep].rename(columns={"preferredName_A": "protein1",
                                        "preferredName_B": "protein2",
                                        "score": "combined_score"})
        # the API returns score in [0,1]; the file format uses [0,1000]
        if out.combined_score.max() <= 1.0:
            out["combined_score"] = out.combined_score * 1000.0
    out = out.drop_duplicates()
    out.to_csv(cache, sep="\t", index=False)
    print(f"[ppi] cached {len(out)} edges -> {cache}", flush=True)
    return out


def load_ppi(src, genes: pd.Index, topk: int = 20,
             min_score: float = 600.0) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    """
    STRING edge list -> adjacency over `genes`.

    Accepts either a protein-links table (protein1/protein2/combined_score,
    whitespace- or comma-separated) or a bare two/three column edge list.
    Gene symbols are matched case-insensitively; any 9606. prefix is stripped.

    top-k per gene caps degree BEFORE symmetrisation. Raw STRING degree spans
    orders of magnitude, and an unbounded hub would dominate the gene-mode
    smoothing regardless of the Laplacian normalisation applied afterwards.
    """
    if isinstance(src, pd.DataFrame):
        df, path = src, "<STRING API>"
    else:
        path, df = str(src), None
        for kw in (dict(sep=None, engine="python"), dict(sep=","),
                   dict(sep="\t")):
            try:
                df = pd.read_csv(path, **kw)
                if df.shape[1] >= 2:
                    break
            except Exception:
                continue
        if df is None or df.shape[1] < 2:
            raise ValueError(
                f"could not parse a >=2 column edge list from {path}")

    cols = [c.lower() for c in df.columns]
    def pick(*cands, default=None):
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return default
    c1 = pick("protein1", "gene1", "source", "node1", default=df.columns[0])
    c2 = pick("protein2", "gene2", "target", "node2", default=df.columns[1])
    cs = pick("combined_score", "score", "weight")

    a = df[c1].astype(str).str.replace(r"^\d+\.", "", regex=True).str.upper()
    b = df[c2].astype(str).str.replace(r"^\d+\.", "", regex=True).str.upper()
    w = (pd.to_numeric(df[cs], errors="coerce").fillna(0.0)
         if cs is not None else pd.Series(np.ones(len(df))))
    n_raw = len(df)
    if cs is not None:
        keep = w >= min_score
        a, b, w = a[keep], b[keep], w[keep]

    up = pd.Index([g.upper() for g in genes])
    pos = pd.Series(np.arange(len(up)), index=up)
    pos = pos[~pos.index.duplicated()]
    ia = pos.reindex(a).to_numpy()
    ib = pos.reindex(b).to_numpy()
    ok = np.isfinite(ia) & np.isfinite(ib) & (ia != ib)
    ia, ib, w = ia[ok].astype(int), ib[ok].astype(int), np.asarray(w)[ok]

    ng = len(genes)
    A = sp.coo_matrix((w.astype(np.float32), (ia, ib)),
                      shape=(ng, ng)).tocsr()
    A = A.maximum(A.T)

    # top-k per row, then re-symmetrise by union
    R, C, V = [], [], []
    for r in range(ng):
        s0, s1 = A.indptr[r], A.indptr[r + 1]
        if s1 <= s0:
            continue
        d, c = A.data[s0:s1], A.indices[s0:s1]
        if d.size > topk:
            sel = np.argpartition(-d, topk - 1)[:topk]
            d, c = d[sel], c[sel]
        R.append(np.full(d.size, r)); C.append(c); V.append(d)
    if R:
        A = sp.coo_matrix((np.concatenate(V),
                           (np.concatenate(R), np.concatenate(C))),
                          shape=(ng, ng)).tocsr()
        A = A.maximum(A.T)
    A.data[:] = 1.0                     # binary: STRING scores are not weights
    A.setdiag(0.0); A.eliminate_zeros()

    deg = np.asarray(A.sum(1)).ravel()
    info = dict(source=str(path), n_raw_edges=int(n_raw),
                min_score=float(min_score), topk=int(topk),
                n_edges=int(A.nnz // 2),
                mean_degree=float(deg.mean()),
                max_degree=float(deg.max()) if ng else 0.0,
                # genes with no PPI edge are UNREGULARISED in the gene mode.
                # If this is large, L_g only acts on a subset and gamma means
                # something different from what it appears to mean.
                frac_genes_isolated=float((deg == 0).mean()))
    return A, info


def normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    Symmetric-normalised Laplacian.

    Unnormalised D-A lets high-degree genes dominate the smoothing. It also
    bounds lambda_max <= 2, which is what makes a fixed gamma transfer across
    datasets -- and, since the Y-update is solved by preconditioned CG, keeps
    the iteration count stable rather than dataset-dependent.
    """
    d = np.asarray(A.sum(1)).ravel()
    inv = np.zeros_like(d); nz = d > 0
    inv[nz] = 1.0 / np.sqrt(d[nz])
    D = sp.diags(inv)
    return (sp.eye(A.shape[0], format="csr") - D @ A @ D).tocsr()


# ===========================================================================
# rank selection
# ===========================================================================

def tucker_congruence(B1: np.ndarray, B2: np.ndarray) -> float:
    """Mean congruence over greedily matched factor pairs."""
    n1 = B1 / np.maximum(np.linalg.norm(B1, axis=0, keepdims=True), 1e-12)
    n2 = B2 / np.maximum(np.linalg.norm(B2, axis=0, keepdims=True), 1e-12)
    C = np.abs(n1.T @ n2)
    used, out = set(), []
    for _ in range(min(C.shape)):
        i, j = np.unravel_index(np.argmax(C), C.shape)
        if C[i, j] < 0: break
        out.append(float(C[i, j])); used.add((i, j))
        C[i, :] = -1; C[:, j] = -1
    return float(np.mean(out)) if out else 0.0


def stratified_halves(stages: Sequence[str], seed: int = 0):
    """Split SAMPLES into halves, stratified by stage."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for st in sorted(set(stages)):
        idx = [i for i, s in enumerate(stages) if s == st]
        rng.shuffle(idx)
        a += idx[: len(idx) // 2 + len(idx) % 2]
        b += idx[len(idx) // 2 + len(idx) % 2:]
    return sorted(a), sorted(b)


def select_rank(X_list, stages, ranks: Sequence[int], device, dtype,
                iters: int = 15, seed: int = 0) -> Tuple[int, List[Dict]]:
    """
    Split-half congruence on the UNREGULARISED model (gamma=0), which is fast.
    Returns (chosen_rank, curve). The chosen rank is a FLOOR for the
    regularised fit, not the final answer.
    """
    import torch
    from fit_STORM_chunked import fit_STORM
    ia, ib = stratified_halves(stages, seed)
    if not ia or not ib:
        return int(ranks[len(ranks) // 2]), []
    ng = X_list[0].shape[1]
    zeroL = torch.sparse_coo_tensor(
        torch.zeros((2, 1), dtype=torch.long),
        torch.zeros(1, dtype=dtype), (ng, ng)).coalesce().to(device)

    curve = []
    for r in ranks:
        cs = []
        for half in (ia, ib):
            ns = sum(X_list[i].shape[0] for i in half)
            zs = torch.sparse_coo_tensor(
                torch.zeros((2, 1), dtype=torch.long),
                torch.zeros(1, dtype=dtype), (ns, ns)).coalesce().to(device)
            _, _, B, _, _, _ = fit_STORM(
                [X_list[i] for i in half], zeroL, zs, rank=r, gamma=0.0,
                iters=iters, device=device, dtype=dtype, seed=seed,
                nonneg_B=True, beta=0.0, verbose=False, return_info=True)
            cs.append(np.asarray(B.detach().cpu()))
        cong = tucker_congruence(cs[0], cs[1])
        curve.append(dict(rank=int(r), congruence=cong))
        print(f"[rank] r={r:<3d} split-half congruence = {cong:.3f}", flush=True)
    ok = [c for c in curve if c["congruence"] >= 0.80]
    chosen = max(c["rank"] for c in ok) if ok else \
        max(curve, key=lambda c: c["congruence"])["rank"]
    return int(chosen), curve


# ===========================================================================
# clustering: sweep -> ONE partition
# ===========================================================================

def testable(sub: pd.DataFrame, min_donors: int, min_bins: int,
             stages: Sequence[str]) -> bool:
    per = sub.groupby("stage").section.nunique()
    big = sub.groupby(["stage", "section"]).size()
    return bool(len(per) >= 2 and (per >= min_donors).sum() >= 2
                and (big >= min_bins).sum() >= 2 * min_donors)


def resolution_sweep(Z: np.ndarray, meta: pd.DataFrame,
                     resolutions: Sequence[float], n_neighbors: int,
                     min_donors: int, min_bins: int, seed: int = 0,
                     adjacency=None):
    """
    Cluster at several resolutions, pick ONE at the knee of the
    untestable-bin-fraction curve, and freeze it.

    The criterion is TESTABILITY, not stability. Stability answers "is this
    partition reproducible"; it is blind to donor counts, and a perfectly
    stable partition can consist of clusters with one donor per arm. Since the
    endpoint is DE with sample as the unit of replication, what binds is
    whether a cluster can support a contrast at all.

    Carrying several resolutions forward would multiply the DE testing and
    reopen the selective-inference problem, so exactly one survives.
    """
    import scanpy as sc, anndata as ad_mod
    A = ad_mod.AnnData(np.ascontiguousarray(Z))
    A.obs = meta.reset_index(drop=True)
    if adjacency is None:
        sc.pp.neighbors(A, n_neighbors=n_neighbors, use_rep="X",
                        random_state=seed)

    curve, parts = [], {}
    for res in resolutions:
        key = f"leiden_{res:g}"
        sc.tl.leiden(A, resolution=float(res), key_added=key,
                     random_state=seed, flavor="igraph", n_iterations=2,
                     directed=False, adjacency=adjacency)
        lab = A.obs[key].astype(str).values
        m = meta.assign(cl=lab)
        bad = 0
        for cl, sub in m.groupby("cl"):
            if not testable(sub, min_donors, min_bins, sorted(set(meta.stage))):
                bad += len(sub)
        curve.append(dict(resolution=float(res), n_clusters=int(len(set(lab))),
                          untestable_frac=float(bad / len(m))))
        parts[float(res)] = lab
        print(f"[cluster] res={res:<5g} k={curve[-1]['n_clusters']:<4d} "
              f"untestable={curve[-1]['untestable_frac']:.1%}", flush=True)

    # knee: finest resolution whose untestable fraction stays under 10%;
    # fixed rule so the same inputs always give the same partition
    ok = [c for c in curve if c["untestable_frac"] <= 0.10]
    chosen = (max(c["resolution"] for c in ok) if ok else
              min(curve, key=lambda c: c["untestable_frac"])["resolution"])
    return chosen, parts[chosen], curve


def triage(meta: pd.DataFrame, labels: np.ndarray, unmatched: np.ndarray,
           stage_order: Sequence[str], min_donors: int, min_bins: int
           ) -> pd.DataFrame:
    """
    Per cluster: which contrast, if any, does it support?

      testable        both stages, enough donors -> matched cross-stage contrast
      stage_specific  near-pure for one stage AND high unmatched mass ->
                      characterise by markers, NOT against a nearest cluster
                      of a different stage (that conflates the stage effect
                      with the niche difference)
      underpowered    paired but too thin -> report, do not test
    """
    m = meta.assign(cl=labels, unmatched=unmatched)
    rows = []
    for cl, sub in m.groupby("cl"):
        comp = sub.stage.value_counts(normalize=True).to_dict()
        donors = sub.groupby("stage").section.nunique().to_dict()
        top_stage = max(comp, key=comp.get)
        pure = comp[top_stage] > 0.90
        ok = testable(sub, min_donors, min_bins, stage_order)
        cls = ("testable" if ok else
               "stage_specific" if (pure and sub.unmatched.mean() > 0.5)
               else "underpowered")
        r = dict(cluster=str(cl), n_bins=int(len(sub)), classification=cls,
                 dominant_stage=top_stage, purity=float(comp[top_stage]),
                 mean_unmatched=float(sub.unmatched.mean()),
                 min_bins_per_section=int(sub.groupby("section").size().min()))
        for s in stage_order:
            r[f"frac_{s}"] = float(comp.get(s, 0.0))
            r[f"donors_{s}"] = int(donors.get(s, 0))
        if "n_nuclei" in sub.columns:
            r["nuclei_mean"] = float(sub.n_nuclei.mean())
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n_bins", ascending=False)


# ===========================================================================
# graph persistence + memory trace
# ===========================================================================

def _rss(tag: str) -> None:
    """Peak resident set size so far, to locate the host-memory high-water
    mark (the all-12 graph build hit 243 GB against a 250 GB request)."""
    try:
        import resource
        gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        print(f"[mem] peak RSS after {tag}: {gb:.1f} GB", flush=True)
    except Exception:
        pass


def save_graph(d: Path, L_s: sp.spmatrix, rep: Dict[str, Any],
               um: np.ndarray, secs: Sequence[str], stages: Sequence[str],
               ns: Sequence[int]) -> None:
    """
    Persist the spatial Laplacian so the fit (or a refit at another rank /
    gamma / cluster rep) skips the graph build. The key file pins sections,
    stages and bin counts, so a graph is never reused for different inputs.
    """
    d.mkdir(parents=True, exist_ok=True)
    sp.save_npz(d / "L_s.npz", L_s.tocsr())
    np.save(d / "unmatched_per_bin.npy", um)
    (d / "graph_report.json").write_text(json.dumps(rep, indent=2, default=str))
    (d / "graph_key.json").write_text(json.dumps(dict(
        sections=list(secs), stages=list(stages), n_obs=[int(n) for n in ns])))


def save_coupling_arrays(d: Path, niches: Optional[dict],
                         couplings: Optional[list]) -> None:
    """niches/<SEC>.npz + niches/<SEC>_counts.npz and couplings/<SRC>__<DST>.npz."""
    if not niches or not couplings:
        return
    (d / "niches").mkdir(parents=True, exist_ok=True)
    (d / "couplings").mkdir(parents=True, exist_ok=True)
    for s_, n in niches.items():
        dens = n.get("density")
        np.savez_compressed(
            d / "niches" / f"{s_}.npz",
            labels=np.asarray(n["labels"], dtype=np.int32),
            coords=np.asarray(n["coords"], dtype=np.float32),
            mass=np.asarray(n["mass"], dtype=np.float32),
            emb=np.asarray(n["emb"], dtype=np.float32),
            density=(np.asarray(dens, dtype=np.float32) if dens is not None
                     else np.full(len(n["mass"]), np.nan, dtype=np.float32)))
        sp.save_npz(d / "niches" / f"{s_}_counts.npz", sp.csr_matrix(n["counts"]))
    for c in couplings:
        np.savez_compressed(
            d / "couplings" / f"{c['src']}__{c['dst']}.npz",
            G=c["G"], tau=np.float32(c["tau"]), kind=str(c["kind"]),
            unmatched_i=c["unmatched_i"], unmatched_j=c["unmatched_j"],
            concentration=c["concentration"])


def link_coupling_arrays(src: Path, dst: Path) -> None:
    """Point a --load-graph run's out dir at the graph dir's niches/couplings."""
    for name in ("niches", "couplings"):
        if (src / name).is_dir() and not (dst / name).exists():
            os.symlink(os.path.relpath(src / name, dst), dst / name)


def load_graph(d: Path, secs: Sequence[str], stages: Sequence[str],
               ns: Sequence[int]):
    key = json.loads((d / "graph_key.json").read_text())
    want = dict(sections=list(secs), stages=list(stages),
                n_obs=[int(n) for n in ns])
    if key != want:
        raise SystemExit(f"--load-graph {d}: graph was built for {key}, "
                         f"this run has {want}")
    L_s = sp.load_npz(d / "L_s.npz").tocsr()
    rep = json.loads((d / "graph_report.json").read_text())
    rep["loaded_from"] = str(d)
    um = np.load(d / "unmatched_per_bin.npy")
    return L_s, rep, um


# ===========================================================================
# main
# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sections", nargs="*", default=None,
                   help="subset of sections; default all found")
    p.add_argument("--stage-order", nargs="*",
                   default=["NA", "moderate", "severe"])
    p.add_argument("--n-top-genes", type=int, default=2000)
    # HVG candidate pool. 'coding' (default since 2026-09-11): only HGNC
    # protein-coding genes enter the per-slice ranking; with 'all', 84% of the
    # 2,000 HVGs on this cohort were clone-named loci, pseudogenes and ncRNAs
    # (low-count, no curated annotation). 'all' + '--hvg-tiebreak order'
    # reproduces runs made before that date.
    p.add_argument("--gene-pool", choices=["coding", "all"], default="coding")
    p.add_argument("--gene-annotation", default=str(DEFAULT_GENE_ANNOTATION),
                   help="HGNC complete set TSV (symbol, locus_group, "
                        "prev_symbol, alias_symbol); used with --gene-pool coding")
    p.add_argument("--hvg-tiebreak", choices=["median_rank", "order"],
                   default="median_rank",
                   help="ordering of equal-vote genes at the --n-top-genes cut")
    # Default is a STATED CHOICE, not a derived one. select_rank returns
    # max(grid) whenever congruence clears 0.80 anywhere, so on this cohort it
    # reported 40 because the grid stopped at 40 -- and the curve it was
    # reading is non-monotonic (0.860, 0.818, 0.787, 0.806, 0.841, 0.840) over
    # split-halves of 3 and 2 samples, which is too noisy to select on. 10 is
    # where that curve actually peaks. Pass --rank 0 to re-enable selection.
    p.add_argument("--rank", type=int, default=10,
                   help="fixed rank; 0 = select by congruence (grid-edge prone)")
    p.add_argument("--rank-grid", nargs="*", type=int,
                   default=[10, 15, 20, 25, 30, 40])
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=35)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--k-within", type=float, default=10.0,
                   help="target mean degree of the radius graph "
                        "(--within-graph radius only)")
    p.add_argument("--k-cross", type=float, default=5.0,
                   help="literal cross degree; only used when "
                        "--lambda-cross is negative")
    # Within-sample graph. Default reproduces the graph Gong et al. used for
    # layer calling (Stereopy SCC: expression kNN union lattice neighbours),
    # so STORM's within-sample smoothing is the paper's, and the OT couplings
    # add the cross-sample edges the paper handled with Harmony.
    p.add_argument("--within-graph", choices=["scc", "radius"], default="scc")
    p.add_argument("--expr-k", type=int, default=10,
                   help="expression kNN neighbours (Stereopy default 10)")
    p.add_argument("--spatial-neigh", choices=["rook", "queen"],
                   default="rook",
                   help="lattice neighbourhood; rook = squidpy grid n_neighs=6 "
                        "on a square lattice")
    p.add_argument("--n-pcs-graph", type=int, default=50,
                   help="per-sample PCs for the expression kNN")
    p.add_argument("--scale-max", type=float, default=10.0,
                   help="clip after z-scoring, before the graph PCA")
    p.add_argument("--lambda-cross", type=float, default=0.5,
                   help="cross degree as a fraction of the realised mean "
                        "within degree; pass a negative value to use "
                        "--k-cross literally")
    p.add_argument("--cluster-rep", choices=["QHD", "QH"], default="QH",
                   help="QHD = Q_k H D_k (programme shape AND per-sample "
                        "activity, default); QH = Q_k H (shape only)")
    p.add_argument("--gene-chunk-cols", type=int, default=256,
                   help="column budget of the chunked Sylvester solve; halve "
                        "on GPU OOM")
    p.add_argument("--save-graph", default=None,
                   help="directory to save L_s + graph report so later fits "
                        "can --load-graph instead of rebuilding (~3 h)")
    p.add_argument("--graph-only", action="store_true",
                   help="build and save the graph (--save-graph), then exit; fit later "
                        "with --load-graph on a GPU node")
    p.add_argument("--load-graph", default=None,
                   help="directory written by --save-graph; must match the "
                        "sections/stages/bin counts of this run")
    # Chain topology. The default is a HAND-PICKED edge file (JSON list of
    # {"src","dst","kind"} with kind in {"within","cross"}); --edges derive
    # builds it from stage + chip-lot metadata (lot-matched bridges), which
    # only works when a lot column exists. The topology used by a run is
    # always written to <out>/edges.json so it can be copied and edited.
    p.add_argument("--edges", default=None,
                   help="path to an edge JSON file (default: <prepped>/edges.json, "
                        "which must exist), or 'derive' to build lot-matched "
                        "bridges from metadata (requires chip_lot in obs)")
    p.add_argument("--n-niches", type=int, default=800)
    p.add_argument("--ppi", default="string",
                   help="'string' or 'string:12.0' to fetch from the STRING "
                        "API (cached); otherwise a path to an edge list "
                        "(protein1/protein2/combined_score, or any >=2 "
                        "column gene edge list)")
    p.add_argument("--ppi-cache", default="../../cache/ppi_cache",
                   help="where fetched STRING networks are cached")
    p.add_argument("--string-version", default="12.0",
                   help="pinned STRING release; recorded in the run record "
                        "because an unpinned fetch is irreproducible")
    p.add_argument("--species", type=int, default=9606)
    p.add_argument("--ppi-min-score", type=float, default=700.0,
                   help="STRING combined_score cut; 700 = high confidence "
                        "(default since 2026-09-11, with the protein-coding "
                        "gene pool); 600 = medium, the original STORM study's "
                        "default and the cut of runs made before that date")
    p.add_argument("--gene-weight", type=float, default=1.0,
                   help="multiplier on the gene Laplacian (effective gene "
                        "gamma = gamma x gene-weight); spatial gamma unchanged")
    p.add_argument("--gene-topk", type=int, default=20,
                   help="max PPI partners per gene, applied before "
                        "symmetrisation to cap hub degree")
    p.add_argument("--resolutions", nargs="*", type=float,
                   default=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    p.add_argument("--n-neighbors", type=int, default=30,
                   help="kNN size for --cluster-graph knn")
    # Clustering graph for the partition. The default reproduces what worked
    # on the AD cohort: Leiden on the EXACT expression kNN (k = --expr-k) of
    # the embedding UNIONED with the tile lattice -- the same graph the
    # paper's SCC clusters on. A plain kNN Leiden (the previous default) on
    # the same embedding gave coherence 0.43 / ARI 0.15 vs the SCC layers;
    # the union graph gave 0.73 / 0.36 and laminar domains shared by all 12
    # sections. 'knn' keeps the old behaviour for comparison.
    p.add_argument("--cluster-graph", choices=["scc", "knn"], default="scc")
    p.add_argument("--min-donors", type=int, default=2)
    p.add_argument("--min-bins", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="the CG Y-step holds several (n_cells, n_genes) dense "
                        "tensors concurrently -- float64 on the full 12-sample "
                        "cell graph OOMs an 80GB A100; float32 halves that")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import torch, scanpy as sc, anndata as ad_mod
    from ChainGraph import ChainConfig, build_chain_graph
    from SCCGraph import per_sample_pca
    from STORM import STORM, _scipy_to_torch_sparse
    from Utils.TensorConstructionUtils import build_irregular_slices

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec: Dict[str, Any] = {"args": vars(a), "started": time.ctime()}
    if a.save_graph and a.load_graph:
        sys.exit("--save-graph and --load-graph are mutually exclusive")
    if a.graph_only and not a.save_graph:
        sys.exit("--graph-only needs --save-graph (the graph is the only output)")

    # ---- load ----------------------------------------------------------
    files = sorted(Path(a.prepped).glob("*_prepped.h5ad"))
    if a.sections:
        files = [f for f in files if f.name.split("_prepped")[0] in a.sections]
    if not files:
        sys.exit(f"no prepped h5ads in {a.prepped}")
    adatas = [ad_mod.read_h5ad(f) for f in files]
    secs = [str(A.obs["section"].iloc[0]) for A in adatas]
    stages = [str(A.obs["stage"].iloc[0]) for A in adatas]
    print(f"[load] {len(adatas)} slices {list(zip(secs, stages))} "
          f"({time.time()-t0:.0f}s)", flush=True)
    _rss("load")

    # ---- genes ---------------------------------------------------------
    pool = None
    rec["gene_pool"] = dict(pool=a.gene_pool, n_all=int(adatas[0].n_vars),
                            hvg_tiebreak=a.hvg_tiebreak)
    if a.gene_pool == "coding":
        import hashlib as _hl
        pool = protein_coding_mask(adatas[0].var_names, a.gene_annotation)
        rec["gene_pool"].update(
            n_pool=int(pool.sum()), annotation=a.gene_annotation,
            annotation_sha1=_hl.sha1(Path(a.gene_annotation).read_bytes()).hexdigest())
        print(f"[genes] pool: {int(pool.sum())} protein-coding of "
              f"{adatas[0].n_vars} genes ({a.gene_annotation})", flush=True)
    gi = select_hvgs(adatas, a.n_top_genes, seed=a.seed, pool=pool,
                     tiebreak=a.hvg_tiebreak)
    adatas = [A[:, gi].copy() for A in adatas]
    print(f"[genes] {len(gi)} HVGs (replicated across slices; ties by "
          f"{a.hvg_tiebreak})", flush=True)

    # ---- gene graph: STRING PPI (external, orthogonal to L_s) ----------
    if str(a.ppi).lower().startswith("string"):
        ver = (str(a.ppi).split(":", 1)[1] if ":" in str(a.ppi)
               else a.string_version)
        src = fetch_string(adatas[0].var_names,
                           cache_dir=a.ppi_cache, version=ver,
                           species=a.species, min_score=a.ppi_min_score)
    else:
        src = a.ppi
    A_g, ppi_info = load_ppi(src, adatas[0].var_names,
                             topk=a.gene_topk, min_score=a.ppi_min_score)
    # hash the resolved adjacency so two runs can be compared field by field
    import hashlib as _hl
    ppi_info["adjacency_sha1"] = _hl.sha1(
        np.ascontiguousarray(A_g.indices).tobytes()
        + np.ascontiguousarray(A_g.indptr).tobytes()).hexdigest()[:16]
    L_g = normalized_laplacian(A_g)
    if a.gene_weight != 1.0:
        # separate strength for the gene mode: the fitter applies ONE gamma to
        # L_s (+) L_g, so scaling L_g here is the only way to strengthen gene
        # smoothing without also changing spatial smoothing
        L_g = (L_g * float(a.gene_weight)).tocsr()
    print(f"[genes] PPI: {ppi_info['n_edges']} edges, mean degree "
          f"{ppi_info['mean_degree']:.1f}, "
          f"{ppi_info['frac_genes_isolated']:.1%} genes isolated "
          f"({time.time()-t0:.0f}s)", flush=True)
    if ppi_info["frac_genes_isolated"] > 0.5:
        print("[genes] WARN over half the HVGs have no PPI edge: L_g acts on "
              "a minority of the gene mode, so gamma does not mean what it "
              "appears to mean. Check symbol mapping before trusting it.",
              flush=True)
    rec["gene_graph"] = ppi_info

    # ---- normalise for the fit -----------------------------------------
    for A in adatas:
        sc.pp.normalize_total(A, target_sum=1e4)
        sc.pp.log1p(A)

    # ---- per-sample embedding for the SCC expression kNN ----------------
    # Stereopy default: scale (clip 10) -> PCA, computed on a copy so A.X --
    # the matrix STORM fits -- stays the sparse log-normalised HVG matrix.
    # Per sample, because within-sample edges must not depend on batch
    # alignment; the cross-sample edges are the OT couplings.
    if a.within_graph == "scc" and not a.load_graph:
        rec["graph_pca"] = {
            s: per_sample_pca(A, n_pcs=a.n_pcs_graph, scale_max=a.scale_max,
                              seed=a.seed)
            for s, A in zip(secs, adatas)}
        print(f"[graph] per-sample PCA({a.n_pcs_graph}) for the expression "
              f"kNN ({time.time()-t0:.0f}s)", flush=True)

    # ---- spatial graph --------------------------------------------------
    ns_all = [int(A.n_obs) for A in adatas]
    cfg = ChainConfig(k_within=a.k_within, k_cross=a.k_cross,
                      n_niches_per_sample=a.n_niches,
                      stage_order=tuple(a.stage_order), seed=a.seed,
                      within_graph=a.within_graph, expr_key="X_pca",
                      expr_k=a.expr_k, spatial_neigh=a.spatial_neigh,
                      lambda_cross=(a.lambda_cross if a.lambda_cross >= 0
                                    else None),
                      verbose=False)
    edges = None
    if a.edges is None:
        a.edges = str(Path(a.prepped) / "edges.json")
        if not Path(a.edges).is_file():
            sys.exit(f"no topology given: write {a.edges} (JSON list of "
                     f"{{src, dst, kind}} with kind in within|cross), "
                     f"or pass --edges derive to derive lot-matched bridges from metadata")
    if a.edges != "derive":
        from ChainGraph import Edge
        spec = json.loads(Path(a.edges).read_text())
        known = set(secs)
        for e in spec:
            if e["src"] not in known or e["dst"] not in known:
                sys.exit(f"--edges: {e} names a section not in this run {sorted(known)}")
            if e.get("kind", "within") not in ("within", "cross"):
                sys.exit(f"--edges: kind must be within|cross, got {e}")
        edges = [Edge(src=e["src"], dst=e["dst"], kind=e.get("kind", "within"),
                      lot_matched=bool(e.get("lot_matched", True)), note=e.get("note", ""))
                 for e in spec]
        print(f"[chain] hand-picked topology from {a.edges}: {len(edges)} edges "
              f"({sum(e.kind == 'cross' for e in edges)} cross-stage)", flush=True)
    elif "chip_lot" not in adatas[0].obs.columns:
        sys.exit("--edges derive needs a chip_lot column in obs; pass an edge "
                 "file instead (see <run>/edges.json of a previous run)")
    if a.load_graph:
        L_s, graph_rep, um = load_graph(Path(a.load_graph), secs, stages,
                                        ns_all)
        link_coupling_arrays(Path(a.load_graph), out)
        print(f"[graph] loaded L_s from {a.load_graph}", flush=True)
    else:
        L_s, graph_rep = build_chain_graph(adatas, cfg, edges=edges)
        # per-bin unmatched mass (max over the edges each slice takes part
        # in): popped from the report so the JSON stays small, saved as .npy
        um_map = graph_rep.pop("unmatched_per_bin", None) or {}
        um = np.concatenate([
            np.asarray(um_map.get(s, np.zeros(n)), dtype=np.float32)
            for s, n in zip(secs, ns_all)])
        niches_arr = graph_rep.pop("_niches", None)
        coup_arr = graph_rep.pop("_couplings", None)
        save_coupling_arrays(out, niches_arr, coup_arr)
        if a.save_graph:
            save_graph(Path(a.save_graph), L_s, graph_rep, um, secs, stages,
                       ns_all)
            save_coupling_arrays(Path(a.save_graph), niches_arr, coup_arr)
            print(f"[graph] saved to {a.save_graph} (+ niches/, couplings/)",
                  flush=True)
        del niches_arr, coup_arr
    rec["graph"] = graph_rep
    (out / "edges.json").write_text(json.dumps(
        [dict(src=e["src"], dst=e["dst"], kind=e["kind"], lot_matched=e.get("lot_matched", True),
              note=e.get("note", "")) for e in graph_rep["edges"]], indent=1))
    np.save(out / "unmatched_per_bin.npy", um)
    (out / "graph_report.json").write_text(
        json.dumps(graph_rep, indent=2, default=str))
    print(f"[graph] L_s {L_s.shape} nnz={L_s.nnz} ({time.time()-t0:.0f}s)",
          flush=True)
    _rss("graph")
    if a.graph_only:
        # The graph build is CPU-only (per-sample PCA, kNN, FGW couplings) and
        # takes hours; the fit needs a GPU for minutes. Splitting them lets the
        # build run on a CPU node and the fit reload it (--load-graph), which
        # reproduces a single-job fit exactly.
        print(f"[graph] --graph-only: saved to {a.save_graph}; stopping before the fit",
              flush=True)
        return

    # ---- tensor ---------------------------------------------------------
    device = torch.device(a.device if torch.cuda.is_available()
                          or a.device == "cpu" else "cpu")
    dtype = torch.float32 if a.dtype == "float32" else torch.float64
    X_list, ns_list, _ = build_irregular_slices(adatas, device=device,
                                                dtype=dtype)
    _rss("tensor")

    # ---- rank -----------------------------------------------------------
    if a.rank > 0:
        rank, curve = a.rank, []
    else:
        rank, curve = select_rank(X_list, stages, a.rank_grid, device, dtype,
                                  seed=a.seed)
        print(f"[rank] chosen {rank} (FLOOR: regularisation supports >= this)",
              flush=True)
    rec["rank"] = dict(chosen=int(rank), curve=curve,
                       note="unregularised split-half congruence; a floor")

    # ---- fit (ONCE) ------------------------------------------------------
    model = STORM(device=device, dtype=dtype, seed=a.seed)
    model.adatas_aligned = adatas
    model.X_list, model.ns_list = X_list, ns_list
    model.Ls_torch = _scipy_to_torch_sparse(L_s, device, dtype)
    model.Lg_torch = _scipy_to_torch_sparse(sp.coo_matrix(L_g), device, dtype)
    print(f"[fit] rank={rank} gamma={a.gamma} iters={a.iters} ...", flush=True)
    model.fit(rank=rank, gamma=a.gamma, iters=a.iters, beta=a.beta,
              nonneg_B=True, gene_chunk_cols=a.gene_chunk_cols, verbose=True)
    model.attach_embeddings(key="X_STORM", also_store_shape=True)
    print(f"[fit] done ({time.time()-t0:.0f}s)", flush=True)
    _rss("fit")

    # Representation to cluster (--cluster-rep):
    #   QHD : Q_k H D_k = obsm["X_STORM"] (default). D_k = diag(w_k) is the
    #         per-sample amplitude of each gene programme, so domains are
    #         defined by which programmes are active AND how active they are
    #         in that sample -- a domain that is the same programme mix at
    #         different intensity in AD vs control is a real difference this
    #         representation can see. The price: D_k can also carry sample-
    #         level scale, so the per-domain max_section / stage purity in
    #         the triage table is the check that batch structure did not leak
    #         into the partition.
    #   QH  : Q_k H = obsm["X_STORM_shape"]. Q_k^T Q_k = I (Procrustes), so
    #         (Q_k H)^T (Q_k H) = H^T H is identical for every slice: the
    #         amplitude-free "programme shape" space. Kept for comparison.
    # Both are saved, plus w (S x R), so either can be re-clustered without a
    # refit (STORM/sweep_resolution.py --rep).
    Z_qhd = np.vstack([np.asarray(A.obsm["X_STORM"]) for A in adatas])
    Z_qh = np.vstack([np.asarray(A.obsm["X_STORM_shape"]) for A in adatas])
    Z = Z_qhd if a.cluster_rep == "QHD" else Z_qh
    rec["cluster_rep"] = a.cluster_rep
    meta = pd.DataFrame(dict(
        section=np.concatenate([[s] * A.n_obs for s, A in zip(secs, adatas)]),
        stage=np.concatenate([[s] * A.n_obs for s, A in zip(stages, adatas)]),
        lot=np.concatenate([[str(A.obs["chip_lot"].iloc[0])] * A.n_obs
                            for A in adatas])))
    if all("n_nuclei" in A.obs for A in adatas):
        meta["n_nuclei"] = np.concatenate(
            [np.asarray(A.obs["n_nuclei"]) for A in adatas])

    # Bin coordinates. Without these the outputs cannot be plotted on tissue,
    # scored for spatial coherence, or compared to a layer figure -- i.e. none
    # of the checks a spatial-domain result actually needs. They are per-slice
    # raw coordinates, so use them WITHIN a section; they are not registered
    # across sections.
    if all("spatial" in A.obsm for A in adatas):
        xy = np.vstack([np.asarray(A.obsm["spatial"])[:, :2] for A in adatas])
        meta["x"], meta["y"] = xy[:, 0], xy[:, 1]
    else:
        print("[warn] no obsm['spatial']; bin_meta.csv will have no "
              "coordinates and no spatial QC will be possible", flush=True)

    np.save(out / "shared_coords.npy", Z)          # what was clustered
    np.save(out / "shared_coords_QHD.npy", Z_qhd)
    np.save(out / "shared_coords_QH.npy", Z_qh)
    np.save(out / "w.npy", np.vstack([np.asarray(w.detach().cpu()).reshape(-1)
                                      for w in model.w_list]))
    np.save(out / "B_gene_loadings.npy",
            np.asarray(model.B.detach().cpu()))
    pd.Series(list(adatas[0].var_names), name="gene").to_csv(
        out / "genes.csv", index=False)
    meta.to_csv(out / "bin_meta.csv", index=False)

    # ---- clustering ------------------------------------------------------
    adjacency, gdeg = None, None
    if a.cluster_graph == "scc":
        from SCCGraph import embedding_lattice_graph
        adjacency, gdeg = embedding_lattice_graph(
            Z, meta.section, meta.x, meta.y, k=a.expr_k,
            neigh=a.spatial_neigh)
        print(f"[cluster] graph: exact kNN(k={a.expr_k}) U lattice; mean "
              f"degree expr {gdeg['expr']:.1f} lattice {gdeg['lattice']:.1f} "
              f"union {gdeg['union']:.1f}", flush=True)
    res, labels, sweep = resolution_sweep(
        Z, meta, a.resolutions, a.n_neighbors, a.min_donors, a.min_bins,
        seed=a.seed, adjacency=adjacency)
    print(f"[cluster] chosen resolution {res:g}", flush=True)
    rec["clustering"] = dict(sweep=sweep, chosen_resolution=float(res),
                             cluster_graph=a.cluster_graph, graph_degrees=gdeg,
                             rule="finest resolution with untestable "
                                  "fraction <= 10%")

    tri = triage(meta, labels, um if len(um) == len(meta) else
                 np.zeros(len(meta)), a.stage_order, a.min_donors, a.min_bins)
    tri.to_csv(out / "cluster_triage.csv", index=False)
    pd.DataFrame(dict(cluster=labels)).to_csv(out / "bin_clusters.csv",
                                              index=False)
    print("\n" + tri.head(30).to_string(index=False))
    print("\n--- triage ---")
    print(tri.classification.value_counts().to_string())

    rec["elapsed_s"] = round(time.time() - t0, 1)
    (out / "run_record.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"\n[done] {rec['elapsed_s']}s -> {out}")
    print("  shared_coords.npy (clustered rep)  shared_coords_QHD.npy  "
          "shared_coords_QH.npy  w.npy")
    print("  B_gene_loadings.npy  genes.csv  bin_meta.csv  "
          "unmatched_per_bin.npy")
    print("  bin_clusters.csv   cluster_triage.csv")
    print("  graph_report.json  run_record.json")


if __name__ == "__main__":
    main()