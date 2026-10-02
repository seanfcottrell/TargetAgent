#!/usr/bin/env python
"""
Plots of one pipeline run's outputs: spatial domains, factor matrices, usage
maps, differential expression and the PPI networks.

    python make_figures.py --run runs/<dataset>/<run>
    python make_figures.py --run runs/<dataset>/<run> --only domains programmes
    python make_figures.py --run runs/<dataset>/<run> --only usage --programmes F1 F3

Every plot is its own file (PNG and PDF); a legend shared by sibling plots is
written once as its own file (`*_legend`). Everything drawn is read from files
the run wrote, located through `<run>/report/manifest.json` (run_all.py writes
it), so nothing here is specific to a dataset. A group whose inputs the run has
not produced yet is skipped.

    qc            genes and transcripts per nucleus; nuclei retained per section
    domains       the spatial domains in each section; composition of each domain
    programmes    the factor matrices: gene loadings (B), enriched terms per
                  programme, per-section amplitude (D_k)
    usage         each programme's usage (Q_k H) drawn on the tissue, per section
    deg           one volcano per differential-expression contrast
    networks      each PPI network the sheaf ranked, and its spectral shift
                  against weighted degree

Run it in `deg-calib`: the network plot imports the sheaf's own graph builder,
which needs gudhi.
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
from style import (CAT, GROUP, INK, MUTED, GRID, SURFACE, SEQ, DIV, FS,   # noqa: E402
                   cap, label, tidy, legend, save)
from paths import load_yaml                                               # noqa: E402

PANEL = 3.4          # default width of one plot, inches


def rd(path, **kw):
    """Every table in this project keeps 'NA' as a literal label."""
    return pd.read_csv(path, keep_default_na=False, **kw)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def jl(path):
    return json.load(open(path)) if os.path.isfile(path) else None


class Run:
    """One run, addressed through its manifest and its data card."""

    def __init__(self, run_dir):
        self.dir = os.path.abspath(run_dir)
        self.M = json.load(open(os.path.join(self.dir, "report", "manifest.json")))
        self.label = os.path.basename(self.dir)
        card = load_yaml(self.M["card"])
        cond = card["condition"]
        self.case, self.ctrl = cond["case_group"], cond["control_group"]
        self.colour = {self.case: GROUP["case"], self.ctrl: GROUP["control"]}
        self.types = list(card["priors"]["cell_types"])
        self.sections = [str(s["section"]) for s in card["samples"]]
        self.group_of = {str(s["section"]): self.ctrl if str(s["stage"]) == str(cond["control_label"])
                         else self.case for s in card["samples"]}
        self.res = json.load(open(self.M["resolution_decision"]))["resolution_str"]
        m = os.path.join(self.dir, "methods.yaml")         # the methods this run used
        self.methods = load_yaml(m) if os.path.isfile(m) else {}
        deg = self.methods.get("deg", {})
        self.hit_fdr = float(deg.get("hit_fdr", 0.05))
        self.hit_lfc = float(deg.get("hit_abs_log2fc", 1.0))

    def fit(self, *p):
        return os.path.join(self.M["fit"], *p)

    def path(self, key, *p):
        return os.path.join(self.M[key], *p)

    def tiles(self):
        meta = rd(self.fit("bin_meta.csv"), dtype={"section": str})
        cl = rd(self.fit(f"bin_clusters_QH_sccg_res{self.res}.csv"), dtype={"cluster": str})
        meta["domain"] = cl.cluster.values
        return meta

    def networks(self):
        """[(domain, programme, residual genes)]: one PPI network per programme
        per domain, as the sheaf stage built them."""
        prov = jl(self.path("sheaf", "charges", "provenance.json")) or {"domains": {}}
        resid = jl(self.path("sheaf", "out_rank", "residual_per_scale.json")) or {}
        return [(str(d), str(pr), list(resid.get(d, {}).get(pr, {}).get("genes", [])))
                for d, v in prov["domains"].items() for pr in v.get("networks", {})]

    def contrast_label(self, row):
        dom = str(row.get("domain", "") or "")
        cid = str(row.get("contrast_id", "") or "")
        ct = str(row.get("cell_type", "") or "")
        ct = "All Cells" if ct == "all" else ct
        base = f"D{dom} {ct}" if dom else f"{cid} {ct}"
        test = str(row.get("test", "") or "")
        return base + ("" if test in ("", "default") else f" ({cap(test)})")

    def deg(self):
        """Every contrast's table, labelled, plus which contrasts were designed
        (their estimate is the contrast's own, not case against control)."""
        summ, designed = {}, set()
        s = self.path("deg", "summary.csv")
        if os.path.isfile(s):
            for r in rd(s).to_dict("records"):
                if str(r.get("domain", "")):
                    summ[f"domain{r['domain']}_{r['cell_type']}"] = self.contrast_label(r)
                else:
                    summ[f"contrast_{r['contrast_id']}"] = self.contrast_label(r)
                    designed.add(self.contrast_label(r))
        rows = []
        for f in sorted(glob.glob(self.path("deg", "deg_*.csv"))):
            stem = os.path.basename(f)[len("deg_"):-len(".csv")]
            t = pd.read_csv(f)
            t["contrast"] = summ.get(stem, stem)
            rows.append(t)
        if not rows:
            return pd.DataFrame(columns=["gene", "padj", "log2FC", "hit", "contrast"]), designed
        return pd.concat(rows, ignore_index=True), designed


def newfig(w=PANEL, h=2.5):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def slug(s):
    out = "".join(c.lower() if c.isalnum() else "_" for c in str(s))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def legend_file(out, name, handles, ncol=None, w=PANEL, h=0.45):
    """A legend on its own, for an encoding shared by several plots."""
    fig = plt.figure(figsize=(w, h))
    fig.legend(handles=handles, loc="center", ncol=ncol or len(handles), frameon=False)
    save(fig, out, name)


def violin(ax, data, positions, colours, width=0.8):
    """One violin per group, median marked, no text."""
    parts = ax.violinplot(data, positions=positions, widths=width,
                          showextrema=False, showmedians=True)
    for body, c in zip(parts["bodies"], colours):
        body.set(facecolor=c, edgecolor=c, alpha=0.55, linewidth=0.7)
    parts["cmedians"].set(color=INK, linewidth=1.0)
    return parts


# ============================================================ nuclei quality
def qc(run, out, **_):
    cells = rd(run.path("celltypes_qc", "cells.csv.gz"),
               usecols=["section", "n_counts", "n_genes"], dtype={"section": str})
    cells["n_genes"], cells["n_counts"] = num(cells.n_genes), num(cells.n_counts)
    cells["group"] = cells.section.map(run.group_of)
    groups = [run.case, run.ctrl]
    cols = [run.colour[g] for g in groups]

    for col, ylab, name in [("n_genes", "genes per nucleus", "qc_genes_per_nucleus"),
                            ("n_counts", "transcripts per nucleus", "qc_transcripts_per_nucleus")]:
        fig, ax = newfig(2.5, 2.5)
        violin(ax, [np.log10(cells.loc[cells.group == g, col].dropna().values) for g in groups],
               [0, 1], cols)
        ax.set_xticks([0, 1]); ax.set_xticklabels(groups)
        ax.set_ylabel(cap(ylab) + " (Log10)")
        tidy(ax)
        save(fig, out, name)

    coh = rd(run.path("celltypes_qc", "composition_by_section.csv"), index_col=0)
    coh.index = coh.index.astype(str)
    coh = coh.reindex(sorted(run.sections, key=lambda s: (run.group_of[s], s)))
    grp = [run.group_of[s] for s in coh.index]
    fig, ax = newfig(4.0, 2.7)
    x = np.arange(len(coh))
    ax.bar(x, num(coh.n_all) / 1e3, color=GRID, width=0.72)
    ax.bar(x, num(coh.n_kept) / 1e3, color=[run.colour[g] for g in grp], width=0.72)
    ax.set_xticks(x); ax.set_xticklabels(coh.index, rotation=60, ha="right")
    ax.set_ylabel(cap("nuclei (thousands)"))
    tidy(ax)
    legend(ax, [Patch(color=GRID, label=cap("segmented, before QC")),
                Patch(color=run.colour[run.case], label=f"{run.case}, " + cap("passing QC")),
                Patch(color=run.colour[run.ctrl], label=f"{run.ctrl}, " + cap("passing QC"))],
           loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=1)
    save(fig, out, "qc_nuclei_retained")


# ============================================================ spatial domains
def domains(run, out, **_):
    tiles = run.tiles()
    doms = [str(d) for d in tiles.domain.value_counts().index]      # largest first
    dcol = {d: CAT[i % len(CAT)] for i, d in enumerate(doms)}

    for s in run.sections:
        fig, ax = newfig(2.0, 2.1)
        t = tiles[tiles.section == s]
        ax.scatter(t.x, t.y, c=[dcol.get(d, MUTED) for d in t.domain], s=0.9,
                   marker="s", lw=0, rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(s, fontsize=FS, pad=2)
        save(fig, out, f"domains_map_{slug(s)}")
    legend_file(out, "domains_map_legend",
                [Line2D([], [], marker="s", ls="", color=dcol[d], ms=5,
                        label=cap(f"domain {d}")) for d in doms],
                ncol=min(len(doms), 4), w=PANEL, h=0.8)

    cells = rd(run.path("composition", "cells_with_domain.csv.gz"),
               usecols=["domain", "type_final"], dtype=str)
    ct = pd.crosstab(cells.domain, cells.type_final, normalize="index")
    order = [d for d in doms if d in ct.index] + [d for d in ct.index if d not in doms]
    ct = ct.reindex(index=order, columns=[t for t in run.types if t in ct.columns])
    tcol = {t: CAT[i % len(CAT)] for i, t in enumerate(run.types)}
    fig, ax = newfig(4.2, 0.30 * len(ct) + 1.5)
    left = np.zeros(len(ct))
    for t in ct.columns:
        ax.barh(range(len(ct)), ct[t], left=left, color=tcol[t], height=0.66,
                edgecolor=SURFACE, lw=0.8, label=t)
        left += ct[t].values
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels([cap(f"domain {d}") for d in ct.index])
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    label(ax, x="fraction of typed nuclei")
    legend(ax, ncol=min(len(ct.columns), 4), loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, out, "domains_composition")


# ============================================================ factor matrices
def _parse_top(cell):
    g, v = str(cell).rsplit(" (", 1)
    return g, float(v.rstrip(")"))


def programmes(run, out, n_genes=10, n_terms=8, programmes=None, **_):
    pa = run.M["progact"]
    top = rd(os.path.join(pa, "program_top_genes.csv"))
    w = rd(os.path.join(pa, "w_by_sample.csv"), index_col=0)
    B = np.load(run.fit("B_gene_loadings.npy"))
    genes = rd(run.fit("genes.csv")).gene.tolist()
    gidx = {g: i for i, g in enumerate(genes)}
    progs = list(top.columns)
    chosen = [p for p in progs if not programmes or p in programmes]

    # --- B: the top genes of each chosen programme, against every programme, so
    #     a gene loading on several programmes is visible as such
    rows = []
    for p in chosen:
        for cell in top[p].head(n_genes):
            g, _ = _parse_top(cell)
            if g in gidx and g not in [r[0] for r in rows]:
                rows.append((g, p))
    L = np.array([B[gidx[g]] for g, _ in rows])
    # Each gene against its own largest loading, so the heatmap answers "which
    # programmes does this gene belong to"; scaling by programme instead washes
    # every row out to the middle of the ramp.
    L = L / np.maximum(L.max(1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(0.30 * len(progs) + 1.6, 0.155 * len(rows) + 0.9))
    im = ax.imshow(L, cmap=SEQ, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(progs))); ax.set_xticklabels(progs)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([g for g, _ in rows])
    for i, (_, p) in enumerate(rows):                  # ring the programme it was taken from
        ax.add_patch(plt.Rectangle((progs.index(p) - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=CAT[1], lw=0.8))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, shrink=0.55)
    cb.set_label(cap("loading, relative to the gene's largest"))
    cb.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, out, "programmes_loadings")

    # --- enriched terms per programme
    enr_path = os.path.join(pa, "program_enrichment.csv")
    if os.path.isfile(enr_path):
        import textwrap
        enr = rd(enr_path)
        enr["fdr"] = num(enr.fdr)
        for p in chosen:
            e = enr[enr.program == p].sort_values("fdr").copy()
            e["_key"] = e.term.map(lambda t: t.split(" (")[0].lower())      # one term in two
            e = e.drop_duplicates("_key").head(n_terms)                     # libraries is one term
            if not len(e):
                continue
            fig, ax = newfig(4.4, 0.26 * len(e) + 1.0)
            y = np.arange(len(e))[::-1]
            ax.barh(y, -np.log10(e.fdr.clip(lower=1e-30)), color=CAT[0], height=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(["\n".join(textwrap.wrap(t.split(" (")[0], 34)[:2]) for t in e.term])
            ax.axvline(-np.log10(0.05), color=MUTED, lw=0.8)
            label(ax, x="-log10 FDR")
            tidy(ax, "x")
            save(fig, out, f"programmes_pathways_{slug(p)}")

    # --- the diagonal D_k: each section's amplitude for each programme
    fig, ax = newfig(5.2, 2.6)
    cols = [c for c in progs if c in w.columns]
    w.index = w.index.astype(str)
    for g in (run.case, run.ctrl):
        sel = [s for s in w.index if run.group_of.get(s) == g]
        off = -0.16 if g == run.case else 0.16
        for j, p in enumerate(cols):
            v = num(w.loc[sel, p]).values
            ax.scatter(np.full(len(v), j + off), v, s=16, color=run.colour[g], zorder=3,
                       edgecolor=SURFACE, lw=0.4, label=g if j == 0 else None)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    label(ax, x="programme", y="amplitude (D$_k$)")
    tidy(ax)
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    save(fig, out, "programmes_amplitude")


# ============================================================ programme usage in situ
def usage(run, out, lo_q=5.0, hi_q=95.0, programmes=None, **_):
    """Q_k H column by column, drawn on the tissue: where each programme is used.

    Each programme gets its own colour scale, fixed across sections. The columns
    are not comparable in magnitude between programmes, since the PARAFAC2
    constraint fixes the usage Gram matrix rather than the per-programme scale,
    so a shared scale would flatten every programme except the largest."""
    QH = np.load(run.fit("shared_coords_QH.npy"))
    meta = rd(run.fit("bin_meta.csv"), dtype={"section": str})
    if len(meta) != QH.shape[0]:
        raise SystemExit(f"bin_meta has {len(meta)} rows for {QH.shape[0]} usage rows")
    meta["x"], meta["y"] = num(meta.x), num(meta.y)
    progs = [f"F{r}" for r in range(QH.shape[1])]

    for j, p in enumerate(progs):
        if programmes and p not in programmes:
            continue
        v = QH[:, j]
        lo, hi = (float(x) for x in np.percentile(v, [lo_q, hi_q]))
        for sec in run.sections:
            m = (meta.section == sec).to_numpy()
            fig, ax = newfig(2.0, 2.1)
            ax.scatter(meta.x[m], meta.y[m], c=v[m], cmap=SEQ, vmin=lo, vmax=hi,
                       s=0.9, marker="s", lw=0, rasterized=True)
            ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(f"{p}, {sec}", fontsize=FS, pad=2)
            save(fig, out, f"usage_{slug(p)}_{slug(sec)}")

        fig, cax = plt.subplots(figsize=(2.0, 0.42))
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(lo, hi)),
                          cax=cax, orientation="horizontal")
        cb.set_label(cap(f"usage of {p}")); cb.outline.set_visible(False)
        save(fig, out, f"usage_{slug(p)}_colorbar")


# ============================================================ differential expression
def _stack_labels(ax, pts, step=0.085):
    """Gene names stacked down from the top of the plot, on the side the point
    sits, each joined to its point by a leader line."""
    for side, sel in (("left", [p for p in pts if p[0] < 0]),
                      ("right", [p for p in pts if p[0] >= 0])):
        sel = sorted(sel, key=lambda p: -p[1])
        xa = 0.02 if side == "left" else 0.98
        for i, (x, y, name) in enumerate(sel):
            ax.annotate(name, xy=(x, y), xycoords="data",
                        xytext=(xa, 0.97 - step * i), textcoords="axes fraction",
                        ha="left" if side == "left" else "right", va="center",
                        fontsize=FS, color=INK,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.5,
                                        shrinkA=0, shrinkB=2))


def deg(run, out, max_labels=7, **_):
    tab, designed = run.deg()
    if tab.empty:
        print("  no DEG tables; skipping"); return
    hitcol = CAT[1]
    for c in dict.fromkeys(tab.contrast):
        t = tab[tab.contrast == c].copy()
        t["y"] = -np.log10(num(t.padj).clip(lower=1e-300))
        hit = t.hit.astype(str).str.lower().isin(["true", "1"])
        fig, ax = newfig(2.9, 2.6)
        ax.scatter(t.log2FC[~hit], t.y[~hit], s=4, color=MUTED, alpha=0.45, lw=0,
                   rasterized=True)
        ax.scatter(t.log2FC[hit], t.y[hit], s=13, color=hitcol, lw=0.3,
                   edgecolor=SURFACE, zorder=3)
        _stack_labels(ax, [(r.log2FC, r.y, r.gene) for r in
                           t[hit].nlargest(min(max_labels, int(hit.sum())), "y").itertuples()])
        for x in (-run.hit_lfc, run.hit_lfc):
            ax.axvline(x, color=GRID, lw=0.6, zorder=0)
        ax.axhline(-np.log10(run.hit_fdr), color=GRID, lw=0.6, zorder=0)
        ax.margins(x=0.20)
        ymax = float(t.y.max()) if len(t) else 1.0
        ax.set_ylim(-0.06 * ymax, ymax * 1.42)
        ax.set_title(c, fontsize=FS, pad=3)
        xlab = ("log2 fold change of the contrast's own estimate" if c in designed
                else f"log2 fold change, {run.case} vs {run.ctrl}")
        label(ax, x=xlab, y="-log10 adjusted p")
        tidy(ax, "both")
        save(fig, out, f"deg_volcano_{slug(c)}")
    legend_file(out, "deg_volcano_legend",
                [Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label=cap("tested gene")),
                 Line2D([], [], marker="o", ls="", color=hitcol, ms=4,
                        label=f"FDR < {run.hit_fdr:g}, |log2FC| > {run.hit_lfc:g}")], ncol=2, w=4.0)


# ============================================================ PPI networks
def _sheaf_graph(run, dom, prog):
    """The graph the sheaf ranked (one per programme per domain), built by the
    analysis code itself."""
    topo = os.path.join(REPO, "FeatureSelection", "Topology")
    if topo not in sys.path:
        sys.path.insert(0, topo)
    from SignificanceRankingPipeline import build_gcn
    from units import load_expr, tag

    sheaf = run.methods.get("sheaf", {})
    genes, M = load_expr(run.path("sheaf", "expr"), dom, prog)
    ch = pd.read_csv(run.path("sheaf", "charges", f"{tag(dom, prog)}_charge.csv")).set_index("gene").charge
    G = build_gcn(genes, run.path("sheaf", "string400.csv"), pd.DataFrame(M, index=genes),
                  ch.to_dict(), score_threshold=float(sheaf.get("string_score", 400)), case="upper",
                  weight_transform=sheaf.get("weight_transform", "rank"))
    del M
    return G, ch


def _halo_labels(ax, xy, idx, names, radius=1.32):
    """Names on a ring around the layout, in the angular order of their nodes,
    each on a leader line. A hairball with names written across it is unreadable."""
    c = xy.mean(0)
    span = np.abs(xy - c).max() or 1.0
    ang = np.arctan2(xy[idx, 1] - c[1], xy[idx, 0] - c[0])
    order = np.argsort(ang)
    for slot, k in enumerate(order):
        a = ang[order[0]] + 2 * np.pi * slot / max(len(order), 1)
        lx, ly = c[0] + radius * span * np.cos(a), c[1] + radius * span * np.sin(a)
        i = idx[k]
        ax.plot([xy[i, 0], lx], [xy[i, 1], ly], color=GRID, lw=0.4, zorder=1)
        ax.text(lx, ly, names[i], fontsize=FS, color=INK, zorder=5,
                ha="left" if np.cos(a) >= 0 else "right", va="center", clip_on=False)


def networks(run, out, **_):
    import networkx as nx
    from matplotlib.collections import LineCollection

    nets = run.networks()
    if not nets:
        print("  no sheaf networks; skipping"); return
    for dom, prog, resid in nets:
        u = f"domain{dom}_{prog}"
        G, ch = _sheaf_graph(run, dom, prog)

        # the PPI graph: layout by co-expression distance, node size by degree,
        # node colour by charge, the degree-residual set ringed and named
        fig, ax = newfig(4.6, 4.6)
        Gc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        nodes = list(Gc.nodes())
        pos = nx.kamada_kawai_layout(Gc, weight="weight")     # weight IS a distance
        xy = np.array([pos[n] for n in nodes])
        xy = xy - xy.mean(0)
        xy /= max(np.linalg.norm(xy, axis=1).max(), 1e-9)
        pos = {n: xy[i] for i, n in enumerate(nodes)}
        sim = np.array([1.0 - Gc[a][b]["weight"] for a, b in Gc.edges()])
        seg = np.array([[pos[a], pos[b]] for a, b in Gc.edges()])
        ax.add_collection(LineCollection(seg, colors=MUTED, alpha=0.20, zorder=1,
                                         linewidths=0.10 + 0.60 * sim))
        q = np.array([float(ch.get(n, 0.0)) for n in nodes])
        lim = float(np.abs(q).max()) or 1.0
        deg_ = np.array([Gc.degree(n) for n in nodes], dtype=float)
        size = 6 + 34 * (deg_ / max(deg_.max(), 1.0))
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c=q, cmap=DIV, vmin=-lim, vmax=lim,
                   edgecolor=SURFACE, lw=0.25, zorder=2)
        sel = [i for i, n in enumerate(nodes) if n in set(resid)]
        if sel:
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size[sel], facecolor="none",
                       edgecolor=INK, lw=1.0, zorder=3)
            _halo_labels(ax, xy, sel, nodes)
        ax.set_axis_off(); ax.set_aspect("equal"); ax.margins(0.26)
        ax.set_title(cap(f"domain {dom}, {prog}"), fontsize=FS, pad=2)
        sm = plt.cm.ScalarMappable(cmap=DIV, norm=plt.Normalize(-lim, lim))
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01, orientation="horizontal",
                          shrink=0.55)
        cb.set_label(cap("charge")); cb.outline.set_visible(False)
        save(fig, out, f"network_{u}")

        # the leave-one-out spectral shift against weighted degree
        ann_path = run.path("sheaf", "out_rank", f"{u}_sheaf_annotated.csv")
        if not os.path.isfile(ann_path):
            continue
        ann = pd.read_csv(ann_path)
        fig, ax = newfig(3.0, 2.8)
        inres = ann.gene.isin(resid)
        ax.scatter(ann.wdegree[~inres], ann.spectral_shift[~inres], s=7, color=MUTED,
                   alpha=0.5, lw=0)
        ax.scatter(ann.wdegree[inres], ann.spectral_shift[inres], s=16, color=CAT[1],
                   edgecolor=SURFACE, lw=0.4, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        label(ax, x="weighted degree", y="leave-one-out spectral shift")
        ax.set_title(cap(f"domain {dom}, {prog}"), fontsize=FS, pad=2)
        tidy(ax, "both")
        save(fig, out, f"residual_{u}")

    legend_file(out, "residual_legend",
                [Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label=cap("network gene")),
                 Line2D([], [], marker="o", ls="", color=CAT[1], ms=4,
                        label=cap("degree-residual set"))], ncol=2, w=4.0)


# ============================================================ entry point
FIGURES = {"qc": qc, "domains": domains, "programmes": programmes, "usage": usage,
           "deg": deg, "networks": networks}
NEEDS = {"qc": lambda r: r.path("celltypes_qc", "cells.csv.gz"),
         "domains": lambda r: r.path("composition", "cells_with_domain.csv.gz"),
         "programmes": lambda r: os.path.join(r.M["progact"], "program_top_genes.csv"),
         "usage": lambda r: r.fit("shared_coords_QH.npy"),
         "deg": lambda r: r.path("deg", "summary.csv"),
         "networks": lambda r: r.path("sheaf", "charges", "provenance.json")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run directory holding report/manifest.json")
    ap.add_argument("--out", default=None, help="default: <run>/report/figures")
    ap.add_argument("--only", nargs="+", default=None, choices=sorted(FIGURES))
    ap.add_argument("--programmes", nargs="+", default=None, metavar="F",
                    help="restrict the programmes and usage groups to these programmes "
                         "(e.g. F1 F3); default: all")
    a = ap.parse_args()

    run = Run(a.run)
    out = a.out or os.path.join(run.dir, "report", "figures")
    print(f"{run.label} -> {out}")
    for name in (a.only or list(FIGURES)):
        print(f"[{name}]", flush=True)
        if not os.path.exists(NEEDS[name](run)):
            print(f"  {os.path.relpath(NEEDS[name](run), run.dir)} not there yet; skipping")
            continue
        FIGURES[name](run, out, programmes=a.programmes)
    print("done")


if __name__ == "__main__":
    main()
