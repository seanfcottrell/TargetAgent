from __future__ import annotations

import json, os
import numpy as np, pandas as pd


def tag(domain, programme):
    """'domain5_F1' -- programme given as 'F1' or 1."""
    pr = programme if str(programme).startswith("F") else f"F{programme}"
    return f"domain{domain}_{pr}"


def networks(charges_dir, domains=None):
    """[(domain, programme), ...] in provenance order, for the domains asked
    (all charged domains when None). A domain absent from the provenance, or
    with no network, contributes nothing."""
    prov = json.load(open(os.path.join(charges_dir, "provenance.json")))
    out = []
    for d, v in prov.get("domains", {}).items():
        if domains is not None and str(d) not in [str(x) for x in domains]:
            continue
        for pr in v.get("networks", {}):
            out.append((str(d), pr))
    return out


def load_expr(expr_dir, domain, programme):
    """(genes, M) for one network: rows of the domain's union matrix restricted
    to that network's node genes, in the network's order. M is genes x nuclei,
    donor-centred log1p CP10K (build_expr.py)."""
    allg = pd.read_csv(os.path.join(expr_dir, f"domain{domain}_genes.csv")).gene.tolist()
    genes = pd.read_csv(os.path.join(expr_dir, f"{tag(domain, programme)}_genes.csv")).gene.tolist()
    M = np.load(os.path.join(expr_dir, f"domain{domain}_expr.npy"), mmap_mode="r")
    pos = pd.Index(allg).get_indexer(genes)
    assert (pos >= 0).all(), f"{tag(domain, programme)}: node gene missing from the domain matrix"
    return genes, np.ascontiguousarray(M[pos])
