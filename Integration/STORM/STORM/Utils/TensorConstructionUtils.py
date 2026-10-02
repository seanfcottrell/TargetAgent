import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
import sklearn.neighbors
import scanpy as sc
import torch
from typing import List, Tuple
import scipy.sparse
from scipy.sparse import issparse
import matplotlib.pyplot as plt


def build_irregular_slices(
    adatas_aligned: List[sc.AnnData],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[torch.Tensor], List[int], List[slice]]:
    """
    Convert aligned AnnData objects -> list of dense torch tensors [n_s_k, n_g].
    Also returns ns_list and row_slices for global stacking order.
    """
    X_list: List[torch.Tensor] = []
    ns_list: List[int] = []
    row_slices: List[slice] = []

    offset = 0
    for ad in adatas_aligned:
        ns, ng = ad.n_obs, ad.n_vars
        ns_list.append(ns)
        row_slices.append(slice(offset, offset + ns))
        offset += ns

        X_np = ad.X.toarray() if scipy.sparse.issparse(ad.X) else np.asarray(ad.X)
        X_t = torch.as_tensor(X_np, device=device, dtype=dtype).contiguous()
        X_list.append(X_t)

    return X_list, ns_list, row_slices
