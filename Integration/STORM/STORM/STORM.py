#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import scipy.sparse as sp

from STORM.Utils.TensorDecompositionUtils import attach_QHD_embeddings
from STORM.fit_STORM_chunked import fit_STORM


def _scipy_to_torch_sparse(L: sp.spmatrix, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    coo = L.tocoo()
    idx = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)).to(device)
    val = torch.from_numpy(coo.data.astype(np.float64)).to(device)
    return torch.sparse_coo_tensor(idx, val, coo.shape, dtype=dtype, device=device).coalesce()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class STORM:
    """Graph-regularised PARAFAC2 over irregular slices: X_k ~ Q_k H D_k B^T.

    The caller supplies the slices (`X_list`, `ns_list`, `adatas_aligned`) and
    the two Laplacians (`Ls_torch` over all bins, `Lg_torch` over genes), then
    calls fit() and attach_embeddings()."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
        seed: int = 0,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.seed = seed
        _set_seeds(seed)

        self.adatas_aligned: List = []
        self.X_list: List[torch.Tensor] = []
        self.ns_list: List[int] = []

        self.Ls_torch: Optional[torch.Tensor] = None
        self.Lg_torch: Optional[torch.Tensor] = None

        self.Q_list: Optional[List[torch.Tensor]] = None
        self.H: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.w_list: Optional[List[torch.Tensor]] = None
        self.Y: Optional[torch.Tensor] = None
        self.fit_info: Optional[Dict] = None

    def fit(
        self,
        rank: int = 30,
        gamma: float = 0.5,
        iters: int = 50,
        rho: float = 1.0,
        tol: float = 1e-5,
        # --- nonnegativity / sparsity on the shared gene factor B ---
        nonneg_B: bool = True,
        beta: float = 0.0,
        hals_sweeps: int = 5,
        nonneg_w: Optional[bool] = None,
        init: str = "svd",
        gene_chunk_cols: int = 256,
        verbose: bool = False,
    ) -> "STORM":
        """
        gene_chunk_cols : column budget per chunk in the chunked Sylvester
                   solve (fit_STORM_chunked). Exact reformulation, so this
                   only trades peak GPU memory against per-chunk overhead;
                   halve it if the Y-step OOMs.
        nonneg_B : constrain B >= 0. Makes each factor a nonnegative gene
                   program, so columns of B can be passed directly to
                   decoupleR / enrichment as weighted signatures.
        beta     : L1 weight on B. Only interpretable because fit_STORM
                   renormalises B's columns to unit L2 each iteration and
                   absorbs the scale into w_k -- otherwise the scale
                   ambiguity B -> B/s, w -> w*s makes the penalty vacuous.
                   Calibrate against the magnitude of the B-step linear
                   term, not against raw expression: start near a low
                   quantile of |RB|/(1+rho/2) and sweep.
        nonneg_w : constrain w_k >= 0 (defaults to nonneg_B). With B >= 0
                   the sign ambiguity migrates entirely into D_k; a negative
                   w_kr flips the whole factor for slice k and makes
                   nonnegative loadings meaningless.
        """
        if self.Ls_torch is None or self.Lg_torch is None:
            raise RuntimeError("Build spatial and gene graphs before calling fit().")
        if not self.X_list:
            raise RuntimeError("Build tensor slices before calling fit().")

        self.Q_list, self.H, self.B, self.w_list, self.Y, self.fit_info = fit_STORM(
            self.X_list,
            self.Lg_torch,
            self.Ls_torch,
            rank=rank,
            gamma=gamma,
            iters=iters,
            rho=rho,
            tol=tol,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
            nonneg_B=nonneg_B,
            beta=beta,
            hals_sweeps=hals_sweeps,
            nonneg_w=nonneg_w,
            init=init,
            gene_chunk_cols=gene_chunk_cols,
            verbose=verbose,
            return_info=True,
        )

        if nonneg_B:
            n_dead = self.fit_info["dead"][-1]
            if n_dead:
                print(f"[STORM] WARNING: {n_dead}/{rank} B columns collapsed. "
                      f"beta={beta} is too large — this is not rank selection.")
            print(f"[STORM] B sparsity={self.fit_info['sparsity'][-1]:.3f}  "
                  f"KKT={self.fit_info['kkt_B'][-1]:.2e}")
        return self

    def attach_embeddings(
        self,
        key: str = "X_STORM",
        also_store_shape: bool = False,
        store_normed: bool = True,
    ) -> "STORM":
        if self.Q_list is None:
            raise RuntimeError("Call fit() before attach_embeddings().")
        attach_QHD_embeddings(
            self.adatas_aligned,
            self.Q_list, self.H, self.w_list,
            key=key,
            also_store_shape=also_store_shape,
            store_normed=store_normed,
        )
        print(f"[STORM] embeddings attached (key='{key}')")
        return self
