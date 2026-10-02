"""
STORM: Subspace Tensor Orthogonal Rotation Model
with nonnegativity + L1 sparsity on the shared gene-mode factor B.

Model
-----
    X_k  ~  Q_k H D_k B^T ,      Q_k^T Q_k = I ,  D_k = diag(w_k)

ADMM splits the graph-regularized reconstruction:

    min  sum_k ||X_k - Q_k H D_k B^T||_F^2
         + gamma * ( tr(Y^T Ls Y) + tr(Y Lg Y^T) )
         + beta * ||B||_1   s.t. B >= 0   (and optionally w_k >= 0)

B-subproblem (this file's main change)
--------------------------------------
Collecting B-dependent terms with a := 1 + rho/2:

    min_{B>=0}  a * tr(B SB B^T) - 2 tr(RB^T B) + beta * ||B||_1

Because B >= 0, ||B||_1 = <B, 11^T> is *linear*, so it folds into the
linear term:  Mt := RB/a - beta/(2a).  The subproblem is then a plain
nonnegative QP with an R x R Gram matrix, solved exactly column-by-column
by HALS:

    b_r  <-  [ b_r + (Mt[:,r] - (B SB)[:,r]) / SB[r,r] ]_+

Optimality is a KKT condition, not a zero-gradient equation:
    Lam := B SB - Mt
    B_gr > 0  =>  Lam_gr == 0
    B_gr = 0  =>  Lam_gr >= 0
"""

import math

import torch
from torch import Tensor
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import scipy.sparse as sp


# ======================================================================
# utilities
# ======================================================================

def _block_row_slices(ns_list: List[int]) -> List[slice]:
    offs, out = 0, []
    for ns in ns_list:
        out.append(slice(offs, offs + ns))
        offs += ns
    return out


def _procrustes_Q(XW_t: Tensor) -> Tensor:
    U, _, Vh = torch.linalg.svd(XW_t, full_matrices=False)
    return U @ Vh


def _diag_of(FtXB: Tensor) -> Tensor:
    return torch.diag(FtXB)


def _make_sym(A):
    return 0.5 * (A + A.transpose(-1, -2))


def solve_spd_then_fallback(A, B, jitter0=1e-8, growth=10.0, tries=6):
    A = _make_sym(A)
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    scale = torch.clamp(A.diagonal(dim1=-2, dim2=-1).abs().mean(), min=1.0)
    jitter = float(jitter0) * float(scale)

    for _ in range(tries):
        try:
            L = torch.linalg.cholesky(A + jitter * I)
            return torch.cholesky_solve(B, L)
        except RuntimeError:
            jitter *= growth
    try:
        return torch.linalg.solve(A + jitter * I, B)
    except RuntimeError:
        return torch.linalg.lstsq(A + jitter * I, B).solution


def _extract_sparse_diag_scipy(L: sp.spmatrix) -> np.ndarray:
    """Extract main diagonal from a scipy sparse matrix."""
    return np.asarray(L.diagonal(), dtype=np.float64)


# ======================================================================
# nonnegative + L1 solvers  (B-step and D-step)
# ======================================================================
def _column_betas(RB: Tensor, beta: float, q: float = 0.9) -> Optional[Tensor]:
    """
    Per-column L1 weight, scaled to each column's own linear term.

    A single scalar beta is not scale-fair: during the solve, column r's
    magnitude tracks w_kr, so a uniform shift is negligible for active
    factors and lethal for weak ones -- which is what collapses columns.
    Referencing beta to quantile(|RB[:,r]|) makes it dimensionless and
    comparable across factors: beta=0.5 means "threshold at half the 90th
    percentile of this column's drive," whatever that column's scale.
    """
    if beta == 0.0:
        return None
    scale = torch.quantile(RB.abs(), q, dim=0)      # (R,)
    return beta * scale.clamp_min(1e-30)

def _hals_nonneg_l1(
    B: Tensor,
    SB: Tensor,
    RB: Tensor,
    a: float,
    beta: float,
    sweeps: int = 5,
    dead_tol: float = 1e-10,
) -> Tensor:
    """
    Exact column-wise block coordinate descent for

        min_{B >= 0}  a * tr(B SB B^T) - 2 tr(RB^T B) + beta * ||B||_1

    B is updated IN PLACE (warm start from the previous ADMM iteration).

    Each column subproblem separates across genes into independent scalar
    quadratics with curvature a*SB[r,r] > 0, so the projected closed-form
    solution is the exact block minimizer -- not a gradient step.

    Parameters
    ----------
    SB : (R, R) PSD Gram, sum_k C_k^T C_k
    RB : (ng, R) linear term, sum_k (X_k + (rho/2) Ytilde_k)^T C_k
    a  : 1 + rho/2
    beta : L1 weight. Meaningful only if B's columns are normalized each
           outer iteration (otherwise the scale ambiguity B -> B/s,
           w -> w*s makes the penalty vacuous).
    """
    #if beta != 0.0:
     #   Mt = RB / a - (beta / (2.0 * a))
    #else:
     #   Mt = RB / a
    betas = _column_betas(RB, beta)
    Mt = RB / a if betas is None else RB / a - betas.unsqueeze(0) / (2.0 * a)

    d = torch.diagonal(SB).clone()
    # A column whose factor has collapsed has SB[r,r] ~ 0; dividing by it
    # would resurrect the column explosively. Skip those instead.
    thresh = dead_tol * torch.clamp(d.max(), min=1.0)

    R = B.shape[1]
    for _ in range(sweeps):
        for r in range(R):
            if d[r] <= thresh:
                continue
            BG_r = B @ SB[:, r]                       # (ng,)
            B[:, r] = torch.clamp(B[:, r] + (Mt[:, r] - BG_r) / d[r], min=0.0)
    return B


def _kkt_residual_B(B: Tensor, SB: Tensor, RB: Tensor,
                    a: float, beta: float) -> Tensor:
    """
    Max KKT violation for the nonnegative B-subproblem.

        Lam = B SB - Mt
        active   (B > 0): |Lam| must be 0
        inactive (B = 0):  Lam must be >= 0
    """
    #Mt = RB / a - (beta / (2.0 * a)) if beta != 0.0 else RB / a
    betas = _column_betas(RB, beta)
    Mt = RB / a if betas is None else RB / a - betas.unsqueeze(0) / (2.0 * a)
    Lam = B @ SB - Mt
    act = B > 0
    r1 = (Lam.abs() * act).max() if bool(act.any()) else Lam.new_zeros(())
    r2 = ((-Lam).clamp_min(0.0) * (~act)).max() if bool((~act).any()) else Lam.new_zeros(())
    return torch.maximum(r1, r2)


def _nnqp_coord(w: Tensor, Gram: Tensor, rhs: Tensor,
                a: float, sweeps: int = 20, dead_tol: float = 1e-10) -> Tensor:
    """
    Small nonnegative QP for the D-step:

        min_{w >= 0}  a * w^T Gram w - 2 w^T rhs

    Projected coordinate descent; R is small so this is cheap and exact
    per coordinate.

    Why constrain w >= 0: with B >= 0 the sign ambiguity that used to sit
    in (B, D) migrates entirely into D. A negative w_kr flips the whole
    factor for slice k, which makes "nonnegative loadings" meaningless and
    is not interpretable for count data. Q_k can absorb any genuinely
    needed global sign.
    """
    g = torch.diagonal(Gram).clone()
    thresh = dead_tol * torch.clamp(g.max(), min=1.0)
    rhs_a = rhs / a
    R = w.shape[0]
    w = w.clone().clamp_min(0.0)
    for _ in range(sweeps):
        max_move = 0.0
        for r in range(R):
            if g[r] <= thresh:
                continue
            gr = torch.dot(Gram[:, r], w)
            new = torch.clamp(w[r] + (rhs_a[r] - gr) / g[r], min=0.0)
            max_move = max(max_move, float((new - w[r]).abs()))
            w[r] = new
        if max_move < 1e-12:
            break
    return w


# ======================================================================
# Sylvester solvers for the Y-step
# ======================================================================

def sylvester_cg_numpy(
    Ls_csr: sp.csr_matrix,       # (S, S) scipy CSR
    Lg_csr: sp.csr_matrix,       # (ng, ng) scipy CSR
    alpha: float,
    gamma: float,
    RHS: np.ndarray,             # (S, ng) numpy array
    Y0: np.ndarray = None,       # (S, ng) warm start
    max_iter: int = 300,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Preconditioned CG for the Sylvester equation, entirely in numpy/scipy.

    Uses scipy CSR @ dense for sparse matmuls (MKL-accelerated on most
    installs), which is 3-10x faster than torch.sparse.mm on CPU.
    """
    S, ng = RHS.shape
    g2 = 2.0 * gamma

    # Mirrors the torch path: gamma == 0 collapses A to alpha*I, so the solve
    # is exactly RHS/alpha and none of the CG state needs allocating.
    if gamma == 0.0:
        return RHS / alpha

    ds = _extract_sparse_diag_scipy(Ls_csr)    # (S,)
    dg = _extract_sparse_diag_scipy(Lg_csr)    # (ng,)
    Mdiag = alpha + g2 * (ds[:, None] + dg[None, :])
    Mdiag_inv = 1.0 / (Mdiag + 1e-30)

    def A_op(Y):
        LsY = Ls_csr @ Y
        YLg = (Lg_csr @ Y.T).T     # Lg symmetric => = Y @ Lg
        return alpha * Y + g2 * (LsY + YLg)

    Y = Y0.copy() if Y0 is not None else np.zeros_like(RHS)

    R = RHS - A_op(Y)
    Z = R * Mdiag_inv
    P = Z.copy()
    rz_old = np.sum(R * Z)

    if np.linalg.norm(R) <= tol:
        return Y

    for _ in range(max_iter):
        AP = A_op(P)
        pAp = np.sum(P * AP) + 1e-30
        alpha_k = rz_old / pAp

        Y += alpha_k * P
        R -= alpha_k * AP

        if np.linalg.norm(R) <= tol:
            break

        Z = R * Mdiag_inv
        rz_new = np.sum(R * Z)
        beta_cg = rz_new / (rz_old + 1e-30)
        P = Z + beta_cg * P
        rz_old = rz_new

    return Y


def sparse_diag(L: torch.Tensor) -> torch.Tensor:
    LC = L.coalesce()
    r, c = LC.indices()
    v = LC.values()
    d = torch.zeros(L.shape[0], dtype=v.dtype, device=v.device)
    m = (r == c)
    if m.any():
        d.index_add_(0, r[m], v[m])
    return d


def lg_component_chunks(Lg: Tensor, ng: int,
                        max_cols: int = 256) -> List[Tensor]:
    """
    Partition the gene columns into chunks, each a UNION OF CONNECTED
    COMPONENTS of Lg's off-diagonal sparsity pattern.

    WHY THIS IS EXACT, AND WHY COMPONENTS MUST NOT BE SPLIT
    -------------------------------------------------------
    A(Y) = alpha*Y + 2*gamma*(Ls @ Y + Y @ Lg). Column-block C by column-block:

        (alpha*Y)[:, C] = alpha * Y[:, C]        column-local
        (Ls @ Y)[:, C]  = Ls @ Y[:, C]           Ls acts on ROWS only, so it
                                                 hits every column independently
        (Y @ Lg)[:, C]  = sum_m Y[:, m] Lg[m, C] column-local ONLY IF no gene
                                                 outside C has an Lg edge into C

    So when C is a union of connected components, A(Y)[:, C] = A_C(Y[:, C]) with
    A_C(V) = alpha*V + 2*gamma*(Ls @ V + V @ Lg[C, C]). alpha > 0 makes each
    A_C SPD, so every block system has the same unique solution the monolithic
    solve would have reached: this is an exact reformulation, not an
    approximation. Only the Krylov path differs (a separate basis per block
    rather than one shared basis).

    A chunk that SPLIT a component would silently drop the cross-column
    coupling and solve a DIFFERENT operator, so components are never broken up;
    one larger than max_cols becomes its own oversized chunk.

    The gene graph is a thresholded PPI network, so in practice it is mostly
    singletons (at STRING 600 on 2000 HVGs: 1875 isolated genes, and 125 genes
    in 18 components, the largest of which is 57) -- which is exactly the
    structure that makes this worth doing.
    """
    if Lg.is_sparse:
        idx = Lg.coalesce().indices()
        rr, cc = idx[0].tolist(), idx[1].tolist()
    else:
        nz = torch.nonzero(Lg, as_tuple=False)
        rr, cc = nz[:, 0].tolist(), nz[:, 1].tolist()

    parent = list(range(ng))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(rr, cc):
        if a != b:                      # diagonal never couples two columns
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups: Dict[int, List[int]] = {}
    for j in range(ng):
        groups.setdefault(find(j), []).append(j)

    chunks: List[List[int]] = []
    cur: List[int] = []
    for comp in sorted(groups.values(), key=len, reverse=True):
        if len(comp) >= max_cols:       # never split a component
            chunks.append(comp)
            continue
        if cur and len(cur) + len(comp) > max_cols:
            chunks.append(cur)
            cur = []
        cur.extend(comp)
    if cur:
        chunks.append(cur)

    dev = Lg.device
    return [torch.as_tensor(ch, dtype=torch.long, device=dev) for ch in chunks]


def _lg_submatrix(Lg: Tensor, cols: Tensor) -> Tensor:
    """
    Lg[C, C], indices remapped to local 0..|C|-1.

    Because C is a union of connected components, every Lg entry with an
    endpoint in C has BOTH endpoints in C -- nothing is dropped here.
    """
    m = int(cols.numel())
    if not Lg.is_sparse:
        return Lg.index_select(0, cols).index_select(1, cols)

    Lc = Lg.coalesce()
    idx, val = Lc.indices(), Lc.values()
    pos = torch.full((Lg.shape[0],), -1, dtype=torch.long, device=cols.device)
    pos[cols] = torch.arange(m, dtype=torch.long, device=cols.device)
    keep = (pos[idx[0]] >= 0) & (pos[idx[1]] >= 0)
    loc = torch.stack([pos[idx[0][keep]], pos[idx[1][keep]]])
    return torch.sparse_coo_tensor(loc, val[keep], (m, m)).coalesce()


def _sylvester_cg_block(Ls: Tensor, Lg: Tensor, alpha: float, gamma: float,
                        RHS: Tensor, Y0: Tensor = None, max_iter: int = 300,
                        tol: float = 1e-6,
                        stats: Optional[Dict[str, Any]] = None) -> Tensor:
    """
    Preconditioned CG on ONE column block. Identical algorithm to the original
    monolithic solver; it just runs on (S x |C|) instead of (S x ng).

    `stats`, if given, accumulates iterations used and the final residual so
    the caller can tell a converged Y-step from one that merely ran out of
    iterations. Without that you cannot tell whether `tol` is doing anything:
    the test is on the ABSOLUTE ||R||, so on real data it may be unreachable
    and the loop simply always runs `max_iter` times.
    """
    ds = sparse_diag(Ls) if Ls.is_sparse else torch.diag(Ls)
    dg = sparse_diag(Lg) if Lg.is_sparse else torch.diag(Lg)

    Mdiag = alpha + 2.0 * gamma * (ds[:, None] + dg[None, :])

    def A(Y):
        t1 = torch.sparse.mm(Ls, Y) if Ls.is_sparse else Ls @ Y
        t2 = (torch.sparse.mm(Lg, Y.T).T) if Lg.is_sparse else (Y @ Lg)
        return alpha * Y + 2.0 * gamma * (t1 + t2)

    def M_inv(R):
        return R / (Mdiag + 1e-30)

    rhs_norm = float(torch.linalg.norm(RHS))
    Y = torch.zeros_like(RHS) if Y0 is None else Y0.clone()
    R = RHS - A(Y)
    Z = M_inv(R)
    P = Z.clone()
    rz_old = torch.sum(R * Z)

    used, hit_tol = 0, False
    rn = float(torch.linalg.norm(R))
    if rn <= tol:
        hit_tol = True
    else:
        for _ in range(max_iter):
            AP = A(P)
            denom = torch.sum(P * AP) + 1e-30
            alpha_k = rz_old / denom
            Y = Y + alpha_k * P
            R = R - alpha_k * AP
            used += 1
            rn = float(torch.linalg.norm(R))
            if rn <= tol:
                hit_tol = True
                break
            Z = M_inv(R)
            rz_new = torch.sum(R * Z)
            beta_cg = rz_new / (rz_old + 1e-30)
            P = Z + beta_cg * P
            rz_old = rz_new

    if stats is not None:
        stats["iters"] = stats.get("iters", 0) + used
        stats["max_iters"] = max(stats.get("max_iters", 0), used)
        stats["blocks"] = stats.get("blocks", 0) + 1
        stats["converged"] = stats.get("converged", 0) + int(hit_tol)
        # relative residual is the scale-free number worth reading
        stats["res2"] = stats.get("res2", 0.0) + rn ** 2
        stats["rhs2"] = stats.get("rhs2", 0.0) + rhs_norm ** 2
    return Y


def sylvester_cg_torch(Ls: Tensor, Lg: Tensor, alpha: float, gamma: float,
                       RHS: Tensor, Y0: Tensor = None, max_iter: int = 300,
                       tol: float = 1e-6,
                       col_chunks: Optional[List[Tensor]] = None,
                       stats: Optional[Dict[str, Any]] = None) -> Tensor:
    """
    torch-based CG (used when device is CUDA), solved one gene-component chunk
    at a time so the iteration state is (S x |C|) rather than (S x ng).

    Peak memory, not arithmetic, is what forces this: CG holds R, Z, P, AP and
    the diagonal preconditioner concurrently, plus the t1/t2 temporaries inside
    every A() call. At full width on this data that is ~10 dense (S x ng)
    tensors on top of the ADMM variables, which is what exhausted an 80GB A100.
    """
    # gamma == 0 collapses A to alpha*I, so alpha*Y = RHS is solved exactly by
    # a division. select_rank() fits with gamma=0.0 AND zero Laplacians, so the
    # full CG machinery there was allocating ~10 full-width tensors to compute
    # something closed-form. This is exact, not a tolerance-level shortcut.
    if gamma == 0.0:
        return RHS / alpha

    ng = RHS.shape[1]
    if col_chunks is None:
        col_chunks = [torch.arange(ng, dtype=torch.long, device=RHS.device)]

    if len(col_chunks) == 1 and int(col_chunks[0].numel()) == ng:
        return _sylvester_cg_block(Ls, Lg, alpha, gamma, RHS, Y0, max_iter,
                                   tol, stats)

    # Per-block absolute tolerance, so the AGGREGATE stopping criterion matches
    # the monolithic one: if every block has ||R_c|| <= tol/sqrt(nc), then
    # ||R||^2 = sum_c ||R_c||^2 <= tol^2.
    tol_c = tol / math.sqrt(len(col_chunks))

    Y = torch.empty_like(RHS)
    for cols in col_chunks:
        RHS_c = RHS.index_select(1, cols)
        Y0_c = Y0.index_select(1, cols) if Y0 is not None else None
        Lg_c = _lg_submatrix(Lg, cols)
        Y_c = _sylvester_cg_block(Ls, Lg_c, alpha, gamma,
                                  RHS_c, Y0_c, max_iter, tol_c, stats)
        Y.index_copy_(1, cols, Y_c)
        del RHS_c, Y0_c, Lg_c, Y_c
    return Y


def _torch_sparse_to_scipy_csr(T: Tensor) -> sp.csr_matrix:
    """Convert a coalesced torch sparse COO tensor to scipy CSR."""
    T = T.coalesce().cpu()
    idx = T.indices().numpy()
    val = T.values().numpy()
    return sp.coo_matrix((val, (idx[0], idx[1])), shape=T.shape).tocsr()


# ======================================================================
# initialisation
# ======================================================================

def _init_B(X_list: List[Tensor], rank: int, mode: str,
            nonneg: bool, device, dtype, seed: int) -> Tensor:
    """
    SVD init on the row-concatenated data (deterministic), or random.
    Under nonnegativity we take |V| -- a random init would be clipped to
    ~50% zeros on the first HALS sweep, which is a poor starting point.
    """
    ng = X_list[0].shape[1]
    if mode == "svd":
        try:
            Xall = torch.vstack(X_list)
            q = min(rank + 10, min(Xall.shape) - 1)
            q = max(q, rank)
            _, _, V = torch.svd_lowrank(Xall, q=q, niter=4)
            B = V[:, :rank].contiguous()
            if nonneg:
                B = B.abs()
            return B.to(device=device, dtype=dtype)
        except Exception:
            pass  # fall through to random
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = torch.rand(ng, rank, generator=g) if nonneg else torch.randn(ng, rank, generator=g)
    return B.to(device=device, dtype=dtype)


# ======================================================================
# main fit
# ======================================================================

@torch.no_grad()
def fit_STORM(
    X_list: List[Tensor],
    Lg: Tensor,
    Ls: Tensor,
    rank: int,
    gamma: float,
    iters: int = 50,
    rho: float = 1.0,
    tol: float = 1e-5,
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    # ---- nonnegativity / sparsity ----
    nonneg_B: bool = True,
    beta: float = 0.0,
    hals_sweeps: int = 5,
    nonneg_w: Optional[bool] = None,   # defaults to nonneg_B
    init: str = "svd",                 # "svd" | "random"
    # ---- Y-step memory ----
    gene_chunk_cols: int = 256,        # 0 / None disables chunking
    force_torch_cg: bool = False,      # testing only: use the torch CG on CPU
    # ---- inner (Y-step) CG ----
    # NOTE: the CG stopping test is on the ABSOLUTE ||R||, so whether any given
    # tol is reachable depends on the scale of RHS. Read the [cg] line in the
    # verbose output before tuning this: if conv=0/N every iteration, the
    # tolerance never fires and only cg_max_iter is controlling accuracy, so
    # loosening tol changes nothing -- and if it does start firing, it stops CG
    # EARLIER, giving the outer ADMM a less accurate Y-step, not a better one.
    cg_tol: float = 1e-4,
    cg_max_iter: int = 300,
    # ---- diagnostics ----
    verbose: bool = False,
    return_info: bool = False,
):
    """
    Returns
    -------
    (Q_list, H, B, w_list, Y)                        if return_info is False
    (Q_list, H, B, w_list, Y, info)                  if return_info is True

    info : dict with per-iteration
        prim_res  : relative ADMM primal residual
        kkt_B     : max KKT violation of the B-subproblem (nonneg case)
        sparsity  : fraction of exactly-zero entries in B
        dead      : number of collapsed (all-zero) columns of B
        w_neg     : number of negative w entries (should be 0 if nonneg_w)

    Notes
    -----
    * beta is only interpretable because B's columns are renormalised to
      unit L2 each outer iteration and the scale absorbed into w_k. Without
      that, the optimizer shrinks B and inflates w rather than sparsifying.
    * Because ||B||_1 is linear on the nonnegative orthant, the penalty
      pushes every entry down with equal force and can collapse whole
      columns. Watch info["dead"] -- column death means beta is too large,
      it is not principled rank selection.
    """
    torch.manual_seed(seed)

    if nonneg_w is None:
        nonneg_w = nonneg_B

    if device is None:
        device = X_list[0].device

    X_list = [x.to(device=device, dtype=dtype) for x in X_list]
    Lg = Lg.to(device=device, dtype=dtype)
    Ls = Ls.to(device=device, dtype=dtype)

    # force_torch_cg exists so the chunked path can be exercised on a CPU-only
    # node. It does not change behaviour on GPU, where use_cpu_cg is already
    # False.
    use_cpu_cg = (device.type == "cpu") and not force_torch_cg

    if use_cpu_cg:
        Ls_csr = (_torch_sparse_to_scipy_csr(Ls) if Ls.is_sparse
                  else sp.csr_matrix(Ls.cpu().numpy()))
        Lg_csr = (_torch_sparse_to_scipy_csr(Lg) if Lg.is_sparse
                  else sp.csr_matrix(Lg.cpu().numpy()))
    else:
        if Ls.is_sparse:
            Ls = Ls.coalesce()
        if Lg.is_sparse:
            Lg = Lg.coalesce()

    nb = len(X_list)
    ns_list = [x.shape[0] for x in X_list]
    ng = X_list[0].shape[1]
    S = sum(ns_list)
    row_chunks = _block_row_slices(ns_list)

    # Gene-column chunks for the Y-step. Computed once: the partition depends
    # only on Lg's sparsity, which is fixed for the whole fit. Skipped when
    # gamma == 0, since that path never runs CG at all.
    col_chunks = None
    if (not use_cpu_cg) and gamma != 0.0 and gene_chunk_cols and gene_chunk_cols < ng:
        col_chunks = lg_component_chunks(Lg, ng, max_cols=int(gene_chunk_cols))
        if verbose:
            widths = [int(c.numel()) for c in col_chunks]
            print(f"[Y-step] {len(col_chunks)} gene chunks, "
                  f"widths min={min(widths)} max={max(widths)} "
                  f"(cap {int(gene_chunk_cols)} of {ng})")

    # ---------------- init factors ----------------
    Q_list = []
    for ns in ns_list:
        G0 = torch.randn(ns, rank, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(G0, mode='reduced')
        Q_list.append(Q[:, :rank].contiguous())

    H = torch.eye(rank, device=device, dtype=dtype)

    B = _init_B(X_list, rank, init, nonneg_B, device, dtype, seed)
    col_norms = torch.clamp(torch.linalg.norm(B, dim=0), min=1e-8)
    B = (B / col_norms).contiguous()
    w_list = [col_norms.clone() for _ in range(nb)]

    # ---------------- ADMM vars ----------------
    Y = torch.zeros(S, ng, device=device, dtype=dtype)
    U = torch.zeros_like(Y)
    eyeR = torch.eye(rank, device=device, dtype=dtype)
    epsR = 1e-8
    a_b = 1.0 + rho / 2.0

    if use_cpu_cg:
        Y_np = np.zeros((S, ng), dtype=np.float64)

    info: Dict[str, Any] = {k: [] for k in
                            ("prim_res", "kkt_B", "sparsity", "dead", "w_neg")}

    # ---------------- main loop ----------------
    for it in range(1, iters + 1):

        # ----- current reconstruction -----
        Yhat_blocks = []
        for k in range(nb):
            C_k = (Q_list[k] @ H) * w_list[k]
            Yhat_blocks.append(C_k @ B.T)
        S_stack = torch.vstack(Yhat_blocks)

        # ----- Y-step: Sylvester via CG -----
        RHS_t = rho * (S_stack - U)

        cg_stats: Dict[str, Any] = {}
        if use_cpu_cg:
            RHS_np = RHS_t.cpu().numpy().astype(np.float64)
            Y_np = sylvester_cg_numpy(
                Ls_csr, Lg_csr, alpha=rho, gamma=gamma,
                RHS=RHS_np, Y0=Y_np, max_iter=cg_max_iter, tol=cg_tol,
            )
            Y = torch.from_numpy(Y_np).to(device=device, dtype=dtype)
        else:
            Y = sylvester_cg_torch(
                Ls, Lg, alpha=rho, gamma=gamma,
                RHS=RHS_t, Y0=Y, max_iter=cg_max_iter, tol=cg_tol,
                col_chunks=col_chunks, stats=cg_stats,
            )

        del RHS_t                      # free before the Q/H/D/B steps allocate

        prim_res = torch.linalg.norm(Y - S_stack) / (torch.linalg.norm(Y) + 1e-12)

        # Ytilde = Y + U is only ever read one row-block at a time, so it is
        # built per block instead of materialising another full (S x ng) copy.
        def _yt(rows):
            return Y[rows, :] + U[rows, :]

        # ----- Q-step (Procrustes) -----
        for k in range(nb):
            rows = row_chunks[k]
            W_k = (H * w_list[k].unsqueeze(0)) @ B.T
            Mk = X_list[k] @ W_k.T + (rho / 2.0) * (_yt(rows) @ W_k.T)
            Q_list[k] = _procrustes_Q(Mk)

        # ----- H-step -----
        SH = torch.zeros(rank, rank, device=device, dtype=dtype)
        RH = torch.zeros(rank, rank, device=device, dtype=dtype)
        for k in range(nb):
            G_k = (B * w_list[k]).T
            SH += G_k @ G_k.T
            QtX = Q_list[k].T @ X_list[k]
            rows = row_chunks[k]
            QtYt = Q_list[k].T @ _yt(rows)
            RH += (QtX + (rho / 2.0) * QtYt) @ G_k.T

        A_h = SH + epsR * eyeR
        H = solve_spd_then_fallback(A_h, (RH.T / a_b)).T

        # ----- D-step -----
        BtB = B.T @ B
        for k in range(nb):
            F_k = Q_list[k] @ H
            Gram = BtB * (F_k.T @ F_k)
            FtX_B = F_k.T @ X_list[k] @ B
            rows = row_chunks[k]
            FtY_B = F_k.T @ _yt(rows) @ B
            rhs_vec = _diag_of(FtX_B) + (rho / 2.0) * _diag_of(FtY_B)

            if nonneg_w:
                w_list[k] = _nnqp_coord(w_list[k], Gram, rhs_vec, a=a_b)
            else:
                A_D = a_b * Gram + epsR * eyeR
                w_list[k] = solve_spd_then_fallback(
                    A_D, rhs_vec.unsqueeze(1)).squeeze(1)

        # ----- B-step -----
        # SB = sum_k C_k^T C_k = sum_k D_k H^T Q_k^T Q_k H D_k
        #    = sum_k D_k (H^T H) D_k          since Q_k^T Q_k = I.
        # No (ns x R) products needed.
        HtH = H.T @ H
        SB = torch.zeros(rank, rank, device=device, dtype=dtype)
        RB = torch.zeros(ng, rank, device=device, dtype=dtype)
        for k in range(nb):
            rows = row_chunks[k]
            wk = w_list[k]
            SB += wk.unsqueeze(1) * HtH * wk.unsqueeze(0)
            C_k = (Q_list[k] @ H) * wk
            RB += X_list[k].T @ C_k
            RB += (rho / 2.0) * (_yt(rows).T @ C_k)
        SB = _make_sym(SB)

        if nonneg_B:
            B = _hals_nonneg_l1(B, SB, RB, a=a_b, beta=beta, sweeps=hals_sweeps)
            kkt = _kkt_residual_B(B, SB, RB, a=a_b, beta=beta)
        else:
            A_bmat = SB + epsR * eyeR
            B = solve_spd_then_fallback(A_bmat, (RB.T / a_b)).T
            kkt = torch.zeros((), device=device, dtype=dtype)

        # ----- column rescale B -> absorb into w_k -----
        # Only rescale live columns: dividing a collapsed (all-zero) column
        # by a clamped 1e-8 and multiplying w_k by it silently destroys that
        # factor's activity record.
        norms = torch.linalg.norm(B, dim=0)
        alive = norms > 1e-8
        if bool(alive.any()):
            B[:, alive] = B[:, alive] / norms[alive]
            for k in range(nb):
                w_list[k] = torch.where(alive, w_list[k] * norms, w_list[k])

        # ----- dual update -----
        Yhat_blocks = []
        for k in range(nb):
            C_k = (Q_list[k] @ H) * w_list[k]
            Yhat_blocks.append(C_k @ B.T)
        S_stack = torch.vstack(Yhat_blocks)
        U = U + (Y - S_stack)

        # ----- diagnostics -----
        n_dead = int((~alive).sum())
        sparsity = float((B == 0).sum()) / float(B.numel())
        w_neg = int(sum(int((w < 0).sum()) for w in w_list))
        info["prim_res"].append(float(prim_res))
        info["kkt_B"].append(float(kkt))
        info["sparsity"].append(sparsity)
        info["dead"].append(n_dead)
        info["w_neg"].append(w_neg)

        # Y-step accuracy. cg_rel is the scale-free number: if it is not small,
        # the outer ADMM is being fed an inexact Y and cannot be expected to
        # converge no matter what the outer tol is set to.
        cg_rel = float("nan")
        if cg_stats.get("rhs2"):
            cg_rel = (cg_stats["res2"] / cg_stats["rhs2"]) ** 0.5
        info.setdefault("cg_rel_res", []).append(cg_rel)
        info.setdefault("cg_iters_max", []).append(cg_stats.get("max_iters", 0))
        info.setdefault("cg_converged_blocks", []).append(
            (cg_stats.get("converged", 0), cg_stats.get("blocks", 0)))

        if verbose:
            print(f"[{it:3d}] prim={float(prim_res):.3e}  kktB={float(kkt):.3e}  "
                  f"sparsity={sparsity:.3f}  dead={n_dead}  w<0={w_neg}")
            if cg_stats:
                print(f"      [cg] rel_res={cg_rel:.3e}  "
                      f"iters_max={cg_stats.get('max_iters', 0)}/{cg_max_iter}  "
                      f"conv={cg_stats.get('converged', 0)}"
                      f"/{cg_stats.get('blocks', 0)} blocks")

        if float(prim_res) < tol:
            break

    if return_info:
        return Q_list, H, B, w_list, Y, info
    return Q_list, H, B, w_list, Y