"""
pino_bench.py — optimizer comparison (Adam vs Muon vs SOAP) for PINO
(Li, Zheng et al., "Physics-Informed Neural Operator", arXiv:2111.03794).

Reproduces, at reduced scale, the Table 1 setting of the paper: an FNO is
trained with a data loss on COARSE data plus a PDE-residual loss on a FINER
grid, then evaluated at 1x / 2x / 4x the data resolution.  The same protocol is
run with three optimizers on three PDEs (Burgers, Darcy, Kolmogorov flow).

Everything is self-contained: datasets are generated here (no downloads),
the FNO, the PINO losses, and the Muon / SOAP optimizers are implemented here.

Author of this file: Claude (for Enio de Sousa Santos).  Not the paper's code.
"""
from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rel_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of ||pred - true|| / ||true|| (per sample)."""
    b = pred.shape[0]
    diff = (pred - true).reshape(b, -1).norm(dim=1)
    nrm = true.reshape(b, -1).norm(dim=1).clamp_min(1e-12)
    return (diff / nrm).mean()


# =============================================================================
# 1. DATA GENERATION
# =============================================================================

# ---------------------------------------------------------------- Burgers ---
# u_t + (u^2/2)_x = nu u_xx on x in [0,1) periodic, t in [0,1], nu = 0.01,
# u0 ~ N(0, 625 (-Delta + 25 I)^{-2})   (paper, Appendix A.1)

def grf_1d_periodic(n_samples, n, alpha=2.0, tau=5.0, sigma=25.0, gen=None, device="cpu"):
    """GRF on [0,1) periodic with covariance sigma^2 (-Delta + tau^2)^(-alpha); zero mean mode."""
    k = torch.arange(n // 2 + 1, device=device, dtype=torch.float64)
    std = sigma * ((2 * math.pi * k) ** 2 + tau ** 2) ** (-alpha / 2)
    std[0] = 0.0
    re = torch.randn(n_samples, n // 2 + 1, generator=gen, dtype=torch.float64).to(device)
    im = torch.randn(n_samples, n // 2 + 1, generator=gen, dtype=torch.float64).to(device)
    coef = std * (re + 1j * im) / math.sqrt(2)
    coef[:, -1] = coef[:, -1].real  # Nyquist real (n even)
    u = torch.fft.irfft(coef * n, n=n, dim=-1)
    return u.float()


@torch.no_grad()
def burgers_solve(u0: torch.Tensor, nu=0.01, T=1.0, dt=2e-4, n_save=101):
    """Pseudo-spectral, 2/3 dealiasing, integrating-factor RK4. u0: (B, n) on [0,1).
    Returns (B, n, n_save) at t = linspace(0, T, n_save)."""
    B, n = u0.shape
    dev = u0.device
    dtype = torch.float64
    k = 2 * math.pi * torch.fft.rfftfreq(n, d=1.0 / n).to(dev, dtype)
    kmax = k.abs().max()
    dealias = (k.abs() <= (2.0 / 3.0) * kmax).to(dtype)
    L = -nu * k ** 2
    steps_total = int(round(T / dt))
    save_every = steps_total // (n_save - 1)
    assert save_every * (n_save - 1) == steps_total, "dt must divide T/(n_save-1)"
    E = torch.exp(L * dt)
    E2 = torch.exp(L * dt / 2)

    def N(vh):
        u = torch.fft.irfft(vh, n=n, dim=-1)
        return -0.5j * k * torch.fft.rfft(u * u, dim=-1) * dealias

    vh = torch.fft.rfft(u0.to(dtype), dim=-1)
    out = torch.empty(B, n, n_save, dtype=torch.float32, device=dev)
    out[:, :, 0] = u0
    for s in range(1, steps_total + 1):
        k1 = dt * N(vh)
        k2 = dt * N(E2 * (vh + k1 / 2))
        k3 = dt * N(E2 * vh + k2 / 2)
        k4 = dt * N(E * vh + E2 * k3)
        vh = E * vh + (E * k1 + 2 * E2 * (k2 + k3) + k4) / 6
        if s % save_every == 0:
            out[:, :, s // save_every] = torch.fft.irfft(vh, n=n, dim=-1).float()
    return out


def make_burgers_data(cfg, seed=0, device=DEVICE, verbose=True):
    g = cfg["gen"]
    gen = torch.Generator().manual_seed(1000 + seed)
    n_total = cfg["n_train"] + cfg["n_val"] + cfg["n_test"]
    n_gen, n_store, nt = g["nx_gen"], g["nx_store"], g["nt_store"]
    stride = n_gen // n_store
    u0_all, u_all = [], []
    t0 = time.time()
    for i in range(0, n_total, g["batch"]):
        b = min(g["batch"], n_total - i)
        u0 = grf_1d_periodic(b, n_gen, gen=gen, device=device)
        sol = burgers_solve(u0, nu=cfg["nu"], dt=g["dt"], n_save=nt)
        u0_all.append(u0[:, ::stride].cpu())
        u_all.append(sol[:, ::stride, :].cpu())
        if verbose:
            print(f"  burgers: {i + b}/{n_total} solved ({time.time() - t0:.1f}s)")
    u0 = torch.cat(u0_all)
    u = torch.cat(u_all)
    assert torch.isfinite(u).all(), "Burgers solver produced non-finite values"
    a, b_ = cfg["n_train"], cfg["n_train"] + cfg["n_val"]
    return {"u0": u0, "u": u, "split": (a, b_)}  # stored at (nx_store, nt_store)


# ------------------------------------------------------------------ Darcy ---
# -div(a grad u) = 1 on (0,1)^2, u = 0 on the boundary,
# a = psi(GRF N(0,(-Delta+9I)^{-2})), psi = 12 if >= 0 else 3  (Appendix A.2)

def grf_darcy_coeff(n_samples, s, K=48, tau=3.0, alpha=2.0, rng=None):
    """Cosine-series GRF on [0,1]^2 (Neumann eigenfunctions), thresholded to {12, 3}."""
    rng = rng or np.random.default_rng(0)
    x = np.linspace(0, 1, s)
    kk = np.arange(K)
    C = np.cos(np.pi * np.outer(x, kk))  # (s, K)
    lam = np.pi ** 2 * (kk[:, None] ** 2 + kk[None, :] ** 2)
    std = (lam + tau ** 2) ** (-alpha / 2)
    out = np.empty((n_samples, s, s), dtype=np.float64)
    for i in range(n_samples):
        xi = rng.standard_normal((K, K)) * std
        f = C @ xi @ C.T
        out[i] = np.where(f >= 0, 12.0, 3.0)
    return out


def darcy_solve_one(a: np.ndarray) -> np.ndarray:
    """5-point FD with harmonic-mean face coefficients; zero Dirichlet; f = 1."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    s = a.shape[0]
    h = 1.0 / (s - 1)
    n = s - 2
    hm = lambda p, q: 2 * p * q / (p + q)
    A = a
    ae = hm(A[1:-1, 1:-1], A[2:, 1:-1])   # face (i+1/2, j)
    aw = hm(A[1:-1, 1:-1], A[:-2, 1:-1])  # face (i-1/2, j)
    an = hm(A[1:-1, 1:-1], A[1:-1, 2:])   # face (i, j+1/2)
    as_ = hm(A[1:-1, 1:-1], A[1:-1, :-2])  # face (i, j-1/2)
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [(ae + aw + an + as_).ravel()]
    # neighbours (only interior ones; boundary values are zero)
    rows.append(idx[:-1, :].ravel()); cols.append(idx[1:, :].ravel()); vals.append(-ae[:-1, :].ravel())
    rows.append(idx[1:, :].ravel()); cols.append(idx[:-1, :].ravel()); vals.append(-aw[1:, :].ravel())
    rows.append(idx[:, :-1].ravel()); cols.append(idx[:, 1:].ravel()); vals.append(-an[:, :-1].ravel())
    rows.append(idx[:, 1:].ravel()); cols.append(idx[:, :-1].ravel()); vals.append(-as_[:, 1:].ravel())
    M = sp.csr_matrix((np.concatenate(vals) / h ** 2, (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    u_int = spla.spsolve(M.tocsc(), np.ones(n * n))
    u = np.zeros((s, s))
    u[1:-1, 1:-1] = u_int.reshape(n, n)
    return u


def make_darcy_data(cfg, seed=0, verbose=True, n_workers=None):
    g = cfg["gen"]
    rng = np.random.default_rng(2000 + seed)
    n_total = cfg["n_train"] + cfg["n_val"] + cfg["n_test"]
    s = g["s_gen"]
    a = grf_darcy_coeff(n_total, s, rng=rng)
    t0 = time.time()
    n_workers = n_workers or max(1, os.cpu_count() or 1)
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_workers) as ex:
            us = []
            for i, u in enumerate(ex.map(darcy_solve_one, list(a), chunksize=8)):
                us.append(u)
                if verbose and (i + 1) % 100 == 0:
                    print(f"  darcy: {i + 1}/{n_total} solved ({time.time() - t0:.1f}s)")
    else:
        us = []
        for i in range(n_total):
            us.append(darcy_solve_one(a[i]))
            if verbose and (i + 1) % 100 == 0:
                print(f"  darcy: {i + 1}/{n_total} solved ({time.time() - t0:.1f}s)")
    u = torch.tensor(np.stack(us), dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.float32)
    sa, sb = cfg["n_train"], cfg["n_train"] + cfg["n_val"]
    return {"a": a, "u": u, "split": (sa, sb)}  # stored at s_gen


# ------------------------------------------------------------- Kolmogorov ---
# w_t + u.grad w = (1/Re) Lap w + f,  f = -4 cos(4y), on [0,2pi]^2, Re = 500,
# initial conditions sampled from the attractor (Section 4, A.4)

def grf_2d_torus(n_samples, n, gen=None, device="cpu"):
    """N(0, 7^{3/2} (-Delta + 49 I)^{-2.5}) on [0,2pi]^2 (used only to start the spin-up)."""
    kx = torch.fft.fftfreq(n, d=1.0 / n).to(device)
    ky = torch.fft.rfftfreq(n, d=1.0 / n).to(device)
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    std = 7 ** 0.75 * (K2 + 49.0) ** (-1.25)
    std[0, 0] = 0
    z = torch.randn(n_samples, n, n // 2 + 1, 2, generator=gen).to(device)
    coef = std * torch.complex(z[..., 0], z[..., 1]) / math.sqrt(2)
    return torch.fft.irfft2(coef * n * n, s=(n, n)).float()


class NSSolver:
    """Pseudo-spectral vorticity solver on [0,2pi]^2, 2/3 dealiasing, IF-RK4."""

    def __init__(self, n, Re=500.0, dt=1e-3, device="cpu"):
        self.n, self.nu, self.dt = n, 1.0 / Re, dt
        dtype = torch.float64
        self.kx = torch.fft.fftfreq(n, d=1.0 / n).to(device, dtype)[:, None]
        self.ky = torch.fft.rfftfreq(n, d=1.0 / n).to(device, dtype)[None, :]
        self.lap = self.kx ** 2 + self.ky ** 2
        lap_inv = self.lap.clone(); lap_inv[0, 0] = 1.0
        self.lap_inv = 1.0 / lap_inv
        kmax = n // 2
        self.dealias = ((self.kx.abs() <= 2 * kmax / 3) & (self.ky.abs() <= 2 * kmax / 3)).to(dtype)
        y = torch.arange(n, device=device, dtype=dtype) * 2 * math.pi / n
        f = (-4 * torch.cos(4 * y))[None, :].expand(n, n)
        self.fh = torch.fft.rfft2(f)
        L = -self.nu * self.lap
        self.E = torch.exp(L * dt)
        self.E2 = torch.exp(L * dt / 2)

    def N(self, wh):
        psih = wh * self.lap_inv
        psih = psih.clone(); psih[..., 0, 0] = 0
        n = self.n
        u = torch.fft.irfft2(1j * self.ky * psih, s=(n, n))
        v = torch.fft.irfft2(-1j * self.kx * psih, s=(n, n))
        wx = torch.fft.irfft2(1j * self.kx * wh, s=(n, n))
        wy = torch.fft.irfft2(1j * self.ky * wh, s=(n, n))
        return -torch.fft.rfft2(u * wx + v * wy) * self.dealias + self.fh

    def step(self, wh):
        dt, E, E2, N = self.dt, self.E, self.E2, self.N
        k1 = dt * N(wh)
        k2 = dt * N(E2 * (wh + k1 / 2))
        k3 = dt * N(E2 * wh + k2 / 2)
        k4 = dt * N(E * wh + E2 * k3)
        return E * wh + (E * k1 + 2 * E2 * (k2 + k3) + k4) / 6


@torch.no_grad()
def make_kf_data(cfg, seed=0, device=DEVICE, verbose=True):
    """Spin up n_traj trajectories to the attractor, then take a snapshot every
    `snap_every` time units; for each snapshot record the window [t, t+T]."""
    g = cfg["gen"]
    n, T, nt = g["n_gen"], cfg["T"], g["nt_store"]
    sub = g["substeps"]
    dt = T / (nt - 1) / sub
    solver = NSSolver(n, Re=cfg["Re"], dt=dt, device=device)
    n_eval = cfg["n_val"] + cfg["n_test"]
    n_needed = cfg["n_pde"] + n_eval
    n_traj = g["n_traj"]
    snaps = math.ceil(n_needed / n_traj)
    # evaluation samples come from their own trajectories (no train/test leakage)
    k_eval = math.ceil(n_eval / snaps)
    assert (n_traj - k_eval) * snaps >= cfg["n_pde"], "increase n_traj"
    gen = torch.Generator().manual_seed(3000 + seed)
    w0 = grf_2d_torus(n_traj, n, gen=gen, device=device)
    wh = torch.fft.rfft2(w0.double())
    t0 = time.time()
    spin_steps = int(round(g["t_spin"] / dt))
    for s in range(spin_steps):
        wh = solver.step(wh)
        if verbose and (s + 1) % max(1, spin_steps // 5) == 0:
            wmax = torch.fft.irfft2(wh, s=(n, n)).abs().max().item()
            print(f"  kf spin-up {s + 1}/{spin_steps}  max|w|={wmax:.2f}  ({time.time() - t0:.1f}s)")
    gap_steps = int(round(g["snap_every"] / dt))
    win_steps = sub * (nt - 1)
    assert gap_steps > win_steps
    store = g["n_store"]
    sx = n // store
    n_dr, nt_dr = cfg["train"]["data_res"]
    sdx, sdt = store // n_dr, (nt - 1) // (nt_dr - 1)
    pool_w0, pool_win, eval_win = [], [], []
    for k in range(snaps):
        frames = [torch.fft.irfft2(wh, s=(n, n)).float()]
        for s in range(1, gap_steps + 1):
            wh = solver.step(wh)
            if s <= win_steps and s % sub == 0:
                frames.append(torch.fft.irfft2(wh, s=(n, n)).float())
        win = torch.stack(frames, dim=-1)[:, ::sx, ::sx, :].cpu()  # (n_traj, store, store, nt)
        assert torch.isfinite(win).all(), "NS solver blew up; lower dt or raise n_gen"
        eval_win.append(win[:k_eval])
        pool_w0.append(win[k_eval:, :, :, 0].clone())
        pool_win.append(win[k_eval:, ::sdx, ::sdx, ::sdt].clone())  # at data resolution
        del win, frames
        if verbose:
            print(f"  kf snapshots {k + 1}/{snaps}  ({time.time() - t0:.1f}s)")
    w0 = torch.cat(pool_w0)
    perm = torch.randperm(w0.shape[0], generator=gen)[: cfg["n_pde"]]
    w0 = w0[perm]
    w_data = torch.cat(pool_win)[perm[: cfg["n_data"]]]
    w_eval = torch.cat(eval_win)
    w_eval = w_eval[torch.randperm(w_eval.shape[0], generator=gen)[:n_eval]]
    # w0: PDE-loss pool (first n_data of it also have w_data); w_eval: val then test
    return {"w0": w0, "w_data": w_data, "w_eval": w_eval, "n_val": cfg["n_val"]}


# =============================================================================
# 2. MODEL: Fourier neural operator (resolution-flexible)
# =============================================================================

class SpectralConv(nn.Module):
    """FNO spectral layer in d dims. Weights are stored as REAL tensors
    (cin, cout, *modes, 2) so every optimizer sees ordinary real parameters."""

    def __init__(self, cin, cout, modes):
        super().__init__()
        self.cin, self.cout, self.modes = cin, cout, tuple(modes)
        self.d = len(modes)
        scale = 1.0 / (cin * cout)
        self.weights = nn.ParameterList(
            [nn.Parameter(scale * torch.rand(cin, cout, *modes, 2)) for _ in range(2 ** (self.d - 1))]
        )
        letters = "xyz"[: self.d]
        self.eq = f"bi{letters},io{letters}->bo{letters}"

    def forward(self, x):
        B = x.shape[0]
        dims = tuple(range(-self.d, 0))
        sizes = x.shape[-self.d:]
        xf = torch.fft.rfftn(x, dim=dims)
        out = torch.zeros(B, self.cout, *xf.shape[2:], dtype=xf.dtype, device=x.device)
        m_eff = [min(m, n // 2) for m, n in zip(self.modes[:-1], sizes[:-1])]
        m_last = min(self.modes[-1], sizes[-1] // 2 + 1)
        for c, w in enumerate(self.weights):
            wc = torch.view_as_complex(w)
            xs, ws = [slice(None), slice(None)], [slice(None), slice(None)]
            for j, (m, n) in enumerate(zip(m_eff, sizes[:-1])):
                neg = (c >> j) & 1
                xs.append(slice(n - m, n) if neg else slice(0, m))
                ws.append(slice(self.modes[j] - m, self.modes[j]) if neg else slice(0, m))
            xs.append(slice(0, m_last)); ws.append(slice(0, m_last))
            out[tuple(xs)] = torch.einsum(self.eq, xf[tuple(xs)], wc[tuple(ws)])
        return torch.fft.irfftn(out, s=sizes, dim=dims)


class FNO(nn.Module):
    """G = Q o (W_L + K_L) o ... o sigma(W_1 + K_1) o P, channels-last input/output."""

    def __init__(self, in_ch, out_ch, modes, width=64, n_layers=4, fc_dim=128, pad_dims=(), pad_frac=0.25,
                 pad_mode="points"):
        super().__init__()
        self.d = len(modes)
        self.pad_dims, self.pad_frac, self.pad_mode = tuple(pad_dims), pad_frac, pad_mode
        self.lift = nn.Linear(in_ch, width)
        self.spectral = nn.ModuleList([SpectralConv(width, width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.proj1 = nn.Linear(width, fc_dim)
        self.proj2 = nn.Linear(fc_dim, out_ch)

    def forward(self, x):  # x: (B, n1, ..., nd, in_ch)
        x = self.lift(x)
        x = x.movedim(-1, 1)  # (B, C, n1..nd)
        sizes = x.shape[2:]
        pads = [0] * (2 * self.d)
        for j in self.pad_dims:  # zero-padding = Fourier continuation on non-periodic dims
            if self.pad_mode == "period":
                # grid includes both endpoints (spacing h = 1/(n-1)); the periodic box then has length
                # (n + p) h = 1 + (p + 1) h, which equals 1 + pad_frac at EVERY resolution iff
                # p = pad_frac (n - 1) - 1.  Exact when pad_frac (n - 1) is an integer.
                p = max(0, int(round(self.pad_frac * (sizes[j] - 1))) - 1)
            else:  # "points": fraction of grid points, rounded up (box length varies with resolution)
                p = int(math.ceil(self.pad_frac * sizes[j]))
            pads[2 * (self.d - 1 - j) + 1] = p
        if any(pads):
            x = F.pad(x, pads)
        L = len(self.spectral)
        for l, (K, W) in enumerate(zip(self.spectral, self.pointwise)):
            x = K(x) + torch.einsum("bc...,oc->bo...", x, W.weight) + W.bias.view(1, -1, *([1] * self.d))
            if l < L - 1:
                x = F.gelu(x)
        if any(pads):
            x = x[(slice(None), slice(None)) + tuple(slice(0, n) for n in sizes)]
        x = x.movedim(1, -1)
        return self.proj2(F.gelu(self.proj1(x)))


# =============================================================================
# 3. OPTIMIZERS
# =============================================================================

def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate polar factor U V^H of G (..., m, n), real or complex (batched).
    Quintic iteration from Keller Jordan's Muon; X^H used so it works for complex."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mH
    X = X / (torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mH
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    if transposed:
        X = X.mH
    return X


class Muon(torch.optim.Optimizer):
    """Muon with an internal AdamW fallback.

    Each param group carries kind in {"muon", "muon_spectral", "adam"}:
      muon          2-D weight (out, in): momentum + Nesterov, orthogonalized by
                    Newton-Schulz, scaled by 0.2*sqrt(max(out,in)) so its RMS
                    matches AdamW (Moonshot's rule; one lr for both kinds).
      muon_spectral FNO spectral weight (cin, cout, *modes, 2): each Fourier mode's
                    complex cin x cout matrix is orthogonalized (batched, complex NS).
      adam          everything else (lifting, last projection, biases, and the
                    spectral weights unless muon_spectral is chosen)."""

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        betas=betas, eps=eps, weight_decay=weight_decay, kind="adam")
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, wd, kind = group["lr"], group["weight_decay"], group["kind"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if kind == "adam":
                    if not st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    b1, b2 = group["betas"]
                    st["step"] += 1
                    st["m"].lerp_(g, 1 - b1)
                    st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                    mhat = st["m"] / (1 - b1 ** st["step"])
                    vhat = st["v"] / (1 - b2 ** st["step"])
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(mhat, vhat.sqrt().add_(group["eps"]), value=-lr)
                    continue
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(group["momentum"]).add_(g)
                u = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                if kind == "muon":
                    U = u.reshape(u.shape[0], -1)
                    O = newton_schulz(U, group["ns_steps"]).reshape(p.shape)
                    scale = 0.2 * math.sqrt(max(U.shape))
                elif kind == "muon_spectral":
                    uc = torch.view_as_complex(u.contiguous())  # (cin, cout, *modes)
                    cin, cout = uc.shape[:2]
                    nm = uc.dim() - 2
                    Ub = uc.permute(*range(2, 2 + nm), 0, 1).reshape(-1, cin, cout)
                    Ob = newton_schulz(Ub, group["ns_steps"])
                    Oc = Ob.reshape(*uc.shape[2:], cin, cout).permute(nm, nm + 1, *range(nm))
                    O = torch.view_as_real(Oc.contiguous())
                    scale = 0.2 * math.sqrt(max(cin, cout))
                else:
                    raise ValueError(kind)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(O, alpha=-lr * scale)


class SOAP(torch.optim.Optimizer):
    """SOAP (Vyas et al., 2024, arXiv:2409.11321): Adam run in the eigenbasis of
    Shampoo's per-dimension preconditioners, eigenbasis refreshed every
    `precondition_frequency` steps by one power-iteration + QR.  Written after
    the authors' reference implementation; dims larger than `max_precond_dim`
    and 1-D tensors are left unrotated (plain Adam along those dims)."""

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=-1, eps=1e-8,
                 weight_decay=0.0, precondition_frequency=10, max_precond_dim=10000,
                 precondition_1d=False, correct_bias=True):
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                        weight_decay=weight_decay, precondition_frequency=precondition_frequency,
                        max_precond_dim=max_precond_dim, precondition_1d=precondition_1d,
                        correct_bias=correct_bias)
        super().__init__(params, defaults)

    @staticmethod
    def _project(x, Q, back=False):
        for q in Q:
            if q is not None:
                x = torch.tensordot(x, q, dims=[[0], [1 if back else 0]])
            else:
                x = x.permute(*range(1, x.dim()), 0)
        return x

    @staticmethod
    def _eig(GG):
        Q = []
        for m in GG:
            if m is None:
                Q.append(None); continue
            try:
                _, q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
            except Exception:
                _, q = torch.linalg.eigh(m.double() + 1e-30 * torch.eye(m.shape[0], device=m.device, dtype=torch.float64))
                q = q.to(m.dtype)
            Q.append(torch.flip(q, [1]))
        return Q

    def _qr_refresh(self, st):
        exp_avg_sq = st["exp_avg_sq"]
        newQ = []
        for ind, (m, q) in enumerate(zip(st["GG"], st["Q"])):
            if m is None:
                newQ.append(None); continue
            est = torch.diag(q.T @ m @ q)
            order = torch.argsort(est, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, order)
            q = q[:, order]
            newQ.append(torch.linalg.qr(m @ q)[0])
        st["exp_avg_sq"] = exp_avg_sq
        st["Q"] = newQ

    def _update_precond(self, g, st, group):
        beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]
        for idx, m in enumerate(st["GG"]):
            if m is None:
                continue
            other = [i for i in range(g.dim()) if i != idx]
            outer = torch.tensordot(g, g, dims=[other, other])
            m.lerp_(outer, 1 - beta)
        if st["Q"] is None:
            st["Q"] = self._eig(st["GG"])
        elif st["step"] > 0 and st["step"] % group["precondition_frequency"] == 0:
            self._qr_refresh(st)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(g)
                    st["exp_avg_sq"] = torch.zeros_like(g)
                    st["GG"] = [None if ((g.dim() == 1 and not group["precondition_1d"]) or sh > group["max_precond_dim"])
                                else torch.zeros(sh, sh, device=g.device, dtype=g.dtype) for sh in g.shape]
                    st["Q"] = None
                    self._update_precond(g, st, group)
                    continue  # first step only builds the preconditioner (as in the reference)
                gp = self._project(g, st["Q"])
                b1, b2 = group["betas"]
                st["step"] += 1
                st["exp_avg"].lerp_(g, 1 - b1)
                st["exp_avg_sq"].mul_(b2).add_(gp.square(), alpha=1 - b2)
                denom = st["exp_avg_sq"].sqrt().add_(group["eps"])
                mp = self._project(st["exp_avg"], st["Q"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    step_size = step_size * math.sqrt(1 - b2 ** st["step"]) / (1 - b1 ** st["step"])
                p.add_(self._project(mp / denom, st["Q"], back=True), alpha=-step_size)
                if group["weight_decay"] > 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
                self._update_precond(g, st, group)


OPTIMIZERS = ("adam", "muon", "muon_spectral", "soap")


def build_optimizer(name: str, model: FNO, lr: float, opt_cfg: dict | None = None):
    opt_cfg = opt_cfg or {}
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **opt_cfg.get("adam", {}))
    if name == "soap":
        return SOAP(model.parameters(), lr=lr, **opt_cfg.get("soap", {}))
    if name in ("muon", "muon_spectral"):
        hidden = [W.weight for W in model.pointwise] + [model.proj1.weight]
        spectral = [w for K in model.spectral for w in K.weights]
        hid_ids = {id(p) for p in hidden}
        spec_ids = {id(p) for p in spectral}
        rest = [p for p in model.parameters() if id(p) not in hid_ids and id(p) not in spec_ids]
        groups = [dict(params=hidden, kind="muon")]
        if name == "muon_spectral":
            groups.append(dict(params=spectral, kind="muon_spectral"))
            groups.append(dict(params=rest, kind="adam"))
        else:
            groups.append(dict(params=spectral + rest, kind="adam"))
        return Muon(groups, lr=lr, **opt_cfg.get("muon", {}))
    raise ValueError(f"unknown optimizer {name}")


# =============================================================================
# 4. PROBLEMS: inputs, PINO losses, evaluation
# =============================================================================

def _down(n_from, n_to):
    assert (n_from - 1) % (n_to - 1) == 0 or n_from % n_to == 0, (n_from, n_to)


class Burgers:
    """Grid layout (B, nx, nt); x in [0,1) periodic, t in [0,1]."""
    name = "burgers"

    def __init__(self, cfg, data):
        self.cfg, self.nu = cfg, cfg["nu"]
        self.u0, self.u = data["u0"], data["u"]  # (N, nx_s), (N, nx_s, nt_s)
        self.nx_s, self.nt_s = self.u.shape[1], self.u.shape[2]
        self.a, self.b = data["split"]

    def model(self):
        m = self.cfg["model"]
        return FNO(3, 1, m["modes"], m["width"], m["layers"], pad_dims=(1,), pad_frac=m["pad_frac"],
                   pad_mode=m.get("pad_mode", "points"))

    def sub(self, res):
        nx, nt = res
        sx, st = self.nx_s // nx, (self.nt_s - 1) // (nt - 1)
        assert sx * nx == self.nx_s and st * (nt - 1) == self.nt_s - 1, f"res {res} not a sub-grid of {(self.nx_s, self.nt_s)}"
        return sx, st

    def inputs(self, u0, nt):
        B, nx = u0.shape
        x = torch.arange(nx, device=u0.device) / nx
        t = torch.linspace(0, 1, nt, device=u0.device)
        X, Tg = torch.meshgrid(x, t, indexing="ij")
        return torch.stack([u0[:, :, None].expand(B, nx, nt), X.expand(B, nx, nt), Tg.expand(B, nx, nt)], -1)

    def get(self, idx, res):
        sx, st = self.sub(res)
        u0 = self.u0[idx][:, ::sx].to(DEVICE)
        u = self.u[idx][:, ::sx, ::st].to(DEVICE)
        return u0, u

    def predict(self, model, u0, nt):
        return model(self.inputs(u0, nt))[..., 0]

    def pde_losses(self, u, u0):
        B, nx, nt = u.shape
        dt = 1.0 / (nt - 1)
        k = 2 * math.pi * torch.fft.fftfreq(nx, d=1.0 / nx).to(u.device)[None, :, None]
        uh = torch.fft.fft(u, dim=1)
        ux = torch.fft.ifft(1j * k * uh, dim=1).real
        uxx = torch.fft.ifft(-(k ** 2) * uh, dim=1).real
        ut = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dt)
        r = ut + (u * ux - self.nu * uxx)[:, :, 1:-1]
        return {"f": (r ** 2).mean(), "ic": rel_l2(u[:, :, 0], u0)}

    def pde_pred_and_losses(self, model, idx):
        res = self.cfg["train"]["pde_res"]
        u0, _ = self.get(idx, res)
        pred = self.predict(model, u0, res[1])
        return pred, self.pde_losses(pred, u0)

    def train_step(self, model, data_idx, pde_idx, w):
        tc = self.cfg["train"]
        losses = {}
        total = 0.0
        if w["data"] > 0 and data_idx is not None:
            u0, u = self.get(data_idx, tc["data_res"])
            losses["data"] = rel_l2(self.predict(model, u0, u.shape[2]), u)
            total = total + w["data"] * losses["data"]
        if (w["f"] > 0 or w["ic"] > 0) and pde_idx is not None:
            _, pl = self.pde_pred_and_losses(model, pde_idx)
            losses.update(pl)
            total = total + w["f"] * pl["f"] + w["ic"] * pl["ic"]
        return total, losses

    @torch.no_grad()
    def evaluate(self, model, idx, res, bs=50):
        errs = []
        for i in range(0, len(idx), bs):
            u0, u = self.get(idx[i:i + bs], res)
            errs.append(rel_l2(self.predict(model, u0, u.shape[2]), u).item() * len(u))
        return sum(errs) / len(idx)

    @torch.no_grad()
    def truth_residual(self, res):
        idx = torch.arange(min(50, self.a))
        u0, u = self.get(idx, res)
        return {k: v.item() for k, v in self.pde_losses(u, u0).items()}


def _dst1(x, dim):
    """Unnormalized DST-I along `dim`: S_k = sum_j x_j sin(pi j k / (N+1)), j, k = 1..N (via FFT)."""
    x = x.movedim(dim, -1)
    N = x.shape[-1]
    z = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    y = torch.cat([z, x, z, -x.flip(-1)], -1)              # odd extension, length 2N + 2
    S = -torch.fft.fft(y, dim=-1).imag[..., 1:N + 1] / 2
    return S.movedim(-1, dim)


def hm1_norm(r):
    """Discrete H^{-1} norm (per sample, then mean) of r on the interior of a uniform grid with zero
    Dirichlet data: sqrt(r^T (-Lap_h)^{-1} r) up to a grid-independent factor (cancels in ratios)."""
    B, N, _ = r.shape
    S = _dst1(_dst1(r, 1), 2)
    k = torch.arange(1, N + 1, dtype=r.dtype, device=r.device)
    lam1 = 2 - 2 * torch.cos(math.pi * k / (N + 1))           # eigenvalues of -Lap_h times h^2
    lam = lam1[:, None] + lam1[None, :]
    val = (S ** 2 / lam).reshape(B, -1).sum(1) * (2.0 / (N + 1)) ** 2
    return val.sqrt().mean()


class Darcy:
    """Grid (B, s, s) on [0,1]^2 including the boundary."""
    name = "darcy"

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.A, self.U = data["a"], data["u"]
        self.s = self.A.shape[1]
        self.a, self.b = data["split"]

    def model(self):
        m = self.cfg["model"]
        return FNO(3, 1, m["modes"], m["width"], m["layers"], pad_dims=(0, 1), pad_frac=m["pad_frac"],
                   pad_mode=m.get("pad_mode", "points"))

    def get(self, idx, s):
        st = (self.s - 1) // (s - 1)
        assert st * (s - 1) == self.s - 1, f"{s} is not a sub-grid of {self.s}"
        return self.A[idx][:, ::st, ::st].to(DEVICE), self.U[idx][:, ::st, ::st].to(DEVICE)

    def predict(self, model, a):
        B, s, _ = a.shape
        x = torch.linspace(0, 1, s, device=a.device)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        inp = torch.stack([a, X.expand(B, s, s), Y.expand(B, s, s)], -1)
        moll = self.cfg["mollifier"] * torch.sin(math.pi * X) * torch.sin(math.pi * Y)
        return model(inp)[..., 0] * moll

    @staticmethod
    def residual(u, a):
        """PINO's strong-form FD residual -(a u_x)_x - (a u_y)_y (central differences)."""
        s = u.shape[1]
        h = 1.0 / (s - 1)
        ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * h)
        uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * h)
        ai = a[:, 1:-1, 1:-1]
        aux, auy = ai * ux, ai * uy
        auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * h)
        auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * h)
        return -(auxx + auyy)

    @staticmethod
    def residual_conservative(u, a):
        """Flux-form residual with harmonic-mean face coefficients (the scheme that generated
        the data).  Square and well-posed, unlike the paper's wide-stencil form, whose exact
        least-squares minimizer can be ~70% away from the true solution (checked in tests)."""
        s = u.shape[1]
        h = 1.0 / (s - 1)
        hm = lambda p, q: 2 * p * q / (p + q)
        ax = hm(a[:, 1:, 1:-1], a[:, :-1, 1:-1])
        ay = hm(a[:, 1:-1, 1:], a[:, 1:-1, :-1])
        Fx = ax * (u[:, 1:, 1:-1] - u[:, :-1, 1:-1]) / h
        Fy = ay * (u[:, 1:-1, 1:] - u[:, 1:-1, :-1]) / h
        return -((Fx[:, 1:, :] - Fx[:, :-1, :]) / h + (Fy[:, :, 1:] - Fy[:, :, :-1]) / h)

    def pde_losses(self, u, a):
        if self.cfg.get("residual", "conservative") == "paper":
            Du = self.residual(u, a)
        else:
            Du = self.residual_conservative(u, a)
        if self.cfg.get("residual_norm", "l2") == "hm1":
            # relative discrete H^{-1} norm of the residual Du - 1 (Dirichlet); same minimizer as L2,
            # but equivalent to the energy-norm error by coercivity (no h^-2 amplification)
            return {"f": hm1_norm(Du - 1.0) / hm1_norm(torch.ones_like(Du))}
        return {"f": rel_l2(Du, torch.ones_like(Du))}

    def train_step(self, model, data_idx, pde_idx, w):
        tc = self.cfg["train"]
        losses, total = {}, 0.0
        if w["data"] > 0 and data_idx is not None:
            a, u = self.get(data_idx, tc["data_res"])
            losses["data"] = rel_l2(self.predict(model, a), u)
            total = total + w["data"] * losses["data"]
        if w["f"] > 0 and pde_idx is not None:
            _, pl = self.pde_pred_and_losses(model, pde_idx)
            losses.update(pl)
            total = total + w["f"] * pl["f"]
        return total, losses

    def pde_pred_and_losses(self, model, idx):
        a, _ = self.get(idx, self.cfg["train"]["pde_res"])
        pred = self.predict(model, a)
        return pred, self.pde_losses(pred, a)

    @torch.no_grad()
    def evaluate(self, model, idx, s, bs=50):
        errs = []
        for i in range(0, len(idx), bs):
            a, u = self.get(idx[i:i + bs], s)
            errs.append(rel_l2(self.predict(model, a), u).item() * len(u))
        return sum(errs) / len(idx)

    @torch.no_grad()
    def truth_residual(self, s):
        idx = torch.arange(min(50, self.a))
        a, u = self.get(idx, s)
        return {k: v.item() for k, v in self.pde_losses(u, a).items()}


class Kolmogorov:
    """Grid (B, n, n, nt) on [0,2pi]^2 x [0,T]; first n_data of the PDE pool also have data."""
    name = "kolmogorov"

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.W0 = data["w0"]          # (n_pde, n_s, n_s)           PDE-loss pool
        self.Wd = data["w_data"]      # (n_data, n_d, n_d, nt_d)    coarse training data
        self.We = data["w_eval"]      # (n_val+n_test, n_s, n_s, nt_s)
        self.n_s, self.nt_s = self.We.shape[1], self.We.shape[3]
        # index convention shared with the other problems: [0,a) train pool, [a,b) val, [b,..) test
        self.a = self.W0.shape[0]
        self.b = self.a + data["n_val"]
        self.nu, self.T = 1.0 / cfg["Re"], cfg["T"]

    def model(self):
        m = self.cfg["model"]
        return FNO(4, 1, m["modes"], m["width"], m["layers"], pad_dims=(2,), pad_frac=m["pad_frac"],
                   pad_mode=m.get("pad_mode", "points"))

    def _sub(self, res, n_s, nt_s):
        n, nt = res
        sx, st = n_s // n, (nt_s - 1) // (nt - 1)
        assert sx * n == n_s and st * (nt - 1) == nt_s - 1, f"res {res} not a sub-grid of {(n_s, nt_s)}"
        return sx, st

    def get(self, idx, res, with_solution=True):
        idx = torch.as_tensor(idx)
        if idx.min() >= self.a:  # val / test
            sx, st = self._sub(res, self.n_s, self.nt_s)
            w = self.We[idx - self.a][:, ::sx, ::sx, ::st].to(DEVICE)
            return w[..., 0], w
        if with_solution:  # training data: stored at data resolution
            assert tuple(res) == tuple(self.cfg["train"]["data_res"])
            w = self.Wd[idx].to(DEVICE)
            return w[..., 0], w
        sx, _ = self._sub((res[0], self.nt_s), self.n_s, self.nt_s)
        return self.W0[idx][:, ::sx, ::sx].to(DEVICE), None

    def predict(self, model, w0, nt):
        B, n, _ = w0.shape
        x = torch.arange(n, device=w0.device) * 2 * math.pi / n
        t = torch.linspace(0, self.T, nt, device=w0.device)
        X, Y, Tg = torch.meshgrid(x, x, t, indexing="ij")
        inp = torch.stack([w0[..., None].expand(B, n, n, nt), X.expand(B, n, n, nt) / (2 * math.pi),
                           Y.expand(B, n, n, nt) / (2 * math.pi), Tg.expand(B, n, n, nt) / self.T], -1)
        return model(inp)[..., 0]

    def residual(self, w):
        B, n, _, nt = w.shape
        dt = self.T / (nt - 1)
        dev = w.device
        kx = torch.fft.fftfreq(n, d=1.0 / n).to(dev)[:, None, None]
        ky = torch.fft.rfftfreq(n, d=1.0 / n).to(dev)[None, :, None]
        lap = kx ** 2 + ky ** 2
        lap_inv = torch.where(lap == 0, torch.zeros_like(lap), 1.0 / lap.clamp_min(1e-12))
        wh = torch.fft.rfft2(w, dim=(1, 2))
        psih = wh * lap_inv
        irf = lambda z: torch.fft.irfft2(z, s=(n, n), dim=(1, 2))
        u, v = irf(1j * ky * psih), irf(-1j * kx * psih)
        wx, wy, wlap = irf(1j * kx * wh), irf(1j * ky * wh), irf(-lap * wh)
        A = u * wx + v * wy - self.nu * wlap          # w_t + A = f
        scheme = getattr(self, "cfg", {}).get("time_scheme", "central")
        if scheme == "cn":
            # Crank-Nicolson: (w_{k+1} - w_k)/dt + (A_{k+1} + A_k)/2 = f, k = 0..nt-2.
            # One equation per unknown frame -> well-posed (unique minimizer), 2nd order in time.
            Du = (w[..., 1:] - w[..., :-1]) / dt + 0.5 * (A[..., 1:] + A[..., :-1])
        else:
            # Central differences on interior frames (as in PINO).  Only nt-2 frames of equations
            # for nt-1 unknown frames: the discrete loss has a family of exact minimizers
            # (leapfrog parasitic mode); different ansatzes/optimizers can land on different ones.
            Du = (w[..., 2:] - w[..., :-2]) / (2 * dt) + A[..., 1:-1]
        y = torch.arange(n, device=dev) * 2 * math.pi / n
        f = (-4 * torch.cos(4 * y))[None, None, :, None].expand_as(Du)
        return Du, f

    @staticmethod
    def sobolev_norm(r, s):
        """Per-sample H^{-s} norm in space (then l2 over time) of r (B, n, n, nt) on the 2pi-torus:
        sqrt(sum_t sum_k (1 + |k|^2)^{-s} |r_hat(k, t)|^2), up to a constant factor."""
        n = r.shape[1]
        k = torch.fft.fftfreq(n, d=1.0 / n).to(r.device)
        wgt = (1 + k[:, None] ** 2 + k[None, :] ** 2) ** (-s)
        rh = torch.fft.fft2(r, dim=(1, 2))
        return ((rh.abs() ** 2) * wgt[None, :, :, None]).reshape(len(r), -1).sum(1).sqrt()

    def pde_losses(self, w, w0):
        Du, f = self.residual(w)
        B = len(w)
        s = getattr(self, "cfg", {}).get("residual_sobolev", 0)
        if s:   # relative H^{-s} norm of the residual (s = 1: H^-1, s = 0.5: H^-1/2); same minimizer
            f_rel = (self.sobolev_norm(Du - f, s) / self.sobolev_norm(f, s)).mean()
        else:
            f_rel = ((Du - f).reshape(B, -1).norm(dim=1) / f.reshape(B, -1).norm(dim=1)).mean()
        return {"f": f_rel, "ic": rel_l2(w[..., 0], w0)}

    def train_step(self, model, data_idx, pde_idx, w):
        tc = self.cfg["train"]
        losses, total = {}, 0.0
        if w["data"] > 0 and data_idx is not None:
            w0, wt = self.get(data_idx, tc["data_res"])
            losses["data"] = rel_l2(self.predict(model, w0, wt.shape[3]), wt)
            total = total + w["data"] * losses["data"]
        if (w["f"] > 0 or w["ic"] > 0) and pde_idx is not None:
            _, pl = self.pde_pred_and_losses(model, pde_idx)
            losses.update(pl)
            total = total + w["f"] * pl["f"] + w["ic"] * pl["ic"]
        return total, losses

    def pde_pred_and_losses(self, model, idx):
        res = self.cfg["train"]["pde_res"]
        w0, _ = self.get(idx, res, with_solution=False)
        pred = self.predict(model, w0, res[1])
        return pred, self.pde_losses(pred, w0)

    @torch.no_grad()
    def evaluate(self, model, idx, res, bs=4):
        errs = []
        for i in range(0, len(idx), bs):
            w0, w = self.get(idx[i:i + bs], res)
            errs.append(rel_l2(self.predict(model, w0, w.shape[3]), w).item() * len(w))
        return sum(errs) / len(idx)

    @torch.no_grad()
    def truth_residual(self, res):
        idx = torch.arange(self.a, min(self.a + 8, self.a + self.We.shape[0]))
        w0, w = self.get(idx, res)
        return {k: v.item() for k, v in self.pde_losses(w, w0).items()}


PROBLEMS = {"burgers": Burgers, "darcy": Darcy, "kolmogorov": Kolmogorov, "kolmogorov64": Kolmogorov, "kolmogorov500": Kolmogorov}


def make_problem(pde, cfg, data):
    p = PROBLEMS[pde](cfg, data)
    p.name = pde
    return p


# =============================================================================
# 5. CONFIGURATION PRESETS
# =============================================================================
# Table 1 of the paper (for reference):
#   Burgers  data 32x25,    PDE 128x100,     test 1x/2x/4x
#   Darcy    data 11x11,    PDE 61x61,       test 11 / 61 / 211
#   KF       data 64^2x33,  PDE 256^2x65,    T = 0.125, Re = 500
# "quick" keeps Burgers/Darcy at paper resolution with fewer iterations and runs
# Kolmogorov (Re = 500) at half the paper's PDE resolution (data 32^2x17, PDE 128^2x33,
# test up to 128^2x33); "kolmogorov64" is a cheaper but biased variant (PDE 64^2x33).

def _presets():
    burgers = dict(
        nu=0.01, n_train=800, n_val=100, n_test=200,
        gen=dict(nx_gen=1024, nx_store=128, nt_store=101, dt=2e-4, batch=250),
        model=dict(modes=(16, 16), width=64, layers=4, pad_frac=0.25, pad_mode="points"),  # "period": experimental, see FNO.forward
        train=dict(iters=4000, bs_data=20, bs_pde=20, data_res=(32, 26), pde_res=(128, 101),
                   weights=dict(data=1.0, f=1.0, ic=1.0)),
        test_res=[(32, 26), (64, 51), (128, 101)], val_res=(128, 101),
    )
    darcy = dict(
        mollifier=1e-3, residual="conservative",  # "paper" = PINO's strong-form FD residual
        n_train=1000, n_val=100, n_test=200,
        gen=dict(s_gen=241),
        model=dict(modes=(12, 12), width=64, layers=4, pad_frac=0.25, pad_mode="points"),  # "period": experimental, see FNO.forward
        train=dict(iters=4000, bs_data=20, bs_pde=20, data_res=11, pde_res=61,
                   weights=dict(data=1.0, f=1.0)),
        test_res=[11, 61, 121], val_res=61,
    )
    # Kolmogorov at Re = 500 (the paper's value), reference data generated at 256^2.
    # Drift test on a state ON THE ATTRACTOR (minimize the PDE loss over the space-time field,
    # starting from a 256^2 reference): the minimizer lies ~8% from the truth on a 64^2 x 33
    # grid but ~0.9% on 128^2 x 33.  So the PDE loss must live on 128^2 (the paper used 256^2).
    # (An early-transient state gave 1.8% / 0.27%; attractor states are much rougher.)
    # The truth's raw residual relative to ||f|| is large (~95% on 128^2) because the
    # advection terms are much larger than f; it is not the right consistency measure.
    kf = dict(
        Re=500.0, T=0.125, n_data=400, n_pde=1000, n_val=50, n_test=100,
        gen=dict(n_gen=256, n_store=128, nt_store=33, substeps=4, n_traj=100, t_spin=15.0, snap_every=1.0),
        model=dict(modes=(8, 8, 8), width=32, layers=4, pad_frac=0.25, pad_mode="points"),  # "period": experimental, see FNO.forward
        train=dict(iters=3000, bs_data=4, bs_pde=1, data_res=(32, 17), pde_res=(128, 33),
                   weights=dict(data=5.0, f=1.0, ic=5.0)),
        test_res=[(32, 17), (64, 33), (128, 33)], val_res=(128, 33),
    )
    # Cheap variant: same data, PDE loss on 64^2 x 33 (~4x cheaper per step), but its
    # minimizer sits ~8% from the truth on attractor states, so it biases the operator.
    kf64 = deepcopy(kf)
    kf64["data_key"] = "kolmogorov"  # same dataset as "kolmogorov"
    kf64["train"].update(bs_pde=2, pde_res=(64, 33))
    kf64["val_res"] = (64, 33)
    # Paper-faithful Table 1 Kolmogorov setting (Re = 500, data 64^2x33, PDE 256^2x33).
    # Needs a large GPU (A100-class); far too slow on a T4.
    kf500 = deepcopy(kf)
    kf500.update(Re=500.0, n_data=800, n_pde=2200, n_val=50, n_test=100)
    kf500["gen"].update(n_gen=256, n_store=256)
    kf500["model"].update(width=64)
    kf500["train"].update(iters=10000, bs_data=2, bs_pde=1, data_res=(64, 33), pde_res=(256, 33))
    kf500["test_res"] = [(64, 33), (128, 33), (256, 33)]
    kf500["val_res"] = (128, 33)

    # Phase 2 (instance-wise fine-tuning): per test instance, start from a phase-1 operator
    # and minimize  sum_k w_k * PDE-loss_k  +  anchor * ||G(a) - G0(a)||^2 / ||G0(a)||^2
    # (anchor = 0 is pure PDE-loss fine-tuning; the paper uses the anchor on hard problems).
    burgers["finetune"] = dict(iters=1000, n_instances=20, anchor=0.0, n_sweep_instances=3, sweep_frac=0.5)
    darcy["finetune"] = dict(iters=1000, n_instances=20, anchor=0.0, n_sweep_instances=3, sweep_frac=0.5)
    kf["finetune"] = dict(iters=1000, n_instances=5, anchor=0.0, n_sweep_instances=2, sweep_frac=0.5)
    kf500["finetune"] = dict(iters=2000, n_instances=5, anchor=0.0, n_sweep_instances=2, sweep_frac=0.5)
    kf64["finetune"] = dict(iters=1000, n_instances=5, anchor=0.0, n_sweep_instances=2, sweep_frac=0.5)

    quick = dict(burgers=burgers, darcy=darcy, kolmogorov=kf, kolmogorov64=kf64)

    full = deepcopy(quick)
    full["burgers"]["train"]["iters"] = 20000          # = paper's 500 epochs x 40 batches
    full["darcy"]["train"]["iters"] = 15000
    full["kolmogorov"]["model"]["width"] = 64          # paper width
    full["kolmogorov"]["n_data"], full["kolmogorov"]["n_pde"] = 800, 2200
    full["kolmogorov"]["train"]["iters"] = 10000
    full["kolmogorov500"] = kf500

    smoke = deepcopy(quick)  # tiny CPU check of the whole pipeline
    smoke.pop("kolmogorov64")
    b = smoke["burgers"]
    b.update(n_train=16, n_val=4, n_test=8)
    b["gen"].update(nx_gen=256, dt=1e-3, batch=28)
    b["model"].update(modes=(8, 8), width=16)
    b["train"].update(iters=20, bs_data=4, bs_pde=4)
    d = smoke["darcy"]
    d.update(n_train=16, n_val=4, n_test=8)
    d["gen"].update(s_gen=121)
    d["model"].update(modes=(6, 6), width=16)
    d["train"].update(iters=20, bs_data=4, bs_pde=4)
    d["test_res"] = [11, 61, 121]
    k = smoke["kolmogorov"]
    k.update(n_data=8, n_pde=12, n_val=2, n_test=4)
    k["gen"].update(n_gen=64, n_store=64, n_traj=6, t_spin=1.0, snap_every=0.25)
    k["model"].update(modes=(4, 4, 4), width=8)
    k["train"].update(iters=10, bs_data=2, bs_pde=1, data_res=(16, 17), pde_res=(32, 33))
    k["test_res"] = [(16, 17), (32, 33), (64, 33)]
    k["val_res"] = (32, 33)
    for p in (b, d, k):
        p["finetune"].update(iters=10, n_instances=2, n_sweep_instances=1)
    return dict(smoke=smoke, quick=quick, full=full)


PRESETS = _presets()

# Base learning rates (the sweep multiplies these by LR_MULTS).
BASE_LR = dict(adam=1e-3, muon=1e-3, muon_spectral=1e-3, soap=3e-3)
LR_MULTS = (1 / 3, 1.0, 3.0)


# =============================================================================
# 6. DATA CACHE, TRAINING, EXPERIMENT DRIVER
# =============================================================================

def load_or_make_data(pde: str, cfg: dict, data_dir: str, preset: str, seed: int = 0):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{cfg.get('data_key', pde)}_{preset}_s{seed}.pt")
    if os.path.exists(path):
        return torch.load(path)
    print(f"Generating {pde} data ({preset}) -> {path}")
    t0 = time.time()
    if pde == "burgers":
        data = make_burgers_data(cfg, seed)
    elif pde == "darcy":
        data = make_darcy_data(cfg, seed)
    else:
        data = make_kf_data(cfg, seed)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    torch.save(data, path)
    print(f"  done in {time.time() - t0:.0f}s")
    return data


# ---- pausing / resuming long runs (e.g. on machines with job time limits) ----
DEADLINE = None          # wall-clock time.time() at which running jobs should pause
CKPT_EVERY_S = 900       # save a resume checkpoint at least this often (seconds)


class Pause(Exception):
    """Raised when DEADLINE is reached; the run can be resumed from its checkpoint."""


def _past_deadline():
    return DEADLINE is not None and time.time() >= DEADLINE


def train_one(problem, opt_name: str, lr: float, seed: int, iters: int | None = None,
              n_evals: int = 10, eval_test: bool = True, opt_cfg: dict | None = None,
              log_every: int = 0, schedule: str = "cosine", ckpt_path: str | None = None):
    """Train one PINO model; returns a result dict with curves and final errors.
    With ckpt_path, progress is checkpointed and an interrupted run resumes from it."""
    cfg = problem.cfg
    tc = cfg["train"]
    iters = iters or tc["iters"]
    set_seed(seed)  # same initial weights for every optimizer at a given seed
    model = problem.model().to(DEVICE)
    opt = build_optimizer(opt_name, model, lr, opt_cfg)
    if schedule == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, iters) / iters)))
    else:
        sched = None
    gen = torch.Generator().manual_seed(10_000 + seed)
    n_pool = problem.a  # training pool (PDE instances)
    n_data = cfg.get("n_data", n_pool)  # Kolmogorov: only the first n_data have solutions
    val_idx = torch.arange(problem.a, problem.b)
    test_idx = torch.arange(problem.b, problem.b + cfg["n_test"])
    w = tc["weights"]
    hist = dict(it=[], time=[], loss=[], val=[], test={str(r): [] for r in cfg["test_res"]}, parts=[])
    eval_at = set(int(round(iters * (i + 1) / n_evals)) for i in range(n_evals))
    train_time = 0.0
    diverged = False
    start = 1
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if sched:
            sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"]); hist = ck["hist"]; train_time = ck["train_time"]
        start = ck["it"] + 1
        print(f"    resumed {opt_name} lr={lr:.2e} seed={seed} at iter {start}", flush=True)

    def save_ckpt(it_done):
        tmp = ckpt_path + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict() if sched else None, gen=gen.get_state(),
                        hist=hist, train_time=train_time, it=it_done), tmp)
        os.replace(tmp, ckpt_path)

    last_save = time.time()
    it = start - 1
    for it in range(start, iters + 1):
        model.train()
        sync(); t0 = time.time()
        d_idx = torch.randint(0, n_data, (tc["bs_data"],), generator=gen)
        p_idx = torch.randint(0, n_pool, (tc["bs_pde"],), generator=gen)
        loss, parts = problem.train_step(model, d_idx, p_idx, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        sync(); train_time += time.time() - t0
        lval = loss.item()
        if not math.isfinite(lval):
            diverged = True
            print(f"    [{opt_name} lr={lr:.2e}] diverged at iter {it}")
            break
        if log_every and it % log_every == 0:
            print(f"    it {it:6d}  loss {lval:.4e}  " + "  ".join(f"{k}={v.item():.3e}" for k, v in parts.items()))
        if it in eval_at:
            model.eval()
            hist["it"].append(it); hist["time"].append(train_time); hist["loss"].append(lval)
            hist["parts"].append({k: v.item() for k, v in parts.items()})
            hist["val"].append(problem.evaluate(model, val_idx, cfg["val_res"]))
            if eval_test:
                for r in cfg["test_res"]:
                    hist["test"][str(r)].append(problem.evaluate(model, test_idx, r))
        if ckpt_path and it < iters and (time.time() - last_save > CKPT_EVERY_S or _past_deadline()):
            save_ckpt(it); last_save = time.time()
            if _past_deadline():
                raise Pause(f"{opt_name} lr={lr:.2e} seed={seed} paused at iter {it}/{iters}")
    res = dict(pde=problem.name, opt=opt_name, lr=lr, seed=seed, iters=iters, diverged=diverged,
               train_time=train_time, sec_per_iter=train_time / it,
               final_val=hist["val"][-1] if hist["val"] and not diverged else float("inf"),
               final_test={k: v[-1] for k, v in hist["test"].items() if v} if not diverged else {},
               history=hist, n_params=sum(p.numel() for p in model.parameters()))
    return res, model


def _rpath(out_dir, pde, opt, lr, seed, tag):
    return os.path.join(out_dir, pde, f"{tag}_{opt}_lr{lr:.2e}_s{seed}.json")


def run_cached(problem, opt, lr, seed, out_dir, tag, **kw):
    path = _rpath(out_dir, problem.name, opt, lr, seed, tag)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = path[:-5] + ".ckpt"
    res, model = train_one(problem, opt, lr, seed, ckpt_path=ckpt, **kw)
    if tag == "main":  # phase-1 weights are the starting point of phase 2
        torch.save(model.state_dict(), path[:-5] + ".pt")
    with open(path, "w") as f:
        json.dump(res, f)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return res


def load_operator(problem, out_dir, opt, lr, seed):
    """Weights of a finished phase-1 main run."""
    path = _rpath(out_dir, problem.name, opt, lr, seed, "main")[:-5] + ".pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run the phase-1 main run for ({opt}, lr={lr:.2e}, seed={seed}) first")
    return torch.load(path, map_location=DEVICE)


# ------------------------------------------------------------- phase 2 -----

def finetune_instance(problem, base_state, opt_name, lr, i, iters=None, anchor=None, n_evals=20,
                      opt_cfg=None, schedule="cosine", ckpt_path=None):
    """Instance-wise fine-tuning (paper Sec. 3.2) of a phase-1 operator on ONE instance i
    (index into the val/test range).  No solution data is used; the true solution only
    serves to measure the error."""
    ft = problem.cfg["finetune"]
    iters = iters or ft["iters"]
    anchor = ft["anchor"] if anchor is None else anchor
    w = {k: v for k, v in problem.cfg["train"]["weights"].items() if k != "data"}
    set_seed(0)
    model = problem.model().to(DEVICE)
    model.load_state_dict(base_state)
    idx = torch.tensor([i])
    pde_res = problem.cfg["train"]["pde_res"]
    fine_res = problem.cfg["test_res"][-1]
    _, truth = problem.get(idx, pde_res)
    with torch.no_grad():
        base_pred, _ = problem.pde_pred_and_losses(model, idx)
    opt = build_optimizer(opt_name, model, lr, opt_cfg)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, iters) / iters)))
             if schedule == "cosine" else None)
    eval_at = set(int(round(iters * (j + 1) / n_evals)) for j in range(n_evals))
    hist = dict(it=[0], time=[0.0], err=[rel_l2(base_pred, truth).item()],
                err_fine=[problem.evaluate(model, idx, fine_res, bs=1)], loss=[float("nan")])
    t = 0.0
    diverged = False
    start = 1

    def save_ckpt(it):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict() if sched else None, hist=hist, t=t, it=it), ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    if ckpt_path and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        if sched:
            sched.load_state_dict(st["sched"])
        hist, t, start = st["hist"], st["t"], st["it"]
        print(f"    resumed fine-tuning {opt_name} instance {i} at step {start}", flush=True)
    last_save = time.time()
    for it in range(start, iters + 1):
        if _past_deadline():
            if ckpt_path:
                save_ckpt(it)
            raise Pause(f"fine-tuning {opt_name} instance {i} interrupted")
        if ckpt_path and time.time() - last_save > CKPT_EVERY_S:
            save_ckpt(it); last_save = time.time()
        model.train()
        sync(); t0 = time.time()
        pred, parts = problem.pde_pred_and_losses(model, idx)
        loss = sum(w[k] * parts[k] for k in parts)
        if anchor > 0:
            loss = loss + anchor * rel_l2(pred, base_pred) ** 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        sync(); t += time.time() - t0
        if not math.isfinite(loss.item()):
            diverged = True
            break
        if it in eval_at:
            model.eval()
            with torch.no_grad():
                p, _ = problem.pde_pred_and_losses(model, idx)
            hist["it"].append(it); hist["time"].append(t); hist["loss"].append(loss.item())
            hist["err"].append(rel_l2(p, truth).item())
            hist["err_fine"].append(problem.evaluate(model, idx, fine_res, bs=1))
    return dict(pde=problem.name, opt=opt_name, lr=lr, instance=int(i), iters=iters, anchor=anchor,
                diverged=diverged, time=t, err0=hist["err"][0],
                final_err=hist["err"][-1] if not diverged else float("inf"),
                final_err_fine=hist["err_fine"][-1] if not diverged else float("inf"),
                pde_res=str(tuple(pde_res) if isinstance(pde_res, (tuple, list)) else pde_res),
                fine_res=str(fine_res), history=hist)


def _ftpath(out_dir, pde, base_tag, opt, lr, i, tag):
    return os.path.join(out_dir, pde, f"{tag}_{base_tag}_{opt}_lr{lr:.2e}_i{i}.json")


def finetune_cached(problem, base_state, base_tag, opt, lr, i, out_dir, tag="ft", **kw):
    path = _ftpath(out_dir, problem.name, base_tag, opt, lr, i, tag)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    ckpt = path[:-5] + ".ckpt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = finetune_instance(problem, base_state, opt, lr, i, ckpt_path=ckpt, **kw)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    r["base"] = base_tag
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(r, f)
    return r


def ft_lr_sweep(problem, base_state, base_tag, opts, out_dir, verbose=True):
    """Pick each optimizer's fine-tuning lr on validation instances (not test)."""
    ft = problem.cfg["finetune"]
    iters = max(5, int(ft["iters"] * ft["sweep_frac"]))
    inst = list(range(problem.a, problem.a + ft["n_sweep_instances"]))
    best = {}
    for opt in opts:
        scores = []
        for m in LR_MULTS:
            lr = BASE_LR[opt] * m
            errs = [finetune_cached(problem, base_state, base_tag, opt, lr, i, out_dir, "ftsweep",
                                    iters=iters, n_evals=1)["final_err"] for i in inst]
            scores.append((float(np.mean(errs)), lr))
            if verbose:
                print(f"  ft-sweep {problem.name:10s} base={base_tag:12s} {opt:14s} lr={lr:.2e}  val err={scores[-1][0]:.4f}")
        best[opt] = min(scores)[1]
    return best


def finetune_runs(problem, base_state, base_tag, best_lr, out_dir, verbose=True):
    ft = problem.cfg["finetune"]
    inst = list(range(problem.b, problem.b + ft["n_instances"]))
    out = []
    for opt, lr in best_lr.items():
        rs = [finetune_cached(problem, base_state, base_tag, opt, lr, i, out_dir) for i in inst]
        out += rs
        if verbose:
            e0 = np.mean([r["err0"] for r in rs]); e = np.mean([r["final_err"] for r in rs])
            print(f"  ft {problem.name:10s} base={base_tag:12s} {opt:14s} lr={lr:.2e}  err {100 * e0:.2f}% -> {100 * e:.2f}%"
                  f"  ({np.mean([r['time'] for r in rs]):.0f}s/instance)")
    return out


def lr_sweep(problem, opts, out_dir, sweep_frac=0.3, seed=0, verbose=True):
    """Short runs over BASE_LR*LR_MULTS; pick each optimizer's lr by validation error."""
    iters = max(10, int(problem.cfg["train"]["iters"] * sweep_frac))
    best = {}
    for opt in opts:
        scores = []
        for m in LR_MULTS:
            lr = BASE_LR[opt] * m
            r = run_cached(problem, opt, lr, seed, out_dir, "sweep", iters=iters, n_evals=1, eval_test=False)
            scores.append((r["final_val"], lr))
            if verbose:
                print(f"  sweep {problem.name:10s} {opt:14s} lr={lr:.2e}  val={r['final_val']:.4f}  ({r['train_time']:.0f}s)")
        best[opt] = min(scores)[1]
    return best


def main_runs(problem, best_lr, seeds, out_dir, verbose=True):
    results = []
    for opt, lr in best_lr.items():
        for s in seeds:
            r = run_cached(problem, opt, lr, s, out_dir, "main")
            results.append(r)
            if verbose:
                ft = "  ".join(f"{k}:{v:.4f}" for k, v in r["final_test"].items())
                print(f"  main  {problem.name:10s} {opt:14s} lr={lr:.2e} seed={s}  test[{ft}]  ({r['train_time']:.0f}s)")
    return results


def load_results(out_dir, tag="main"):
    rows = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if fn.startswith(tag + "_") and fn.endswith(".json"):
                with open(os.path.join(root, fn)) as f:
                    rows.append(json.load(f))
    return rows


# =============================================================================
# 7. REPORTING
# =============================================================================

def summary_table(results):
    import pandas as pd
    recs = []
    for r in results:
        rec = dict(pde=r["pde"], optimizer=r["opt"], lr=r["lr"], seed=r["seed"],
                   time_s=r["train_time"], diverged=r["diverged"])
        for i, (k, v) in enumerate(r["final_test"].items()):
            rec[f"err {['1x', '2x', '4x'][i]} (%)"] = 100 * v
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    cols = [c for c in df.columns if c.startswith("err")] + ["time_s"]
    g = df.groupby(["pde", "optimizer"])[cols]
    if df["seed"].nunique() > 1:
        mean, std = g.mean(), g.std()
        out = mean.copy().astype(object)
        for c in cols:
            fmt = "{:.0f} ± {:.0f}" if c == "time_s" else "{:.2f} ± {:.2f}"
            out[c] = [fmt.format(m, s) for m, s in zip(mean[c], std[c])]
        return out
    return g.mean().round(3)


def ft_summary_table(ft_results):
    """Phase 2: mean ± std over instances of start / final error and time."""
    import pandas as pd
    df = pd.DataFrame([dict(pde=r["pde"], base=r["base"], optimizer=r["opt"], lr=r["lr"],
                            start=100 * r["err0"], final=100 * r["final_err"], final_fine=100 * r["final_err_fine"],
                            time_s=r["time"], diverged=r["diverged"]) for r in ft_results])
    if df.empty:
        return df
    g = df.groupby(["pde", "base", "optimizer"])
    out = g[["start", "final", "final_fine", "time_s"]].agg(["mean", "std"]).round(3)
    out.columns = [f"{a} {b}" for a, b in out.columns]
    out = out.rename(columns=lambda c: c.replace("start", "err before (%)").replace("final_fine", "err @ finest test grid (%)")
                     .replace("final", "err after (%)"))
    out["n_instances"] = g.size()
    out["lr"] = g["lr"].first()
    return out


def plot_ft_curves(ft_results, pde, base=None, x="it", ax=None):
    """Mean (over instances) error vs iteration or time during phase-2 fine-tuning."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    colors = dict(adam="#1f77b4", muon="#ff7f0e", muon_spectral="#9467bd", soap="#2ca02c")
    rs = [r for r in ft_results if r["pde"] == pde and (base is None or r["base"] == base) and not r["diverged"]]
    for opt in sorted({r["opt"] for r in rs}):
        sel = [r for r in rs if r["opt"] == opt]
        L = min(len(r["history"]["err"]) for r in sel)
        errs = np.array([r["history"]["err"][:L] for r in sel])
        xs = np.array([r["history"]["it" if x == "it" else "time"][:L] for r in sel]).mean(0)
        m, lo, hi = np.exp(np.log(errs).mean(0)), np.percentile(errs, 25, 0), np.percentile(errs, 75, 0)
        ax.plot(xs, m, color=colors.get(opt), label=f"{opt} (lr {sel[0]['lr']:.0e})")
        ax.fill_between(xs, lo, hi, color=colors.get(opt), alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("fine-tuning iteration" if x == "it" else "fine-tuning time (s)")
    ax.set_ylabel("rel. L2 error vs truth (geo. mean, IQR)")
    ax.set_title(f"{pde} — phase 2" + (f" (base: {base})" if base else ""))
    ax.legend()
    return ax


def plot_curves(results, pde, x="it", res_key=None, ax=None, logy=True):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    colors = dict(adam="#1f77b4", muon="#ff7f0e", muon_spectral="#9467bd", soap="#2ca02c")
    for r in sorted((r for r in results if r["pde"] == pde), key=lambda r: (r["opt"], r["seed"])):
        h = r["history"]
        key = res_key or list(h["test"].keys())[-1]
        xs = h["it"] if x == "it" else h["time"]
        ax.plot(xs, h["test"][key], color=colors.get(r["opt"]), alpha=0.85,
                label=f"{r['opt']} (lr {r['lr']:.0e})" if r["seed"] == min(q["seed"] for q in results if q["opt"] == r["opt"] and q["pde"] == pde) else None)
    ax.set_xlabel("iteration" if x == "it" else "training time (s)")
    ax.set_ylabel(f"test rel. L2 @ {key}")
    if logy:
        ax.set_yscale("log")
    ax.set_title(pde)
    ax.legend()
    return ax
