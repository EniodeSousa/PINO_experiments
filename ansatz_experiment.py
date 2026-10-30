"""Experiment: is PINO's advantage the FNO ansatz or the grid-based residual?

One chaotic Kolmogorov instance (Re 500, T = 0.125, state on the attractor) is solved from the
PDE loss alone, without data, with three methods:

  fno       FNO (input: w0 on the grid) + grid residual: Fourier derivatives in space and a
            time residual set by TIME_SCHEME (default "cn", Crank-Nicolson; "central" is the
            paper's central difference).  This is PINO's single-instance solve.
  mlp_grid  coordinate MLP (x, y, t) -> w, evaluated on the grid, + exactly the same grid residual.
  pinn      the same MLP with two outputs (w, psi) and autograd derivatives at collocation points
            drawn from the same grid (vorticity / stream-function form; standard PINN).

For each grid we also compute the error of the exact minimizer of the discrete grid loss
("floor"): a method using the grid residual that reaches the floor has no optimization deficit.

Why Crank-Nicolson: central differences on the interior frames give N-1 equations for N unknown
frames, so that loss has a whole family of exact minimizers and its "floor" is not defined. The
CN residual is square, with a unique minimizer.

Usage:  python ansatz_experiment.py            (resumable; results in runs/ansatz_cn/*.json)
        TIME_SCHEME=central python ansatz_experiment.py   (the paper's residual; runs/ansatz_central/)
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

import pino_bench as pb

HERE = os.path.dirname(os.path.abspath(__file__))
TIME_SCHEME = os.environ.get("TIME_SCHEME", "cn")   # "cn" (well-posed) or "central" (as in PINO)
OUT = os.environ.get("ANSATZ_OUT", os.path.join(HERE, "runs", f"ansatz_{TIME_SCHEME}"))
RE, T, NT = 500.0, 0.125, 33
W_IC = 5.0                      # weight of the initial-condition term (as in the operator experiments)
GRIDS = [16, 32, 64]            # spatial points per direction; NT time points throughout
SEEDS = [1, 2]                  # reference instances (independent attractor states)
METHODS = ["fno", "mlp_grid", "pinn"]
CFG = dict(steps=3000, lr=1e-3, mlp_width=128, mlp_depth=4, n_freq=16, pinn_batch=4096,
           fno_width=32, fno_modes=(8, 8, 8))


# ---------------------------------------------------------------------------- reference
def reference(seed):
    """Window [0, T] of a 256^2 pseudo-spectral solution started on the attractor (cached)."""
    path = os.path.join(OUT, f"reference_s{seed}.pt")
    if os.path.exists(path):
        return torch.load(path)
    cfg = dict(Re=RE, T=T, n_pde=1, n_data=0, n_val=1, n_test=0, train=dict(data_res=(256, NT)),
               gen=dict(n_gen=256, n_store=256, nt_store=NT, substeps=4, n_traj=2, t_spin=12.0, snap_every=0.2))
    w = pb.make_kf_data(cfg, seed=seed, device="cpu", verbose=False)["w_eval"][0].double()
    os.makedirs(OUT, exist_ok=True)
    torch.save(w, path)
    return w


def sub(w, n):
    s = w.shape[0] // n
    return w[::s, ::s, :]


def residual_problem():
    p = pb.Kolmogorov.__new__(pb.Kolmogorov)
    p.nu, p.T, p.cfg = 1.0 / RE, T, {"time_scheme": TIME_SCHEME}
    return p


def grid_loss(prob, w, w0):
    """Same loss as the PINO experiments: relative residual + W_IC * relative IC mismatch."""
    Du, f = prob.residual(w)
    r = (Du - f).norm() / f.norm()
    ic = (w[..., 0] - w0).norm() / w0.norm()
    return r + W_IC * ic, r.item(), ic.item()


def floor_error(w_ref, n):
    """Exact minimizer of the discrete residual (initial frame fixed to the truth) vs the reference.
    With TIME_SCHEME = "cn" the minimizer is unique; with "central" it is the one nearest the truth."""
    prob = residual_problem()
    wt = sub(w_ref, n)[None]
    w0, v = wt[..., :1], wt[..., 1:].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([v], lr=1.0, max_iter=400, history_size=50, tolerance_grad=1e-12,
                            tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        Du, f = prob.residual(torch.cat([w0, v], -1))
        loss = ((Du - f) ** 2).mean()
        loss.backward()
        return loss
    opt.step(closure)
    w = torch.cat([w0, v.detach()], -1)
    return pb.rel_l2(w, wt).item()


# ---------------------------------------------------------------------------- ansätze
class FourierMLP(nn.Module):
    """(x, y, t) -> outputs; periodic in x, y through Fourier features, tanh MLP."""

    def __init__(self, n_out, width, depth, n_freq):
        super().__init__()
        self.register_buffer("k", torch.arange(1, n_freq + 1, dtype=torch.float64).float())
        layers, d = [], 4 * n_freq + 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):  # x, y in [0, 2pi), t in [0, T]
        kx, ky = x[:, None] * self.k, y[:, None] * self.k
        feats = torch.cat([torch.sin(kx), torch.cos(kx), torch.sin(ky), torch.cos(ky), (t / T)[:, None]], 1)
        return self.net(feats)


def grid_coords(n):
    x = torch.arange(n, dtype=torch.float32) * 2 * math.pi / n
    t = torch.linspace(0, T, NT)
    X, Y, Tt = torch.meshgrid(x, x, t, indexing="ij")
    return X.reshape(-1), Y.reshape(-1), Tt.reshape(-1)


def make_method(method, n, w0):
    """Returns (parameters, field_fn, loss_fn); field_fn() gives w on the n x n x NT grid."""
    torch.manual_seed(0)
    prob = residual_problem()
    if method == "fno":
        cfg = dict(Re=RE, T=T, model=dict(modes=CFG["fno_modes"], width=CFG["fno_width"], layers=4,
                                          pad_frac=0.25, pad_mode="points"))
        kp = pb.Kolmogorov.__new__(pb.Kolmogorov)
        kp.cfg, kp.T, kp.nu = cfg, T, 1.0 / RE
        model = kp.model()
        w0f = w0.float()[None]

        def field():
            return kp.predict(model, w0f, NT)

        def loss_fn():
            w = field()
            return grid_loss(prob, w, w0f)
        return model, field, loss_fn

    width, depth, nf = CFG["mlp_width"], CFG["mlp_depth"], CFG["n_freq"]
    X, Y, Tt = grid_coords(n)
    w0f = w0.float()[None]
    if method == "mlp_grid":
        model = FourierMLP(1, width, depth, nf)

        def field():
            return model(X, Y, Tt)[:, 0].reshape(1, n, n, NT)

        def loss_fn():
            return grid_loss(prob, field(), w0f)
        return model, field, loss_fn

    # pinn: outputs (w, psi); residual by autograd at a random batch of grid points
    model = FourierMLP(2, width, depth, nf)
    y_f = -4 * torch.cos(4 * Y)               # forcing at every grid point
    f_scale = y_f.pow(2).mean().sqrt()
    w0_flat = w0.float().reshape(-1)
    x0 = torch.arange(n, dtype=torch.float32) * 2 * math.pi / n
    X0, Y0 = [a.reshape(-1) for a in torch.meshgrid(x0, x0, indexing="ij")]
    gen = torch.Generator().manual_seed(0)

    def field():
        return model(X, Y, Tt)[:, 0].reshape(1, n, n, NT)

    def loss_fn():
        idx = torch.randint(0, X.numel(), (min(CFG["pinn_batch"], X.numel()),), generator=gen)
        x, y, t = (X[idx].clone().requires_grad_(True), Y[idx].clone().requires_grad_(True),
                   Tt[idx].clone().requires_grad_(True))
        out = model(x, y, t)
        w, psi = out[:, 0], out[:, 1]
        g = lambda f_, v: torch.autograd.grad(f_, v, torch.ones_like(f_), create_graph=True)[0]
        wx, wy, wt = g(w, x), g(w, y), g(w, t)
        wxx, wyy = g(wx, x), g(wy, y)
        px, py = g(psi, x), g(psi, y)
        pxx, pyy = g(px, x), g(py, y)
        u, v = py, -px                                  # velocity from the stream function
        res = wt + u * wx + v * wy - (wxx + wyy) / RE - y_f[idx]
        r = res.pow(2).mean().sqrt() / f_scale
        poisson = (pxx + pyy + w).pow(2).mean().sqrt() / w0_flat.pow(2).mean().sqrt()
        w_init = model(X0, Y0, torch.zeros_like(X0))[:, 0]
        ic = (w_init - w0_flat).norm() / w0_flat.norm()
        return r + poisson + W_IC * ic, r.item(), ic.item()
    return model, field, loss_fn


# ---------------------------------------------------------------------------- one solve
def solve(method, n, seed, steps=None, n_evals=30):
    path = os.path.join(OUT, f"{method}_n{n}_s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    steps = steps or CFG["steps"]
    w_ref = reference(seed)
    wt = sub(w_ref, n).float()[None]
    model, field, loss_fn = make_method(method, n, wt[0, ..., 0])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    eval_at = set(int(round(steps * (j + 1) / n_evals)) for j in range(n_evals))
    hist = dict(step=[], time=[], err=[], residual=[], ic=[])
    t_train = 0.0
    for s in range(1, steps + 1):
        if pb._past_deadline():
            raise pb.Pause(f"{method} n={n} seed={seed} interrupted at step {s}")
        t0 = time.time()
        loss, r, ic = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        t_train += time.time() - t0
        if s in eval_at:
            with torch.no_grad():
                err = pb.rel_l2(field(), wt).item()
            hist["step"].append(s); hist["time"].append(t_train); hist["err"].append(err)
            hist["residual"].append(r); hist["ic"].append(ic)
    res = dict(method=method, n=n, nt=NT, seed=seed, steps=steps, cfg=CFG, final_err=hist["err"][-1],
               best_err=min(hist["err"]), time=t_train, n_params=sum(p.numel() for p in model.parameters()),
               history=hist)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(path, "w"))
    return res


def floors():
    path = os.path.join(OUT, "floors.json")
    fl = json.load(open(path)) if os.path.exists(path) else {}
    for seed in SEEDS:
        for n in GRIDS:
            key = f"n{n}_s{seed}"
            if key not in fl:
                fl[key] = floor_error(reference(seed), n)
                json.dump(fl, open(path, "w"))
    return fl


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        for seed in SEEDS:
            reference(seed)
        print("floors:", {k: round(100 * v, 2) for k, v in floors().items()}, flush=True)
        for seed in SEEDS:                       # instance-major: a full comparison exists after seed 1
            for n in GRIDS:
                for m in METHODS:
                    r = solve(m, n, seed)
                    print(f"{m:9s} n={n:3d} seed={seed}  err={100 * r['final_err']:.2f}%  "
                          f"(best {100 * r['best_err']:.2f}%)  {r['time'] / 60:.1f} min", flush=True)
        print("ALL DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
