"""A hard single-instance solve from scratch: does the optimizer decide success? (notebook 08)

Kolmogorov flow at Re = 100 over a long horizon T = 1 (8x the operator experiments), on a
64^2 x 65 grid with the well-posed Crank-Nicolson residual. A randomly initialized FNO
(input: w0) is trained on the PDE loss alone (residual + initial condition, no data, no
pretrained operator) with Adam, Muon, muon_spectral and SOAP. The unique CN minimizer of each
instance ("floor") is computed exactly, so every curve can be read as distance to the floor.
Main metric: time (and steps) to reach a given multiple of the floor.

Resumable; set BUDGET_S to pause cleanly. Results: runs/exp5_hard/*.json
"""
import json
import math
import os
import time

import torch

import pino_bench as pb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "exp5_hard")
RE, T, NT, N = 100.0, 1.0, 65, 64
W_IC = 5.0
SEEDS = [1, 2]
OPTS = {"adam": 1e-3, "muon": 1e-3, "muon_spectral": 1e-3, "soap": 3e-3}
STEPS = 1500
MODEL = dict(modes=(8, 8, 8), width=32, layers=4, pad_frac=0.25, pad_mode="points")


def problem(sobolev=0.0):
    """sobolev = s > 0 measures the residual in the relative H^-s norm in space (same minimizer)."""
    p = pb.Kolmogorov.__new__(pb.Kolmogorov)
    p.cfg = dict(Re=RE, T=T, time_scheme="cn", model=MODEL, residual_sobolev=sobolev)
    p.nu, p.T = 1.0 / RE, T
    return p


def reference(seed):
    path = os.path.join(OUT, f"reference_s{seed}.pt")
    if os.path.exists(path):
        return torch.load(path)
    cfg = dict(Re=RE, T=T, n_pde=1, n_data=0, n_val=1, n_test=0, train=dict(data_res=(N, NT)),
               gen=dict(n_gen=256, n_store=N, nt_store=NT, substeps=4, n_traj=2, t_spin=10.0, snap_every=1.5))
    w = pb.make_kf_data(cfg, seed=seed, device="cpu", verbose=False)["w_eval"][0].double()
    os.makedirs(OUT, exist_ok=True)
    torch.save(w, path)
    return w


def floor(seed):
    path = os.path.join(OUT, "floors.json")
    fl = json.load(open(path)) if os.path.exists(path) else {}
    if str(seed) in fl:
        return fl[str(seed)]
    prob, w = problem(), reference(seed)[None]
    w0, v = w[..., :1], w[..., 1:].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([v], lr=1.0, max_iter=600, history_size=50, tolerance_grad=1e-12,
                            tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        Du, f = prob.residual(torch.cat([w0, v], -1))
        loss = ((Du - f) ** 2).mean()
        loss.backward()
        return loss
    opt.step(closure)
    fl[str(seed)] = pb.rel_l2(torch.cat([w0, v.detach()], -1), w).item()
    json.dump(fl, open(path, "w"))
    return fl[str(seed)]


def solve(opt_name, seed, n_evals=50, sobolev=0.0, lr=None):
    norm = f"_hs{sobolev:g}" if sobolev else ""
    path = os.path.join(OUT, f"{opt_name}{norm}_s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    prob = problem(sobolev)
    w = reference(seed).float()[None]
    w0 = w[..., 0]
    pb.set_seed(0)                                   # same initial FNO for every optimizer
    model = prob.model()
    lr = lr or OPTS[opt_name]
    opt = pb.build_optimizer(opt_name, model, lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    eval_at = set(int(round(STEPS * (j + 1) / n_evals)) for j in range(n_evals))
    hist = dict(step=[0], time=[0.0], err=[1.0], residual=[float("nan")], ic=[float("nan")])
    t_train, start = 0.0, 1
    ckpt = path.replace(".json", ".ckpt")

    def save_ckpt(s):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        hist=hist, t_train=t_train, step=s), ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    if os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        hist, t_train, start = st["hist"], st["t_train"], st["step"]
        print(f"    resumed {opt_name} seed={seed} at step {start}", flush=True)
    last_save = time.time()
    for s in range(start, STEPS + 1):
        if pb._past_deadline():
            save_ckpt(s)
            raise pb.Pause(f"{opt_name} seed={seed} paused at step {s}")
        if time.time() - last_save > 600:
            save_ckpt(s); last_save = time.time()
        t0 = time.time()
        pred = prob.predict(model, w0, NT)
        l = prob.pde_losses(pred, w0)
        loss = l["f"] + W_IC * l["ic"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        t_train += time.time() - t0
        if not math.isfinite(loss.item()):
            break
        if s in eval_at:
            with torch.no_grad():
                err = pb.rel_l2(prob.predict(model, w0, NT), w).item()
            hist["step"].append(s); hist["time"].append(t_train); hist["err"].append(err)
            hist["residual"].append(l["f"].item()); hist["ic"].append(l["ic"].item())
    res = dict(opt=opt_name, lr=lr, seed=seed, steps=STEPS, Re=RE, T=T, grid=[N, N, NT], floor=floor(seed),
               residual_sobolev=sobolev,
               final_err=hist["err"][-1], time=t_train, history=hist)
    json.dump(res, open(path, "w"))
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return res


def run():
    for seed in SEEDS:
        reference(seed)
        print(f"seed {seed}: CN floor {100 * floor(seed):.2f}%", flush=True)
    for seed in SEEDS:
        for o in OPTS:
            r = solve(o, seed)
            print(f"{o:14s} seed={seed}  err={100 * r['final_err']:.2f}%  floor={100 * r['floor']:.2f}%  "
                  f"{r['time'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        run()
        print("ALL DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
