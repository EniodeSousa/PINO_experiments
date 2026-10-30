"""Reruns with the well-posed Crank-Nicolson time residual (resumable; set BUDGET_S to pause cleanly).

1. ansatz_experiment (FNO vs MLP+grid vs PINN) with TIME_SCHEME=cn  -> runs/ansatz_cn/
2. Kolmogorov phase-2 fine-tuning of the optimizer comparison (Adam / Muon / SOAP), starting from
   the same Adam-trained operator, with the CN residual          -> runs/kf_finetune_cn/
   plus the unique CN-minimizer error ("floor") of each fine-tuning instance.
"""
import json
import os
import sys
import time

import torch

os.environ.setdefault("TIME_SCHEME", "cn")
import pino_bench as pb
import ansatz_experiment as ae
import experiments as ex

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.environ.get("OPT_RUNS", ex.ROOT)   # data + operator weights of the optimizer study
FT_OUT = os.path.join(HERE, "runs", "kf_finetune_cn")


def ansatz_part():
    for seed in ae.SEEDS:
        ae.reference(seed)
    print("CN floors:", {k: round(100 * v, 2) for k, v in ae.floors().items()}, flush=True)
    for seed in ae.SEEDS:
        for n in ae.GRIDS:
            for m in ae.METHODS:
                r = ae.solve(m, n, seed)
                print(f"{m:9s} n={n:3d} seed={seed}  err={100 * r['final_err']:.2f}%  {r['time'] / 60:.1f} min", flush=True)


def finetune_part():
    cfg = ex.cfg_for("kolmogorov")
    cfg["time_scheme"] = "cn"
    data = torch.load(os.path.join(OLD, "data", "kolmogorov_big_s0.pt"))
    prob = pb.make_problem("kolmogorov", cfg, data)
    base = torch.load(os.path.join(OLD, "results", "kolmogorov", "main_adam_lr1.00e-03_s0.pt"), map_location="cpu")
    lrs = {o: pb.BASE_LR[o] for o in ex.OPTS}            # same learning rates as the original fine-tuning
    os.makedirs(FT_OUT, exist_ok=True)
    fpath = os.path.join(FT_OUT, "floors.json")
    fl = json.load(open(fpath)) if os.path.exists(fpath) else {}
    res = tuple(cfg["train"]["pde_res"])
    for i in range(prob.b, prob.b + cfg["finetune"]["n_instances"]):
        if str(i) not in fl:                             # unique CN minimizer on the PDE grid vs reference
            _, w = prob.get(torch.tensor([i]), res)
            w = w.double()
            w0, v = w[..., :1], w[..., 1:].clone().requires_grad_(True)
            opt = torch.optim.LBFGS([v], lr=1.0, max_iter=400, history_size=50, tolerance_grad=1e-12,
                                    tolerance_change=1e-15, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                Du, f = prob.residual(torch.cat([w0, v], -1))
                loss = ((Du - f) ** 2).mean()
                loss.backward()
                return loss
            opt.step(closure)
            fl[str(i)] = pb.rel_l2(torch.cat([w0, v.detach()], -1), w).item()
            json.dump(fl, open(fpath, "w"))
    print("fine-tuning instances, CN floor:", {k: round(100 * v, 2) for k, v in fl.items()}, flush=True)
    pb.finetune_runs(prob, base, "adam_s0", lrs, FT_OUT)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        ansatz_part()
        print("ANSATZ DONE", flush=True)
        finetune_part()
        print("ALL DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
