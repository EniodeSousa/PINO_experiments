"""Kolmogorov fine-tuning with the residual measured in a negative Sobolev norm (notebook 03 follow-up).

SOAP fine-tunes 5 test instances with the Crank-Nicolson residual measured in
    s = 0    relative L2 norm      (the baseline of notebook 03)
    s = 1    relative H^-1 norm    (weight (1 + |k|^2)^-1 per spatial Fourier mode)
    s = 0.5  relative H^-1/2 norm  (weight (1 + |k|^2)^-1/2), closer to the advection scaling |k|
from 3 different starting operators: the Adam-, Muon- and SOAP-trained operators of the optimizer
study (fine-tuning itself is deterministic, so the starting operator plays the role of a seed).
Same steps (400) and learning rate (3e-3) as before. The CN system is square, so every norm has
the same exact minimizer and the same floor (computed per instance, runs/kf_finetune_cn/floors.json).
Results: runs/kf_finetune_hs{0,1,0.5}/. Resumable (checkpoints every 15 min); set BUDGET_S to pause.
"""
import json
import os
import shutil
import time

import torch

import pino_bench as pb
import experiments as ex

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.environ.get("OPT_RUNS", ex.ROOT)          # data + operator weights of the optimizer study
NORMS = [0.0, 1.0, 0.5]
N_INSTANCES = 5
OPT, LR = "soap", 3e-3
STARTS = {"adam_s0": "main_adam_lr1.00e-03_s0.pt", "muon_s0": "main_muon_lr1.00e-03_s0.pt",
          "soap_s0": "main_soap_lr3.00e-03_s0.pt"}
FLOORS = os.path.join(HERE, "runs", "kf_finetune_cn", "floors.json")


def out_dir(s):
    return os.path.join(HERE, "runs", f"kf_finetune_hs{s:g}")


def problem(s, data):
    cfg = ex.cfg_for("kolmogorov")
    cfg["time_scheme"] = "cn"
    cfg["residual_sobolev"] = s
    cfg["finetune"]["n_instances"] = N_INSTANCES
    return pb.make_problem("kolmogorov", cfg, data)


def floors(prob):
    """Error of the unique CN minimizer on the PDE grid, per instance (independent of the norm)."""
    fl = json.load(open(FLOORS)) if os.path.exists(FLOORS) else {}
    res = tuple(prob.cfg["train"]["pde_res"])
    for i in range(prob.b, prob.b + N_INSTANCES):
        if str(i) in fl:
            continue
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
        os.makedirs(os.path.dirname(FLOORS), exist_ok=True)
        json.dump(fl, open(FLOORS, "w"))
    return fl


def run():
    data = torch.load(os.path.join(OLD, "data", "kolmogorov_big_s0.pt"))
    # the L2 / SOAP / Adam-start runs on the first two instances already exist (notebook 03 rerun)
    for f in os.listdir(os.path.join(HERE, "runs", "kf_finetune_cn", "kolmogorov")):
        if f.startswith("ft_adam_s0_soap_"):
            dst = os.path.join(out_dir(0.0), "kolmogorov", f)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(os.path.join(HERE, "runs", "kf_finetune_cn", "kolmogorov", f), dst)
    fl = floors(problem(0.0, data))
    print("CN floor (%):", {k: round(100 * v, 2) for k, v in sorted(fl.items())}, flush=True)
    for tag, fname in STARTS.items():
        base = torch.load(os.path.join(OLD, "results", "kolmogorov", fname), map_location="cpu")
        for s in NORMS:
            print(f"--- start {tag}, H^-{s:g} residual", flush=True)
            pb.finetune_runs(problem(s, data), base, tag, {OPT: LR}, out_dir(s))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        run()
        print("ALL DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
