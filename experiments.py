"""Exact configurations of the CPU-scale experiments and a resumable driver.

Run one problem (resumes from checkpoints if interrupted; finished runs are cached):

    python experiments.py burgers        # ~13 h on 2 CPU cores
    python experiments.py darcy          # ~6 h
    python experiments.py kolmogorov     # ~20 h (data generation ~1.5 h)

Set BUDGET_S (seconds) to pause cleanly before a job time limit; rerun to resume.
"""
import os
import sys
import time
from copy import deepcopy

import torch

import pino_bench as pb

OPTS = ["adam", "muon", "soap"]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PINO_ROOT", os.path.join(HERE, "runs"))     # data + checkpoints + results


def cfg_for(pde):
    """Configuration used for the reported results (CPU scale, FNO width 32)."""
    c = deepcopy(pb.PRESETS["quick"][pde])
    # Padding of the non-periodic axes: fraction of grid points, rounded up (as in the original FNO).
    c["model"].update(pad_frac=0.25, pad_mode="points")
    if pde == "burgers":
        c.update(n_train=800, n_val=100, n_test=200)
        c["model"].update(width=32, modes=(12, 12))
        c["train"].update(iters=6000, bs_data=8, bs_pde=8)
        c["finetune"].update(iters=1000, n_instances=5)
        c["seeds"], c["sweep"] = [0, 1], 0.2
    elif pde == "darcy":
        c.update(n_train=1000, n_val=100, n_test=200)
        c["model"].update(width=32, modes=(12, 12))
        c["train"].update(iters=8000, bs_data=10, bs_pde=10)
        c["finetune"].update(iters=1000, n_instances=5)
        c["seeds"], c["sweep"] = [0, 1], 0.2
    elif pde == "kolmogorov":  # Re 500: reference at 256^2 on the attractor, PDE loss on 128^2 x 33
        c.update(n_data=200, n_pde=400, n_val=10, n_test=40)
        c["gen"].update(n_gen=256, n_store=128, n_traj=16, t_spin=8.0, snap_every=0.4)
        c["model"].update(width=32, modes=(8, 8, 8))
        c["train"].update(iters=3000, bs_data=2, bs_pde=1)
        c["finetune"].update(iters=400, n_instances=2)
        c["seeds"], c["sweep"] = [0], None
    else:
        raise ValueError(pde)
    return c


def get_problem(pde, cfg=None, root=ROOT):
    cfg = cfg or cfg_for(pde)
    data = pb.load_or_make_data(pde, cfg, os.path.join(root, "data"), "big")
    return pb.make_problem(pde, cfg, data)


def run(pde, root=ROOT):
    """LR sweep (with upward extension) -> main runs (seed-major) -> fine-tuning of the Adam operator."""
    cfg = cfg_for(pde)
    prob = get_problem(pde, cfg, root)
    res_dir = os.path.join(root, "results")
    if cfg["sweep"]:
        lrs = pb.lr_sweep(prob, OPTS, res_dir, sweep_frac=cfg["sweep"])
        sweep_iters = max(10, int(cfg["train"]["iters"] * cfg["sweep"]))
        for o in OPTS:  # if the best lr is at the top of the grid, extend upward (x3) until it isn't
            top = pb.BASE_LR[o] * max(pb.LR_MULTS)
            best_lr = lrs[o]
            best_val = pb.run_cached(prob, o, best_lr, 0, res_dir, "sweep", iters=sweep_iters,
                                     n_evals=1, eval_test=False)["final_val"]
            while best_lr >= top * 0.999 and top < pb.BASE_LR[o] * 100:
                top *= 3
                r = pb.run_cached(prob, o, top, 0, res_dir, "sweep", iters=sweep_iters, n_evals=1, eval_test=False)
                if r["final_val"] < best_val:
                    best_lr, best_val = top, r["final_val"]
            lrs[o] = best_lr
    else:
        lrs = {o: pb.BASE_LR[o] for o in OPTS}
    print(f"[{pde}] learning rates: { {k: f'{v:.1e}' for k, v in lrs.items()} }", flush=True)
    for s in cfg["seeds"]:
        pb.main_runs(prob, lrs, [s], res_dir)
    base = pb.load_operator(prob, res_dir, "adam", lrs["adam"], 0)
    pb.finetune_runs(prob, base, "adam_s0", lrs, res_dir)
    print(f"[{pde}] DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    budget = os.environ.get("BUDGET_S")
    if budget:
        pb.DEADLINE = time.time() + float(budget)
    try:
        for p in sys.argv[1:]:
            run(p)
    except pb.Pause as e:
        print(f"PAUSED ({e}); rerun the same command to resume", flush=True)
