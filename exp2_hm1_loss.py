"""Precondition the loss instead of the optimizer: H^-1 PDE loss on Darcy (notebook 07).

Same protocol as the Darcy optimizer study, but the PDE loss uses the relative discrete H^{-1}
norm of the residual (cfg["residual_norm"] = "hm1") instead of its L2 norm. Runs Adam and SOAP
(LR sweep with upward extension, 2 seeds), then phase-2 fine-tuning with the H^{-1} loss:
  (a) from the H^{-1}-trained Adam operator, (b) from the L2-trained Adam operator of the original
      study (directly comparable with the original L2 fine-tuning: same start, different loss).
Resumable; set BUDGET_S to pause cleanly.
"""
import os
import time

import torch

import pino_bench as pb
import experiments as ex

OLD = os.environ.get("OPT_RUNS", ex.ROOT)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "exp2_hm1")

_cfg_for = ex.cfg_for


def cfg_hm1(pde):
    c = _cfg_for(pde)
    c["residual_norm"] = "hm1"
    return c


def run():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    link = os.path.join(ROOT, "data", "darcy_big_s0.pt")
    if not os.path.exists(link):
        os.symlink(os.path.join(OLD, "data", "darcy_big_s0.pt"), link)
    ex.cfg_for = cfg_hm1
    ex.OPTS = ["adam", "soap"]
    ex.run("darcy", root=ROOT)                               # (a): sweep, 2 seeds, FT from H^-1 Adam operator
    prob = ex.get_problem("darcy", cfg_hm1("darcy"), ROOT)
    base = torch.load(os.path.join(OLD, "results", "darcy", "main_adam_lr3.00e-03_s0.pt"), map_location="cpu")
    lrs = {"adam": 3e-3, "muon": 9e-3, "soap": 9e-3}        # the lrs of the original L2 fine-tuning
    pb.finetune_runs(prob, base, "l2adam_s0", lrs, os.path.join(ROOT, "results"))   # (b)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        run()
        print("ALL DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
