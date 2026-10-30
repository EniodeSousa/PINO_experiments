"""Three more training seeds (2, 3, 4) for the Darcy L^2-vs-H^-1 comparison (notebook 07, slide 16),
for a total of 5 seeds per row: L2 + Adam, L2 + SOAP, H^-1 + Adam, H^-1 + SOAP.

Same data (generated once, data seed 0), model, iterations, batches and the learning rates chosen by
the original sweeps (L2: Adam 3e-3, SOAP 9e-3; H^-1: both 9e-3); only the training seed changes
(initialization and batch order). Resumable; set BUDGET_S to pause cleanly.
"""
import os
import time

import torch

import pino_bench as pb
import experiments as ex
import exp2_hm1_loss as e2

OLD = os.environ.get("OPT_RUNS", ex.ROOT)
SEEDS = [2, 3, 4]
RUNS = [  # (label, cfg, root, learning rates)
    ("L2", ex.cfg_for, OLD, {"adam": 3e-3, "soap": 9e-3}),
    ("H^-1", e2.cfg_hm1, e2.ROOT, {"adam": 9e-3, "soap": 9e-3}),
]


def run():
    probs = [(label, ex.get_problem("darcy", cfg("darcy"), root), root, lrs) for label, cfg, root, lrs in RUNS]
    for s in SEEDS:
        for label, prob, root, lrs in probs:
            print(f"--- Darcy {label}, seed {s}", flush=True)
            pb.main_runs(prob, lrs, [s], os.path.join(root, "results"))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        run()
        print("ALL DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
