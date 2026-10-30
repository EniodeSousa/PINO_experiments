"""Hard solve from scratch (exp5_hard_case.py, notebook 08) with the residual in negative Sobolev norms.

SOAP (lr 3e-3), a random FNO, Kolmogorov Re 100, T = 1, 64^2 x 65, Crank-Nicolson residual,
1500 steps, on 5 instances, with the residual measured in
    L2 (baseline; instances 1-2 already exist from notebook 08), H^-1 and H^-1/2.
All three losses have the same exact minimizer, so the floor per instance is shared.
Results: runs/exp5_hard/soap{,_hs1,_hs0.5}_s{1..5}.json. Resumable; set BUDGET_S to pause.
"""
import os
import time

import torch

import pino_bench as pb
import exp5_hard_case as e5

SEEDS = [1, 2, 3, 4, 5]
NORMS = [0.0, 1.0, 0.5]


def run():
    for seed in SEEDS:
        e5.reference(seed)
        print(f"instance {seed}: CN floor {100 * e5.floor(seed):.2f}%", flush=True)
    for seed in SEEDS:
        for s in NORMS:
            r = e5.solve("soap", seed, sobolev=s)
            print(f"soap  H^-{s:g}  instance={seed}  err={100 * r['final_err']:.2f}%  floor={100 * r['floor']:.2f}%  "
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
