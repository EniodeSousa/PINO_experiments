"""Add the `muon_spectral` optimizer to the Burgers and Darcy comparison.

muon_spectral = Muon that also orthogonalizes each Fourier mode's complex (cin x cout) weight
matrix, so the orthogonalized update covers ~99.6% of the FNO's parameters instead of 0.35%.
Same protocol as the other optimizers (LR sweep at 20% length with upward extension, 2 seeds,
phase-2 fine-tuning of the same Adam operator). Runs of the other optimizers are cached, so
only muon_spectral is computed. Resumable; set BUDGET_S to pause cleanly.
"""
import os
import time

import torch

import pino_bench as pb
import experiments as ex

ROOT = ex.ROOT   # data + results of the study
ex.OPTS = ["adam", "muon", "soap", "muon_spectral"]

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        for pde in ["darcy", "burgers"]:          # Darcy first: cheaper, and where Muon's gain was largest
            ex.run(pde, root=ROOT)
        print("ALL DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
