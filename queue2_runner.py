"""Second queue: Kolmogorov fine-tuning in negative Sobolev norms, then the hard solve from scratch
in the same norms. Each stage is resumable; prints QUEUE DONE when both are finished."""
import os
import time

import torch

import pino_bench as pb

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        import kf_sobolev_finetune
        kf_sobolev_finetune.run(); print("STAGE kf_sobolev DONE", flush=True)
        import exp5_sobolev
        exp5_sobolev.run(); print("STAGE exp5_sobolev DONE", flush=True)
        import darcy_more_seeds
        darcy_more_seeds.run(); print("STAGE darcy_more_seeds DONE", flush=True)
        print("QUEUE DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
