"""Run the queued experiments in order, each resumable; prints QUEUE DONE when all finished.
Stages: CN reruns (notebooks 03, 06) -> muon_spectral (01, 02) -> H^-1 loss (07) -> hard solve (08)."""
import os
import time

import torch

import pino_bench as pb

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    if os.environ.get("BUDGET_S"):
        pb.DEADLINE = time.time() + float(os.environ["BUDGET_S"])
    try:
        import cn_reruns
        cn_reruns.ansatz_part(); cn_reruns.finetune_part(); print("STAGE cn_reruns DONE", flush=True)
        import run_muon_spectral, experiments as ex
        ex.OPTS = ["adam", "muon", "soap", "muon_spectral"]
        for pde in ["darcy", "burgers"]:
            ex.run(pde, root=run_muon_spectral.ROOT)
        print("STAGE muon_spectral DONE", flush=True)
        ex.OPTS = ["adam", "muon", "soap"]
        import exp2_hm1_loss
        exp2_hm1_loss.run(); print("STAGE exp2 DONE", flush=True)
        ex.cfg_for = exp2_hm1_loss._cfg_for          # undo the H^-1 patch before anything else
        import exp5_hard_case
        exp5_hard_case.run(); print("STAGE exp5 DONE", flush=True)
        print("QUEUE DONE", flush=True)
    except pb.Pause as e:
        print(f"PAUSED ({e})", flush=True)
