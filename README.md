# Optimizers for physics-informed neural operators: Adam vs Muon vs SOAP

This repository compares three optimizers, **Adam**, **Muon** and **SOAP**, for training the
**physics-informed neural operator (PINO)** of Li, Zheng et al.,
[*Physics-Informed Neural Operator for Learning Partial Differential Equations*](https://arxiv.org/abs/2111.03794)
(arXiv:2111.03794), on the paper's three benchmark problems:

- **Burgers:** viscous Burgers, ν = 0.01.
- **Darcy:** steady Darcy flow with a two-phase coefficient.
- **Kolmogorov:** chaotic Kolmogorov flow at Re = 500.

Both phases of PINO are compared:
- **Phase 1, operator learning.** A Fourier neural operator trained with a data loss on a coarse grid plus a PDE-residual loss on a finer grid, as in the paper's Table 1.
- **Phase 2, instance-wise fine-tuning.** The trained operator is refined on single test instances with the PDE loss alone.

Everything is self-contained: data generators, the FNO, the PINO losses, and the Muon and SOAP optimizers are all in `pino_bench.py`. Nothing is downloaded.

> This is independent code, not the authors' implementation. Results are from **CPU-scale runs** (2 cores, FNO width 32
> against the paper's 64, fewer iterations), so absolute errors are above the paper's. The comparison between optimizers is like-for-like.

## Notebooks

| Notebook | Contents |
|---|---|
| [`01_burgers.ipynb`](notebooks/01_burgers.ipynb) | Burgers: learning-rate sweep, operator learning, fine-tuning |
| [`02_darcy.ipynb`](notebooks/02_darcy.ipynb) | Darcy: same structure |
| [`03_kolmogorov.ipynb`](notebooks/03_kolmogorov.ipynb) | Kolmogorov flow, Re = 500: same structure |
| [`04_checks_loss_and_resolution.ipynb`](notebooks/04_checks_loss_and_resolution.ipynb) | Checks behind the setup: (1) the paper's Darcy residual vs a flux-conservative one; (2) on which grid the Kolmogorov PDE loss is faithful; (3) the Darcy fine-tuning floor; (4) discretization dependence along the zero-padded time axis |
| [`05_colab_full_scale.ipynb`](notebooks/05_colab_full_scale.ipynb) | GPU (Colab) version for larger-scale runs; **not run** at full scale |

Notebooks 01–03 read the saved results in `results/`. Set `RUN_EXPERIMENT = True` in any of them to regenerate a result from scratch.
Notebook 04 regenerates everything it needs from fixed seeds, in about 15 minutes.

## Main results

Test relative L² error (%). Operator learning shows the mean over seeds, on the grid where the PDE loss is imposed.

| | Adam | Muon | SOAP |
|---|---|---|---|
| **Burgers**, operator, 128×101 (2 seeds) | 0.79 | 0.66 | **0.53** |
| **Darcy**, operator, 61×61 (2 seeds) | 8.44 | **5.53** | **5.47** |
| **Kolmogorov**, operator, 128²×33 (1 seed) | 34.7 | **34.3** | 35.2 |
| **Burgers**, fine-tuning, 0.85 → (5 instances) | 0.27 | 0.24 | **0.19** |
| **Darcy**, fine-tuning, 2.54 → (5 instances) | 0.12 | 0.12 | 0.12 |
| **Kolmogorov**, fine-tuning, 31.5 → (2 instances) | 20.2 | 20.1 | **10.2** |

**Findings:**
- **Operator learning.** On Burgers and Darcy, Muon and SOAP beat Adam by 15–35% at about the same wall time, with gaps larger than the seed spread. SOAP is best on Burgers; Muon and SOAP tie on Darcy. At this scale, Kolmogorov operator learning stays in an early regime (a "no change" prediction scores 37%) and the optimizers barely differ. SOAP fits the coarse data grid far better there, 24.8%, but not the fine grids.
- **Fine-tuning.** On Darcy, all three reach exactly the error of the discrete PDE loss's own minimizer (notebook 04), so accuracy is set by the grid, not the optimizer. On the hard chaotic Kolmogorov solve, **SOAP halves the error** in about the same time.

## Setup

| | Burgers | Darcy | Kolmogorov (Re 500, T = 0.125) |
|---|---|---|---|
| Data-loss grid | 32 × 26 | 11 × 11 | 32² × 17 |
| PDE-loss grid | 128 × 101 | 61 × 61 | 128² × 33 |
| Test grids | 32×26, 64×51, 128×101 | 11, 61, 121 | 32²×17, 64²×33, 128²×33 |
| Reference solver | pseudo-spectral, 1024 modes | finite volume, 241² | pseudo-spectral, 256², attractor states |
| Train / val / test | 800 / 100 / 200 | 1000 / 100 / 200 | 200 data + 400 PDE / 10 / 40 |
| Iterations, seeds | 6000, 2 | 8000, 2 | 3000, 1 |
| Learning rates | swept: all 9e-3 | swept: Adam 3e-3, others 9e-3 | defaults: 1e-3, 1e-3, 3e-3 |

**Common to all three problems:**
- **Model:** FNO with 4 layers and width 32.
- **Schedule and training data:** cosine learning-rate schedule, with identical initialization, data and batches for every optimizer.
- **Learning-rate sweep:** base rate × {1/3, 1, 3} at 20% of the run length, scored on validation and extended upward whenever the best rate fell on the edge of the grid.
- **Optimizer details:**
  - Muon orthogonalizes the momentum of the hidden weight matrices (Newton–Schulz, update RMS matched to AdamW) and uses AdamW for the rest.
  - SOAP runs Adam in the eigenbasis of Shampoo's preconditioners, refreshed every 10 steps, on all tensors.
- **Fine-tuning:** starts from the Adam-trained operator (seed 0) and minimizes the PDE loss only, without the anchor term.

**Deliberate deviations from the paper:**
- **Darcy residual:** flux-conservative instead of nested central differences. The paper's version has a least-squares minimizer ~70% from the true solution (notebook 04).
- **Kolmogorov PDE grid:** 128² instead of 256². On attractor states, a 64² loss's minimizer lies ~8–9% from the reference against ~1–2% for 128² (notebook 04).

## Reproducing

```bash
pip install -r requirements.txt
python experiments.py burgers        # ~13 h on 2 CPU cores
python experiments.py darcy          # ~6 h
python experiments.py kolmogorov     # ~20 h, including ~1.5 h of data generation
```
- **Resumable runs:** every run checkpoints and resumes if interrupted. Set `BUDGET_S=<seconds>` to pause cleanly before a job time limit, then rerun the same command.
- **Output location:** new results go to `runs/results/`.
- **Analysis:** `analysis.py` builds the tables and figures in the notebooks.

## Repository layout

```
pino_bench.py     data generators, FNO, PINO losses, Muon, SOAP, training / fine-tuning drivers
experiments.py    exact configurations of the reported runs + resumable driver
analysis.py       tables and figures from the saved results
results/          saved run summaries and learning curves (JSON) for every run
weights/          the Burgers Adam operator (seed 0), used by notebook 04
notebooks/        01–05 as listed above
```

Model weights for the other runs (~300 MB) and the generated datasets (~800 MB) are not included; `experiments.py` regenerates them.

## Caveats

- **Smaller scale than the paper.** The models are half the paper's width with less training, so absolute errors are 1.5–3.5× the paper's on Burgers and Darcy and far higher on Kolmogorov.
- **Thin Kolmogorov evidence.** One seed, untuned learning rates and two fine-tuning instances. Muon and SOAP also ran during a host slowdown, so compare Kolmogorov by iterations, not wall time.
- **The Burgers 64×51 test grid is worse for every optimizer.** It is the only test grid not used in training, and the time axis is zero-padded. Notebook 04 discusses this discretization dependence.

## References

- Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, A. Anandkumar. *Physics-Informed Neural Operator for Learning Partial Differential Equations.* arXiv:2111.03794.
- Z. Li et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021.
- K. Jordan et al. *Muon: an optimizer for hidden layers in neural networks* (2024).
- N. Vyas et al. *SOAP: Improving and Stabilizing Shampoo using Adam.* arXiv:2409.11321.
