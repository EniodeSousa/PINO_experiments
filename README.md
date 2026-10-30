# Optimizers, losses and ansatzes for physics-informed neural operators

This repository studies what makes the **physics-informed neural operator (PINO)** of Li, Zheng et al.,
[*Physics-Informed Neural Operator for Learning Partial Differential Equations*](https://arxiv.org/abs/2111.03794)
(arXiv:2111.03794), train well. It started as a comparison of four optimizers (**Adam**, **Muon**, **Muon-spec** and **SOAP**) on the paper's three benchmarks, and grew into four follow-up experiments:

| Question | Notebooks | Short answer |
|---|---|---|
| Does the optimizer matter? | 01–03 | Somewhat for operator learning (up to 1.5×); a lot on hard single-instance solves, where **SOAP** wins |
| Is the ansatz or the grid residual what makes PINO work? | 06 | The **FNO ansatz**: with the same grid loss, an MLP stalls near 40% while the FNO reaches the discrete solution |
| Does the *norm* of the residual matter? | 07, 09 | Much more than the optimizer: an **H⁻¹ residual** cuts the Darcy error 5–7× (5 seeds); on Kolmogorov an **H^(−1/2) residual** wins in 17 of 20 comparisons |
| On a hard solve from scratch, does the optimizer decide? | 08 | Yes: SOAP ends 2–10× lower than Adam |

Everything is self-contained: data generators, the FNO, the PINO losses, and the Muon, Muon-spec and SOAP optimizers are all in `pino_bench.py`. Nothing is downloaded.

> This is independent code, not the authors' implementation. All results are from **CPU-scale runs** (2 cores, FNO width 32
> against the paper's 64, fewer iterations), so absolute errors are above the paper's on Burgers and Kolmogorov. Comparisons within this repository are like-for-like.

## Notebooks

| Notebook | Contents |
|---|---|
| [`01_burgers.ipynb`](notebooks/01_burgers.ipynb) | Burgers: learning-rate sweep, operator learning, fine-tuning, for each optimizer |
| [`02_darcy.ipynb`](notebooks/02_darcy.ipynb) | Darcy: same structure |
| [`03_kolmogorov.ipynb`](notebooks/03_kolmogorov.ipynb) | Kolmogorov flow, Re = 500: same structure, plus the fine-tuning rerun with a Crank–Nicolson time residual |
| [`04_checks_loss_and_resolution.ipynb`](notebooks/04_checks_loss_and_resolution.ipynb) | Checks behind the setup: (1) the paper's Darcy residual vs a flux-conservative one; (2) on which grid the Kolmogorov PDE loss is faithful; (3) the Darcy fine-tuning floor; (4) discretization dependence along the zero-padded time axis |
| [`05_colab_full_scale.ipynb`](notebooks/05_colab_full_scale.ipynb) | GPU (Colab) version of 01–03 for larger-scale runs; **not run** at full scale |
| [`06_ansatz_fno_vs_mlp_vs_pinn.ipynb`](notebooks/06_ansatz_fno_vs_mlp_vs_pinn.ipynb) | Same PDE loss, three ansatzes: FNO, coordinate MLP on the grid, PINN with autograd |
| [`07_hm1_loss_darcy.ipynb`](notebooks/07_hm1_loss_darcy.ipynb) | Darcy with the residual measured in the discrete H⁻¹ norm instead of L² |
| [`08_hard_solve_from_scratch.ipynb`](notebooks/08_hard_solve_from_scratch.ipynb) | A long-horizon Kolmogorov solve from a random FNO, PDE loss only |
| [`09_sobolev_norms_kolmogorov.ipynb`](notebooks/09_sobolev_norms_kolmogorov.ipynb) | Kolmogorov residual measured in L², H⁻¹ and H^(−1/2): fine-tuning from 3 starting operators and solves from scratch, 5 instances each |

The notebooks read the saved results in `results/`; set `RUN_EXPERIMENT = True` in 01–03 to regenerate a result from scratch, and see each notebook's "Reproducing" cell for the others. Notebook 04 regenerates everything it needs from fixed seeds in about 15 minutes.

## Main results

Test relative L² error (%). Operator learning shows the mean over seeds, on the grid where the PDE loss is imposed; fine-tuning shows the mean over test instances, starting from the same Adam-trained operator.

| | Adam | Muon | SOAP | Muon-spec |
|---|---|---|---|---|
| **Burgers**, operator, 128×101 (2 seeds) | 0.79 | 0.66 | **0.53** | **0.49** |
| **Darcy**, operator, 61×61 (Adam, SOAP: 5 seeds; Muon, Muon-spec: 2) | 8.91 | **5.53** | **5.00** | 14.3 |
| **Kolmogorov**, operator, 128²×33 (1 seed) | 34.7 | **34.3** | 35.2 | – |
| **Burgers**, fine-tuning, 0.85 → (5 instances) | 0.27 | 0.24 | **0.19** | 0.20 |
| **Darcy**, fine-tuning, 2.54 → (5 instances) | 0.12 | 0.12 | 0.12 | 0.12 |
| **Kolmogorov**, fine-tuning, 31.5 → (2 instances, Crank–Nicolson) | 20.2 | 20.1 | **10.1** | – |
| **Hard solve from scratch**, Kolmogorov Re 100, T = 1 (2 instances; floors 1.52, 0.78) | 7.64, 6.50 | 6.12, 2.09 | **3.14, 0.66** | 39.5, 39.2* |

\* Muon-spec's learning rate was not tuned for the hard solve; at 1e-3 it barely moves.

**Darcy with an H⁻¹ residual** (notebook 07), 61×61, mean ± std over 5 seeds: Adam **1.23 ± 0.03**, SOAP **1.01 ± 0.01**, against 8.91 ± 0.90 and 5.00 ± 0.60 with the L² residual (paper's PINO: 1.56).

**Kolmogorov with negative-Sobolev residuals** (notebook 09), SOAP, mean over 5 instances:

| | L² | H⁻¹ | H^(−1/2) |
|---|---|---|---|
| Fine-tuning, Re 500, from the Adam / Muon / SOAP-trained operator (floor 1.2) | 10.8 / 10.3 / 12.6 | 6.5 / 7.0 / 9.6 | **5.8 / 6.1 / 8.6** |
| From scratch, Re 100, T = 1 (floor 0.96) | 1.54 | 0.63 | **0.42** |

**Same loss, different ansatz** (notebook 06), Kolmogorov Re 500, mean of 2 instances:

| PDE grid | Floor | FNO | MLP + grid | PINN |
|---|---|---|---|---|
| 16² × 33 | 40.3 | **40.3** | 41.1 | 44.9 |
| 32² × 33 | 28.7 | **28.7** | 39.4 | 41.8 |
| 64² × 33 | 13.8 | **10.6** | 39.2 | 41.7 |

The **floor** is the error of the exact minimizer of the discrete loss: where perfect optimization of that loss would land.

## Findings

- **Fix the loss before the optimizer.** On Darcy, measuring the residual in H⁻¹ instead of L² gains 5–7× on the fine grids and cuts the seed-to-seed spread 30×; the best optimizer swap gained 1.2–1.8×. The L² norm of a second-order residual weights error frequencies like |k|⁴ and chases high-frequency wiggles; H⁻¹ weights them like |k|², the energy norm in which Darcy is well posed.
- **On Navier–Stokes the milder H^(−1/2) norm is best.** Both negative norms beat L² in all 20 Kolmogorov comparisons, and H^(−1/2) wins 17 of them, as the energy estimate suggests for advection-dominated flow (its constant scales like Re^(1/3) rather than Re for H⁻¹). Fine-tuning gains 1.5–1.8×; from scratch the negative norms end below the discrete floor, an implicit-regularization effect rather than a better discrete solve.
- **SOAP is the consistent optimizer.** Best or tied in every comparison, and decisive on hard solves: half Adam's error on Kolmogorov fine-tuning, 2–10× lower from scratch. Muon is a reliable second.
- **Muon-spec is inconsistent.** Orthogonalizing each Fourier mode's weights ties SOAP on Burgers but is the worst optimizer on Darcy. Its short (20%-length) Darcy sweep looked best of all, then the full run dipped early and recovered only to 14%: short sweeps can mislead.
- **Easy vs hard solves.** On Darcy every optimizer fine-tunes to the exact discrete minimizer, so accuracy is set by the grid. On Kolmogorov the loss allows 0.7% but every optimizer stops at 10–20%: the optimizer is the bottleneck.
- **The FNO ansatz is what makes the PDE loss work.** With the same grid residual, a coordinate MLP stalls near 40% on every grid, as does a PINN. Caveat: only the FNO receives w₀ as an input.
- **Check that the discrete loss is well posed.** The paper's nested central-difference Darcy residual has its exact minimizer ~70% from the truth, and central differences in time give N−1 equations for N frames (a family of exact minimizers). Flux-conservative and Crank–Nicolson residuals fix both.
- **Kolmogorov operator learning never left the early regime at this scale** (a "no change" prediction scores 37%).

## Setup

| | Burgers | Darcy | Kolmogorov (Re 500, T = 0.125) |
|---|---|---|---|
| Data-loss grid | 32 × 26 | 11 × 11 | 32² × 17 |
| PDE-loss and fine-tuning grid | 128 × 101 | 61 × 61 | 128² × 33 |
| Test grids | 32×26, 64×51, 128×101 | 11, 61, 121 | 32²×17, 64²×33, 128²×33 |
| Reference solver | pseudo-spectral, 1024 modes | finite volume, 241² | pseudo-spectral, 256², attractor states |
| Train / val / test | 800 / 100 / 200 | 1000 / 100 / 200 | 200 data + 400 PDE / 10 / 40 |
| Iterations, seeds | 6000, 2 | 8000, 2 | 3000, 1 |
| Learning rates | swept: all 9e-3 | swept: Adam 3e-3, Muon and SOAP 9e-3, Muon-spec 1e-3 | defaults: Adam and Muon 1e-3, SOAP 3e-3 |

**Common to all three problems:**
- **Model:** FNO with 4 layers and width 32.
- **Schedule and training data:** cosine learning-rate schedule, with identical initialization, data and batches for every optimizer.
- **Learning-rate sweep:** base rate × {1/3, 1, 3} at 20% of the run length, scored on validation and extended upward whenever the best rate fell on the edge of the grid.
- **Optimizers:**
  - **Muon** orthogonalizes the momentum of the hidden weight matrices (Newton–Schulz, update RMS matched to AdamW), 0.35% of the weights, and uses AdamW for the rest, including the spectral weights.
  - **Muon-spec** (`muon_spectral`) also orthogonalizes each Fourier mode's complex c_in × c_out weight matrix.
  - **SOAP** runs Adam in the eigenbasis of Shampoo's per-axis preconditioners, refreshed every 10 steps, on every weight tensor (all but the biases).
- **Fine-tuning:** starts from the Adam-trained operator (seed 0) and minimizes the PDE loss only, without the anchor term.

**Deliberate deviations from the paper:**
- **Darcy residual:** flux-conservative instead of nested central differences (notebook 04).
- **Kolmogorov PDE grid:** 128² instead of 256². On attractor states, a 64² loss's minimizer lies ~8–9% from the reference against ~1–2% for 128² (notebook 04).
- **Kolmogorov time residual:** the operator runs use the paper's central differences; the fine-tuning rerun, notebook 06 and notebook 08 use Crank–Nicolson (`cfg["time_scheme"] = "cn"`). The fine-tuning numbers were unchanged by the switch.

## Reproducing

```bash
pip install -r requirements.txt
python experiments.py burgers        # Adam, Muon, SOAP: ~13 h on 2 CPU cores
python experiments.py darcy          # ~6 h
python experiments.py kolmogorov     # ~20 h, including ~1.5 h of data generation
python run_muon_spectral.py          # adds Muon-spec on Darcy and Burgers (~8 h; needs the two runs above)
python cn_reruns.py                  # notebook 06 (~5 h) + Kolmogorov fine-tuning with Crank–Nicolson (~3.5 h)
python exp2_hm1_loss.py              # notebook 07, H^-1 loss on Darcy (~5 h; needs the Darcy run)
python exp5_hard_case.py             # notebook 08, hard solve from scratch (~13 h)
python kf_sobolev_finetune.py        # notebook 09, Kolmogorov fine-tuning in L2 / H^-1 / H^-1/2 (~30 h; needs the Kolmogorov run)
python exp5_sobolev.py               # notebook 09, from-scratch solves in the three norms (~20 h)
python darcy_more_seeds.py           # notebook 07, seeds 2-4 for the L2 and H^-1 Darcy runs (~8 h)
```
- **Resumable runs:** every run checkpoints and resumes if interrupted. Set `BUDGET_S=<seconds>` to pause cleanly before a job time limit, then rerun the same command. `queue_runner.py` chains `cn_reruns.py`, `run_muon_spectral.py`, `exp2_hm1_loss.py` and `exp5_hard_case.py`; `queue2_runner.py` chains the last three.
- **Output location:** new results go to `runs/`.
- **Analysis:** `analysis.py` builds the tables and figures in the notebooks.

## Repository layout

```
pino_bench.py          data generators, FNO, PINO losses (incl. Crank-Nicolson and negative-Sobolev residual options), Muon, SOAP,
                       training / fine-tuning drivers
experiments.py         exact configurations of the optimizer runs + resumable driver
run_muon_spectral.py   Muon-spec on Darcy and Burgers
ansatz_experiment.py   FNO vs MLP + grid vs PINN on one Kolmogorov instance (notebook 06)
cn_reruns.py           notebook 06 with Crank-Nicolson + the Kolmogorov fine-tuning rerun
exp2_hm1_loss.py       H^-1 PDE loss on Darcy (notebook 07)
exp5_hard_case.py      hard solve from scratch (notebook 08)
kf_sobolev_finetune.py Kolmogorov fine-tuning with negative-Sobolev residual norms (notebook 09)
exp5_sobolev.py        hard solve from scratch with negative-Sobolev residual norms (notebook 09)
darcy_more_seeds.py    extra training seeds for notebook 07
queue_runner.py, queue2_runner.py   run the scripts above in order, resumably
analysis.py            tables and figures from the saved results
results/               saved run summaries and learning curves (JSON) for every run
  burgers/ darcy/ kolmogorov/   optimizer comparison (notebooks 01-03)
  cn_finetune/                  Kolmogorov fine-tuning with Crank-Nicolson (notebook 03)
  ansatz/ ansatz_central/       notebook 06 (Crank-Nicolson / the paper's central differences)
  hm1/                          notebook 07
  hard_solve/                   notebooks 08 and 09 (from-scratch solves)
  kf_norms/                     notebook 09 (fine-tuning in L2 / H^-1 / H^-1/2)
weights/               the Burgers Adam operator (seed 0), used by notebook 04
notebooks/             01-09 as listed above
```

Model weights for the other runs and the generated datasets (about 1 GB) are not included; the scripts regenerate them.

## Caveats

- **Smaller scale than the paper.** The models are half the paper's width with less training, so absolute errors are 1.5–3.5× the paper's on Burgers and far higher on Kolmogorov operator learning.
- **Thin Kolmogorov evidence.** One seed, untuned learning rates and two fine-tuning instances; the hard solve uses two instances and one initialization. Muon and SOAP also ran during a host slowdown in the Kolmogorov operator runs, so compare those by iterations, not wall time.
- **The Burgers 64×51 test grid is worse for every optimizer.** It is the only test grid not used in training, and the time axis is zero-padded. Notebook 04 discusses this discretization dependence.
- **Ansatz comparison confounder.** The FNO receives w₀ as an input field; the MLPs only see coordinates.

## References

- Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, A. Anandkumar. *Physics-Informed Neural Operator for Learning Partial Differential Equations.* arXiv:2111.03794.
- Z. Li et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021.
- K. Jordan et al. *Muon: an optimizer for hidden layers in neural networks* (2024).
- N. Vyas et al. *SOAP: Improving and Stabilizing Shampoo using Adam.* arXiv:2409.11321.
