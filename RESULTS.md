# Results: Tier 1 replica of arXiv:2608.15042 on Spartan

**Run** `20261015-234246-32990966` · trained 15–17 Oct 2026, sampled 18–19 Oct 2026
**Code** commit `61720e5b` · sbi 0.27.0 · torch 2.14.1+cu126 · Python 3.10.4 · one NVIDIA A100 80GB PCIe (Spartan `gpu-a100`)

## Summary

We trained the paper's flow-matching posterior estimator, with the same forward model and network, on 10⁷ simulations. We then drew 3.2×10⁶ posterior samples at the NuFIT 6.0 best fit.

**1,204 samples (0.038%) have χ² < 45.** That is 1.8× the paper's rate (640, 0.021%) and well above the Tier 1 target of 600 or more. However, **25.4% have χ² < 2500**, against the paper's 31.9%. Compared with the paper's network, ours puts more probability mass very close to the best fit but more in the far tail. The ten independent sampling seeds agree to within counting noise. Both results come from a single training run each, so run-to-run variance is not yet known.

## Headline comparison

Fractions are relative to all draws, including the 2.6% that fall outside the prior box, which is the convention of the paper's 31.9% (956,607 / 3×10⁶).

| | This run | This run, scaled to 3×10⁶ | Paper (3×10⁶ draws) | Earlier Mac run (10⁵ draws) |
|---|---|---|---|---|
| draws | 3,200,000 | 3,000,000 | 3,000,000 | 100,000 |
| χ² < 45 | **1,204 (0.0376 ± 0.0011%)** | ~1,130 | 640 (0.021%) | 3 |
| χ² < 100 | 5,965 (0.186%) | ~5,590 | — | — |
| χ² < 2500 | **812,877 (25.40%)** | ~762,000 | 956,607 (31.9%) | 19.8% |
| min χ² | 1.86 | | — | 7.8 |
| inside prior box | 97.4% | | — | — |

The uncertainties are Poisson counting errors only. The ratio of our χ² < 45 rate to the paper's is 1.76 ± 0.09 from counting statistics alone, before any run-to-run variance between trainings.

**Which denominator.** sbi's default `posterior.sample` rejects draws outside the prior, so the paper's 3×10⁶ samples were probably all inside the box. Relative to our 3,117,108 in-box samples, the fractions are **26.1%** below χ² 2500 and **0.039%** below χ² 45. This is the convention used in the title of `figures/chi2.png`. Either way the conclusions are the same: about 1.8× the paper's rate below χ² 45, and 26% against 32% below χ² 2500.

## Setup

| | This run | Paper |
|---|---|---|
| Forward model | Type-I seesaw, G (23 dims) → L (11 dims), `simulate.py` | same |
| Prior | box prior on G, Eq. 3.2 | same |
| Training set | 10⁷ prior simulations, seed 0; 90% train / 10% validation | 10⁷ |
| Network | sbi `posterior_flow_nn("transformer")`: 100-dim tokens, 5 layers, 10 heads, MLP ratio 4; z-score x only | transformer, 100-dim tokens |
| Conditioning | L at the NuFIT 6.0 best fit (11 dims: log₁₀ Δm²₂₁, log₁₀ Δm²₃₁, \|U_PMNS\|) | same |
| Optimiser | Adam, constant lr 1e-3, batch 1024, gradient clip 5, TF32 matmuls | Adam, max lr 5e-4, batch 256 |
| Stopping | sbi early stopping (patience 20) → 146 epochs, **1.28×10⁶ steps** | 5×10⁴ steps |
| Sampling | ODE (zuko adaptive solver, atol 1e-6, rtol 1e-5); out-of-box draws discarded | ODE |
| χ² | Eq. 3.15 on the five observables T, NuFIT 6.0 normal ordering | same |

These differences in the optimisation settings are the main reason not to read the comparison as like-for-like. We trained about 25× more steps, at a larger batch and a higher, constant learning rate. sbi 0.27 has no learning-rate scheduler, so we did not add one.

## Training

- Simulating 10⁷ points took 24 s with 8 processes. The cache (`sims_1e7_seed0.npz`) reloads in 1.2 s.
- Training ran in two Slurm jobs: 7.0 h, then a time-budget checkpoint and automatic resume, then 2.3 h. Early stopping fired after **146 epochs**, with a best validation loss of **4.7601**.
- Speed: **20.7 ms/step** for the training pass (26.1 ms including sbi's validation pass), or 229 s per epoch with 8,789 steps per epoch. Total training time was 9.3 GPU-hours.

The validation loss was essentially flat after about 40 epochs:

| epoch | 10 | 20 | 30 | 40 | 70 | 80 | 130 (best) | 146 (stop) |
|---|---|---|---|---|---|---|---|---|
| best validation loss | 4.7668 | 4.7649 | 4.7630 | 4.7618 | 4.7609 | 4.7605 | 4.7601 | 4.7601 |

About 70% of the training time (epochs 40–146, roughly 6.8 h) improved the validation loss by only 0.04%. We have not tested whether those epochs changed the χ² tails.

## Posterior samples

The 2×10⁵ draws made at the end of training, plus ten independent jobs of 3×10⁵ draws each (seeds 1–10), were merged into 3.2×10⁶ draws.

| set | seed | draws | χ² < 45 | χ² < 100 | χ² < 2500 |
|---|---|---|---|---|---|
| training job | 0 | 200,000 | 92 | 370 | 50,876 (25.44%) |
| task 1 | 1 | 300,000 | 122 | 568 | 76,208 (25.40%) |
| task 2 | 2 | 300,000 | 97 | 573 | 76,323 (25.44%) |
| task 3 | 3 | 300,000 | 103 | 551 | 76,362 (25.45%) |
| task 4 | 4 | 300,000 | 96 | 517 | 75,947 (25.32%) |
| task 5 | 5 | 300,000 | 108 | 573 | 76,749 (25.58%) |
| task 6 | 6 | 300,000 | 110 | 550 | 75,765 (25.26%) |
| task 7 | 7 | 300,000 | 113 | 527 | 76,174 (25.39%) |
| task 8 | 8 | 300,000 | 110 | 603 | 76,258 (25.42%) |
| task 9 | 9 | 300,000 | 124 | 569 | 76,272 (25.42%) |
| task 10 | 10 | 300,000 | 129 | 564 | 75,943 (25.31%) |
| **total** | | **3,200,000** | **1,204** | **5,965** | **812,877 (25.40%)** |

**Consistency between seeds**
- χ² < 45: the per-task counts have mean 111.2 and standard deviation 11.1. The Poisson expectation is √111 ≈ 10.5.
- χ² < 2500: the per-task fractions have standard deviation 0.09 percentage points, against a binomial expectation of 0.08.

The seeds are therefore statistically consistent, and the totals are not driven by any single task. The training job's own count (92, against about 74 expected at the array's rate) is about 2σ high.

**Medians of the posterior samples.** These are the medians of the 2×10⁵ training-job draws inside the box, with no χ² cut, compared with the NuFIT best fit:

| | posterior median | NuFIT 6.0 best fit |
|---|---|---|
| sin²θ₁₂ | 0.371 | 0.308 |
| sin²θ₁₃ | 0.0349 | 0.0222 |
| sin²θ₂₃ | 0.470 | 0.470 |
| Δm²₂₁ [meV²] | 81.8 | 74.9 |
| Δm²₃₁ [meV²] | 2290 | 2513 |

The median χ² is about 5,800, so a typical posterior sample is still far from the data. The accurate samples sit in a narrow core, which is what the χ² cuts select.

## Figures

`figures/` in the run directory shows the posterior with the χ² < 2500 subset highlighted. `runs/20261015-234246-32990966-chi2lt45/figures/` shows only the 1,204 samples with χ² < 45; it was made locally from `G_chi2_lt_45.npy` with `plot_posterior.py --cut 45`.

**χ² distribution** (`chi2.png`). log₁₀ χ² is roughly bell-shaped, peaking near χ² ≈ 5×10³ with a long tail up to about 10⁶. The χ² < 45 samples sit in the extreme low tail, and the minimum is χ² = 1.86.

**Observables** (`observables.png`)
- **All samples:** broad. Large fractions fall outside the plotted windows (the spikes at the window edges are clipped values), most strongly for Δm²₂₁, Δm²₃₁ and sin²θ₁₃.
- **χ² < 2500:** the distributions are centred on the NuFIT best fit for all five observables, but much wider than the 3σ bands, as a cut this loose allows.
- **χ² < 45:** all five observables lie within or at the edge of the NuFIT 3σ bands. The posterior is unbiased, and the accurate samples are accurate in every observable at once, not just in the sum χ².

**Light-neutrino masses** (`physics.png`, χ² < 45; normal ordering by construction)
- m₁ median 0.71 meV (5–95%: 0.07–3.6 meV); m₂ ≈ 8.7 meV; m₃ ≈ 50 meV.
- Σmᵢ median 59.7 meV (5–95%: 58.2–63.4 meV). This is close to the normal-ordering minimum of about 58.8 meV and far below the DESI+CMB line at 113 meV.
- m_ee median 2.75 meV (5–95%: 1.1–4.4 meV). This is an order of magnitude below the KamLAND-Zen bound of 36 meV.

**Phases.** δ_CP, α₂₁ and α₃₁ show no visible correlations in the χ² < 45 set. With 1,204 points this is weak evidence either way. δ_CP does not enter χ² (Eq. 3.15 uses only T), so the χ² cut does not constrain it. The χ² < 2500 set shows a mild concentration of α₂₁ around π.

**Seesaw parameters** (`params.png`, `corner_scale.png`)
- **Heavy-neutrino masses:** one of |M₁|/M₃ or |M₂|/M₃ is about 0.1–0.2 and the other is nearly unconstrained. These are two mirror-image branches from relabelling N₁ ↔ N₂. log₁₀(v²/M₃ / meV) peaks at about 0.5, and it correlates with the smaller mass ratio.
- **Electron-column Yukawas:** Y₁ₑ and Y₂ₑ (real and imaginary parts) are pulled towards 0.
- **Muon and tau Yukawas** of rows 1 and 2 are pushed towards |·| ≈ 1, the edge of the prior box, while row 3 stays close to the prior. The posterior pressing against the box edge means the prior range limits these parameters.
- The Majorana phases arg M₁ and arg M₂ are flat.

These figures still need to be compared side by side with the paper's.

## Compute and throughput

| | value | note |
|---|---|---|
| training step | 20.7 ms at batch 1024 | ≈ 49,500 training samples/s; the paper's laptop managed ≈ 5,300/s (batch 256, RTX 4070 Laptop) |
| epoch, 10⁷ sims | 229 s | expected to scale roughly linearly with the number of simulations |
| posterior sampling, trained network | 462 samples/s per A100 | the paper's 3×10⁶ draws took 11 h (≈ 76/s); ours took 1.8 GPU-hours |
| posterior sampling, barely trained network | ≈ 1,250 samples/s | the trained vector field needs more adaptive ODE steps |
| GPU memory per sampling chunk | 5.5 / 21.9 / 54.6 GiB | for chunks of 5×10⁴ / 2×10⁵ / 5×10⁵; production used 2×10⁵ |
| total GPU time | ≈ 11.2 h | 9.3 h training, 0.1 h in-job draws, 1.8 h bulk draws |

On `gpu-a100-short`, each user is limited to 2 GPUs at once, so the 10 bulk tasks of about 11 min each ran two at a time. With queueing, the bulk sampling took about 4 h of wall time.

## Caveats

1. **One training run.** We don't yet know how much the χ² fractions vary between training seeds. Until a second seed is run, "different from the paper" is supported, but "better than the paper" is not.
2. **Optimisation differs from the paper**, as listed under Setup. The trade-off we see (more samples below χ² 45, fewer below 2500) could come from the much longer training, the batch size, or the learning rate. We haven't separated these.
3. **Prior-box edges.** Several Yukawa marginals pile up at |Y| ≈ 1, so these results are conditional on the paper's box prior.

## Next steps

1. Compare the figures side by side with the paper's, in particular the χ² < 45 mass and phase distributions.
2. **Second training seed** with the same settings (about 9 GPU-hours, or about 3 h with `--max-epochs 40`). This measures the run-to-run variance behind the comparison with the paper.
3. **More draws from this network:** 3×10⁷ draws should give roughly 11,000 samples below χ² 45, for about 18 GPU-hours.
4. **Training closer to the paper's settings** (batch 256, lr 5e-4, about 5×10⁴ steps). This tests whether the χ² < 2500 gap closes.
5. For any longer runs, cap the epochs (`--max-epochs 40`), given the plateau above.

## Reproducing

From the repository root on Spartan (see the README, "Running on Spartan (Tier 1)"):

```bash
sbatch spartan/train_tier1.slurm                                   # trains, resubmits itself if needed
FM_RUN=<run dir> sbatch --time=0:45:00 spartan/sample_tier1.slurm  # 10 × 3e5 draws
FM_RUN=<run dir> sbatch --dependency=afterok:<array job id> spartan/merge_tier1.slurm
```

Outputs are in `/data/gpfs/projects/punim0011/mhhaya/fmpe/runs/20261015-234246-32990966/`:
- `summary.json`: training, timing and the in-job draws
- `samples_summary.json`: totals over all 3.2×10⁶ draws
- `samples/`: per-seed samples, plus `log.jsonl`
- `G_chi2_lt_45.npy`: the 1,204 accurate parameter points
- `figures/`
