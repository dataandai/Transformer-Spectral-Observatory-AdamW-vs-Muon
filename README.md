# Transformer Spectral Observatory: AdamW vs Muon

**A small, reproducible notebook lab for comparing how AdamW and Muon shape Transformer weight-matrix spectra during training.**

This repository trains a tiny Qwen-style decoder-only language model on WikiText-2 and logs both ordinary loss curves and matrix-level diagnostics for attention and MLP projection weights.

The purpose is simple:

> Same tiny Transformer, same data, same training loop, different optimizer.  
> Compare not only validation loss, but also the spectra, ranks, subspaces, gradients, and weight-space trajectories of the learned matrices.

This is an educational and diagnostic experiment. It is not a benchmark, not a proof that Muon is better than AdamW, and not a complete theory of Transformer training dynamics.

---

## What this is

This repository is a **spectral training-dynamics observatory**.

It is useful for asking questions such as:

- Do AdamW and Muon reach similar validation loss through different matrix trajectories?
- Does one optimizer concentrate matrix energy into fewer singular directions?
- Does one produce broader stable-rank behavior?
- Do dominant singular subspaces drift differently across checkpoints?
- Are gradients aligned with already-dominant spectral directions, or do they push into weaker directions?
- Are weight updates smooth, oscillatory, or abrupt under different optimizers?

The project is designed to make those questions easy to inspect in a small controlled setting.

---

## What this is not

This repository does **not** claim that:

- Muon is universally better than AdamW,
- spectral diagnostics predict downstream quality,
- stable rank is a direct measure of intelligence or representation quality,
- top singular vectors are necessarily features,
- the gradient covariance proxy is a full Fisher matrix,
- small WikiText-2 runs generalize to frontier-scale LLM training,
- word-level tokenization is competitive with modern tokenizer pipelines.

The outputs are diagnostic signals.

They should be interpreted as exploratory measurements, not final conclusions.

---

## Core idea

Most training dashboards show:

- training loss,
- validation loss,
- perplexity.

Those metrics tell us whether optimization is working, but they do not tell us how the internal matrices are changing.

This notebook treats each Transformer projection matrix as a linear operator evolving through training:

```math
W_0, W_1, W_2, \ldots, W_t
```

At diagnostic checkpoints, it measures how that operator changes.

The main practical thesis is:

> Similar validation loss does not imply similar internal matrix geometry.

AdamW and Muon may produce similar external curves while inducing different spectra, effective ranks, gradient anisotropy, subspace drift, and weight-space trajectories.

---

## Why AdamW vs Muon?

AdamW is the standard baseline optimizer for Transformer training.

Muon is a matrix-aware optimizer that applies orthogonalized momentum-style updates to matrix parameters. This makes it especially interesting for diagnostics based on singular values, rank, and subspace motion.

This repository does not try to settle optimizer performance.

It asks a narrower question:

> If AdamW and Muon train the same tiny Transformer, do their learned matrices look different along spectral and trajectory diagnostics?

That is a measurement question, not a leaderboard claim.

---

## Repository contents

Current files include:

```text
README.md
loss_curves_adamw_vs_muon.csv
matrix_diagnostics_adamw_vs_muon.csv
tiny_qwen_wikitext2_matrix_diagnostics_adamw_muon_enhanced_dynamics_fixed.ipynb
tiny_qwen_wikitext2_matrix_diagnostics_adamw_muon_print_titles_fixed.ipynb
```

Expected generated outputs:

```text
loss_curves_adamw_vs_muon.csv
matrix_diagnostics_adamw_vs_muon.csv
```

---

## Model and dataset

The notebook uses a compact Qwen-style causal language model.

Typical design choices:

- small decoder-only Transformer,
- WikiText-2 dataset,
- simple word-level tokenizer,
- attention and MLP projection matrices tracked,
- dense diagnostic logging over training.

The setup is intentionally small.

The goal is not to train a strong language model. The goal is to make internal optimizer-induced matrix dynamics visible and cheap to reproduce.

---

## Diagnostics

### 1. Loss curves

The notebook logs:

- training loss,
- validation loss,
- approximate perplexity.

These provide the external optimization baseline.

The spectral diagnostics should always be read alongside loss curves. Geometry without loss is just expensive numerology.

---

### 2. Singular value spectra

For each tracked matrix:

```math
W = U \Sigma V^\top
```

the notebook records singular-value information.

Tracked quantities include:

- Frobenius norm,
- spectral norm,
- top singular values,
- singular-value curves,
- top-k energy ratio,
- row and column norm outlier ratios.

These diagnostics show whether matrix energy is broad or concentrated.

---

### 3. Stable rank

Stable rank is defined as:

```math
\mathrm{srank}(W) =
\frac{\|W\|_F^2}{\|W\|_2^2}
```

Interpretation:

- lower stable rank means energy is concentrated in fewer dominant directions,
- higher stable rank means energy is distributed across more directions.

In this repository, stable rank is used as a soft spectral-spread proxy.

It is not a quality metric.

---

### 4. Top-k spectral energy

Top-k spectral energy measures how much of the matrix energy is captured by the leading singular directions:

```math
E_k =
\frac{\sum_{i=1}^{k} \sigma_i^2}
{\sum_i \sigma_i^2}
```

High top-k energy suggests spectral concentration.

Lower top-k energy suggests broader spectral support.

This is useful for comparing whether AdamW or Muon produces more concentrated projection matrices in the tested setup.

---

### 5. Subspace alignment and drift

The notebook compares dominant singular subspaces across checkpoints.

For example:

```math
V_k^{(t)}
\quad \text{vs.} \quad
V_k^{(t+\Delta)}
```

A high overlap score means the dominant subspace is relatively stable.

A low overlap score means the dominant subspace has rotated or changed.

This is a weight-space diagnostic. It should not be overinterpreted as a direct representation-similarity measure.

---

### 6. Gradient projection ratio

The current gradient matrix is:

```math
G_t = \nabla_W L_t
```

The notebook measures how much of the gradient lies in the current dominant spectral directions of the weight matrix.

Conceptually:

```math
\frac{\|P_k(G_t)\|_F^2}{\|G_t\|_F^2}
```

where \(P_k\) projects the gradient onto the top-k spectral subspace of \(W\).

Interpretation:

- high projection ratio: updates mostly reinforce already-dominant directions,
- low projection ratio: updates are more distributed or push into weaker directions.

This can reveal whether the optimizer is refining the current dominant matrix structure or introducing new directions.

---

### 7. Gradient covariance / Fisher proxy

The true Fisher matrix is:

```math
F =
\mathbb{E}
\left[
\nabla_\theta \log p_\theta(y|x)
\nabla_\theta \log p_\theta(y|x)^\top
\right]
```

The full empirical Fisher is too expensive for this small notebook lab.

Instead, the notebook uses a cheap layer-level proxy based on the current matrix gradient:

```math
GG^\top
\quad \text{or} \quad
G^\top G
```

Tracked quantities include:

- trace proxy,
- largest eigenvalue proxy,
- effective rank proxy,
- top-k gradient-energy ratio.

This is not a full Fisher estimate.

It is only a compact diagnostic for gradient anisotropy and update concentration.

---

### 8. Weight velocity and acceleration

At diagnostic checkpoints, the finite-difference velocity is:

```math
v_t = W_t - W_{t-\Delta}
```

and acceleration is:

```math
a_t = v_t - v_{t-\Delta}
```

Tracked quantities include:

- velocity Frobenius norm,
- velocity spectral norm,
- acceleration Frobenius norm,
- relative acceleration,
- velocity cosine similarity.

These metrics describe whether optimizer trajectories are smooth, abrupt, or oscillatory in weight space.

They are not physical velocities. They are just finite-difference trajectory diagnostics.

---

## Suggested interpretation

Use cautious language.

| Avoid saying | Prefer saying |
|---|---|
| Muon learns better geometry | Muon produced different spectral diagnostics in this run |
| stable rank proves better representations | stable rank indicates broader or narrower spectral energy |
| subspace drift means feature discovery | top-k singular subspace changed between checkpoints |
| Fisher proxy | gradient-covariance proxy |
| optimizer X is better | optimizer X had lower loss or different diagnostics under this setup |
| this explains LLM training | this provides a small-scale diagnostic view |

---

## Minimal run

Install requirements:

```bash
pip install torch datasets transformers matplotlib pandas numpy tqdm
```

Open the notebook:

```bash
jupyter notebook tiny_qwen_wikitext2_matrix_diagnostics_adamw_muon_enhanced_dynamics_fixed.ipynb
```

Run all cells.

You can change:

- optimizer,
- learning rate,
- batch size,
- model size,
- number of training steps,
- diagnostic interval,
- tracked matrix name patterns,
- top-k singular directions.

---

## Recommended experiment protocol

For a cleaner AdamW-vs-Muon comparison:

1. Fix model architecture.
2. Fix dataset and tokenizer.
3. Fix seed.
4. Fix batch size and training schedule.
5. Tune AdamW and Muon learning rates separately enough to avoid unfair defaults.
6. Run both optimizers.
7. Compare loss curves first.
8. Compare spectral diagnostics only after confirming both runs are valid.
9. Repeat over multiple seeds if making claims.

A minimal results table should include:

| Optimizer | Seed | Final train loss | Final val loss | Mean stable rank | Mean top-k energy | Mean subspace drift |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 1 | TBD | TBD | TBD | TBD | TBD |
| Muon | 1 | TBD | TBD | TBD | TBD | TBD |

For stronger claims, report mean and standard deviation across seeds.

---

## How to read the CSV outputs

### `loss_curves_adamw_vs_muon.csv`

This file stores scalar optimization metrics over training.

Typical columns may include:

- optimizer,
- step,
- train loss,
- validation loss,
- perplexity.

Use this file to check whether a run is healthy before interpreting geometry.

### `matrix_diagnostics_adamw_vs_muon.csv`

This file stores matrix-level diagnostics over checkpoints.

Typical columns may include:

- optimizer,
- step,
- module name,
- Frobenius norm,
- spectral norm,
- stable rank,
- top-k energy,
- subspace alignment,
- gradient projection ratio,
- gradient covariance proxy metrics,
- velocity and acceleration metrics.

Use this file to compare optimizer-induced matrix dynamics layer by layer.

---

## What would make the project stronger?

### Multi-seed runs

Single-run differences are interesting but fragile.

The next step is to run the same configuration across multiple seeds and report averages.

### Better optimizer tuning

AdamW and Muon may require different learning rates and weight decay settings.

A fair comparison should not assume that identical hyperparameters are optimal for both.

### End-of-schedule evaluation

Optimizer rankings can change over the full learning-rate schedule.

Do not overinterpret early curves unless the experiment is explicitly about early training.

### Activation-side diagnostics

Weight spectra are useful, but weight geometry and activation geometry are not identical.

Adding activation CKA, SVCCA, or token-level activation spectra would make the comparison more complete.

### Larger models

The tiny model makes diagnostics cheap, but larger models are needed before making broader claims.

### Cleaner command-line runner

The project would become more reproducible if the notebook were backed by a small CLI:

```bash
python train_observatory.py --optimizer adamw --seed 1 --run-name adamw_seed1
python train_observatory.py --optimizer muon --seed 1 --run-name muon_seed1

python compare_runs.py \
  --run-a runs/adamw_seed1 \
  --run-b runs/muon_seed1 \
  --metrics loss stable_rank topk_energy subspace_drift grad_projection
```

The notebook can remain the readable educational entry point.

The CLI would make repeated experiments easier.

---

## Limitations

- The model is intentionally tiny.
- WikiText-2 is small and not representative of modern pretraining.
- The tokenizer is intentionally simple.
- Weight-space metrics are affected by parameterization and normalization.
- SVD diagnostics can become expensive for larger matrices.
- The gradient covariance proxy is not a full Fisher matrix.
- Similar spectral diagnostics do not imply similar model behavior.
- Different spectral diagnostics do not automatically imply better or worse behavior.
- Single-seed results should be treated as examples, not evidence.

---

## References and related ideas

This project is related to:

- AdamW optimization,
- Muon and matrix-aware optimization,
- singular-value diagnostics,
- stable rank and effective rank,
- SVCCA and CKA-style representation comparison,
- natural gradient and Fisher geometry,
- K-FAC and curvature-aware optimization,
- empirical spectral analysis of neural-network weights.

These ideas are used here as practical diagnostics, not as proof of a complete theory.

---

## Scope statement

This repository is best understood as a small educational observatory for optimizer-induced matrix dynamics.

It is useful if you want to inspect what changes inside Transformer weight matrices when training with AdamW versus Muon.

It is not useful as evidence that one optimizer is universally better unless the experiments are extended with proper hyperparameter tuning, multiple seeds, larger models, and task-level evaluation.

The core value is reproducibility and visibility:

> Put more instruments on a tiny Transformer training run, then compare what AdamW and Muon do inside the matrices.

