# Transformer Spectral Observatory   AdamW vs Muon 

A small research and educational project for visualizing how Transformer weight matrices evolve during training.

The notebook trains a tiny Qwen-style language model on WikiText-2 using a simple word-level tokenizer and records matrix diagnostics throughout training.

The main goal is to demonstrate that different optimizers can reach similar losses while producing significantly different internal matrix geometries.

---

## Features

- Tiny Qwen-style Transformer
- WikiText-2 dataset
- Simple word-level tokenizer
- AdamW optimizer
- Muon optimizer
- Train/Validation loss tracking
- Matrix diagnostics during training
- Spectral analysis of weight matrices
- Checkpoint-based measurements

---

## Motivation

Most training analyses focus on:

- training loss
- validation loss
- perplexity

However, these metrics do not reveal how the internal representation geometry evolves.

This project tracks the evolution of Transformer weight matrices and shows that two optimizers can produce very different spectral structures even when their losses are similar.

---

## Recorded Metrics

For a selected matrix (for example FFN `W_out`) the notebook records:

### Frobenius Norm

Measures total matrix energy.

\[
||W||_F
\]

---

### Spectral Norm

Largest singular value.

\[
||W||_2 = \sigma_1
\]

Measures the strongest amplification direction.

---

### Stable Rank

A soft rank estimate:

\[
\text{stable-rank}(W)
=
\frac{||W||_F^2}
     {||W||_2^2}
\]

Useful for understanding whether energy concentrates into a few dominant directions.

---

### Top-k Energy Ratio

\[
\frac{
\sum_{i=1}^{k}\sigma_i^2
}{
\sum_i \sigma_i^2
}
\]

Measures how much energy is captured by the largest singular values.

---

### Row and Column Norms

Tracks how individual neurons and features evolve.

---

### Subspace Alignment

Measures how dominant singular subspaces change between checkpoints.

---

### Gradient Projection Ratio

Measures how strongly gradients align with dominant singular directions.

---

## Example Observation

In our experiments:

- AdamW reduced stable rank significantly.
- Muon produced much larger matrix norms.
- Muon preserved higher effective dimensionality.
- Validation losses remained similar.

This suggests that optimizer choice affects not only convergence speed but also the geometry of learned representations.

---

## Outputs

### Loss Curves

- train loss
- validation loss

Stored in:

```
loss_curves_adamw_vs_muon.csv
```

### Matrix Diagnostics

Stored in:

```
matrix_diagnostics_adamw_vs_muon.csv
```

---

## Requirements

```bash
pip install torch torchvision
pip install datasets
pip install transformers
pip install matplotlib
pip install pandas
pip install numpy
```

---

## Running

Open the notebook:

```bash
jupyter notebook
```

Run all cells.

You can modify:

- optimizer
- learning rate
- weight decay
- batch size
- model size
- checkpoint interval
- target matrix

and observe how matrix geometry changes during training.

---

## Educational Purpose

This repository was created as a compact laboratory for studying:

- spectral learning dynamics
- optimizer-induced geometry
- singular value evolution
- effective rank
- Transformer internal representations

rather than maximizing language modeling performance.

---

## Future Work

Potential extensions:

- Hessian spectrum tracking
- Fisher Information analysis
- CKA representation similarity
- LoRA training dynamics
- Quantization effects
- Layer-wise geometry comparison
- Scaling-law experiments

---

## License

MIT
