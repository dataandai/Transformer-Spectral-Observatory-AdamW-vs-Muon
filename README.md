# Transformer Spectral Observatory   AdamW vs  Muon  Toward  Loss Curves

A compact educational laboratory for observing **training dynamics inside Transformer weight matrices**.

The accompanying Jupyter notebook trains a tiny Qwen-style causal language model on WikiText-2 with a simple word-level tokenizer and compares **AdamW** with **Muon**. During training it logs train/validation loss and intra-layer matrix diagnostics for attention and FFN projection matrices.


## Core idea

Most training dashboards show only scalar metrics:

- train loss
- validation loss
- perplexity

Those tell us whether optimization works, but not **how the network reorganizes internally**.

This project treats each Transformer matrix as a geometric object evolving through training:

```text
W_0, W_1, W_2, ..., W_t
```

The main thesis is:

> Similar validation loss does not imply similar internal matrix geometry.

AdamW and Muon may reach similar losses while producing different spectra, effective ranks, subspace rotations, gradient anisotropy, and weight-space trajectories.

## Diagnostics included

### Spectral diagnostics

For each tracked weight matrix:

```math
W = U \Sigma V^T
```

The notebook tracks:

- Frobenius norm
- spectral norm
- stable rank
- top-k energy ratio
- singular value curves
- row/column norm outlier ratios

### Stable rank

```math
\mathrm{srank}(W)=\frac{\|W\|_F^2}{\|W\|_2^2}
```

This is a soft rank estimate. Lower stable rank means the matrix energy is concentrated into fewer dominant directions.

### Subspace alignment and drift

The notebook compares dominant singular subspaces between checkpoints:

```math
V_k^{(t)} \quad \text{vs.} \quad V_k^{(t+\Delta)}
```

It records overlap and drift. This shows whether a layer keeps refining the same dominant directions or rotates into new ones.

### Gradient projection ratio

The notebook measures whether the current gradient aligns with the dominant singular directions of the weight matrix:

```math
\langle \nabla_W L, u_i v_i^T \rangle
```

High projection means updates are concentrated in already-dominant directions. Low projection means updates are distributed across weaker or emerging directions.

### Gradient covariance / Fisher-proxy diagnostics

The true Fisher Information Matrix is:

```math
F = \mathbb{E}\left[\nabla_\theta \log p_\theta(y|x)\nabla_\theta \log p_\theta(y|x)^T\right]
```

Computing the full empirical Fisher is expensive. The notebook therefore uses a cheap layer-level proxy based on the singular values of the current matrix gradient `G`:

```math
GG^T \quad \text{and} \quad G^T G
```

It logs:

- trace proxy
- largest eigenvalue proxy
- effective rank proxy
- top-k gradient/Fisher-proxy energy

This is not a full empirical Fisher, but it is useful for observing whether the current update signal is isotropic or concentrated.

### Weight velocity and acceleration

At diagnostic checkpoints:

```math
v_t = W_t - W_{t-\Delta}
```

```math
a_t = v_t - v_{t-\Delta}
```

These finite-difference quantities show how the optimizer moves through weight space.

The notebook logs:

- velocity Frobenius norm
- velocity spectral norm
- acceleration Frobenius norm
- relative acceleration
- velocity cosine

## Why this matters

Two optimizers can produce similar validation loss while creating very different internal structures:

- different spectral concentration
- different effective dimensionality
- different dominant subspaces
- different gradient anisotropy
- different trajectory smoothness

This repository is meant as a small microscope for those differences.

## Files

Expected outputs from the notebook:

```text
loss_curves_adamw_vs_muon.csv
matrix_diagnostics_adamw_vs_muon.csv
```

## Requirements

```bash
pip install torch datasets transformers matplotlib pandas numpy tqdm
```

## Run

```bash
jupyter notebook tiny_qwen_wikitext2_matrix_diagnostics_adamw_muon_enhanced_dynamics.ipynb
```

Run all cells. You can change:

- optimizer
- learning rate
- batch size
- model size
- diagnostic interval
- tracked matrix patterns
- top-k singular directions

## References

- Amari, S. Natural gradient learning and Fisher information geometry.
- Martens, J. and Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*.
- Raghu, M. et al. (2017). *SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability*.
- Morcos, A. et al. (2018). *Insights on Representational Similarity in Neural Networks with Canonical Correlation*.
- Kornblith, S. et al. (2019). *Similarity of Neural Network Representations Revisited*.
- Martin, C. H. and Mahoney, M. W. (2019/2021). *Traditional and Heavy-Tailed Self-Regularization in Neural Network Models* / *Implicit Self-Regularization in Deep Neural Networks*.

## License

MIT


---

# README Addendum: Extended Training-Dynamics Diagnostics

This addendum expands the motivation and interpretation of the extended diagnostics used in the notebook. It can be appended to the main `README.md` under a section such as **Research Notes** or **Extended Diagnostics**.

---

## Why look beyond loss curves?

Training and validation loss are necessary but incomplete observables. They tell us whether the model is improving on the language-modeling objective, but they do not reveal *how* the internal parameters reorganize during optimization.

For a Transformer layer, a projection matrix is not just a table of scalar weights. It is a linear operator:

```math
W: \mathbb{R}^{d_{in}} \rightarrow \mathbb{R}^{d_{out}}
```

During training, this operator changes continuously:

```math
W_0, W_1, W_2, \ldots, W_t
```

The central idea of this project is that two optimizers may produce similar loss curves while following very different trajectories in weight space. Therefore, we track the geometry and dynamics of matrices directly.

---

## 1. Spectral dynamics

The singular value decomposition

```math
W = U \Sigma V^T
```

separates a weight matrix into input directions, output directions, and amplification strengths. The singular values reveal how much the matrix stretches different directions in representation space.

The notebook tracks:

- Frobenius norm
- spectral norm
- stable rank
- top-k spectral energy
- singular-value trajectories

These metrics answer questions such as:

- Is the matrix energy becoming more concentrated?
- Is the layer becoming effectively low-rank?
- Are a few dominant directions taking over?
- Does one optimizer create a more anisotropic operator than another?

This is related to work on empirical spectral densities and heavy-tailed self-regularization in neural-network weight matrices, where trained networks often develop non-random spectral structure rather than remaining close to random matrix baselines.

---

## 2. Stable rank as effective dimensionality

The stable rank is defined as:

```math
\mathrm{srank}(W) = \frac{\|W\|_F^2}{\|W\|_2^2}
```

Unlike the algebraic rank, stable rank is continuous and sensitive to spectral concentration. If one singular value dominates, stable rank decreases. If energy is distributed across many directions, stable rank is larger.

In this notebook, stable rank is used as a proxy for the effective dimensionality of a learned projection.

Interpretation:

- decreasing stable rank: energy concentrates into fewer dominant directions
- increasing stable rank: energy spreads across more directions
- optimizer differences in stable rank: different implicit geometric biases

---

## 3. Subspace drift and alignment

The top singular vectors define dominant input and output subspaces:

```math
V_k^{(t)}, \quad U_k^{(t)}
```

Comparing these subspaces across checkpoints tells us whether training is refining the same directions or discovering new ones.

The notebook tracks a subspace-overlap score between checkpoints. High overlap means the dominant subspace is stable. Low overlap means the layer is rotating into different directions.

This is related in spirit to SVCCA and CKA-style representation-similarity methods, but here the comparison is applied directly to weight-matrix subspaces rather than activations.

Interpretation:

- high alignment: stable dominant directions
- low alignment: rapid representational reorganization
- optimizer-specific alignment: different path geometry through weight space

---

## 4. Gradient projection onto spectral directions

The gradient matrix

```math
G_t = \nabla_W L_t
```

can be decomposed relative to the current singular directions of `W`. The notebook measures how much of the gradient lies in the dominant rank-k spectral subspace:

```math
\frac{\|P_k(G_t)\|_F^2}{\|G_t\|_F^2}
```

where `P_k` is the projection onto the span of the leading singular directions.

This answers:

- Is the optimizer reinforcing already-dominant directions?
- Is the gradient creating new directions?
- Are updates spectrally concentrated or diffuse?

A high projection ratio means updates are mostly aligned with already-important directions. A low ratio means the update signal is spread into weaker or emerging directions.

---

## 5. Fisher / gradient-covariance proxy

The Fisher Information Matrix for a likelihood model is:

```math
F = \mathbb{E}\left[
\nabla_\theta \log p_\theta(y|x)
\nabla_\theta \log p_\theta(y|x)^T
\right]
```

For language models, this object is natural because the model explicitly represents conditional token distributions:

```math
p_\theta(x_{t+1} \mid x_{\leq t})
```

The Fisher measures how sensitive the model distribution is to parameter changes. It is therefore not merely a local curvature diagnostic; it is a geometry induced by the predictive distribution.

The full Fisher is too large to compute for even modest neural networks. The notebook therefore uses a cheap layer-level proxy based on the current gradient matrix. For a matrix gradient `G`, it examines spectral quantities related to:

```math
GG^T \quad \text{or} \quad G^TG
```

This is not a full empirical Fisher. It is a tractable diagnostic for gradient anisotropy and update concentration.

Tracked quantities include:

- gradient/Fisher-proxy trace
- largest eigenvalue proxy
- effective rank proxy
- top-k energy ratio

Interpretation:

- high top eigenvalue: update signal concentrated in one dominant direction
- high effective rank: update signal distributed across many directions
- optimizer differences: different curvature/geometry interaction

This connects to natural-gradient and K-FAC literature, where Fisher structure is used to define more geometry-aware optimization methods.

---

## 6. Weight velocity and acceleration

If we view training as a trajectory through weight space, then the finite-difference velocity is:

```math
v_t = W_t - W_{t-\Delta}
```

and the finite-difference acceleration is:

```math
a_t = v_t - v_{t-\Delta}
```

These are not physical velocity and acceleration, but they are useful trajectory diagnostics.

The notebook tracks:

- velocity Frobenius norm
- velocity spectral norm
- acceleration Frobenius norm
- relative acceleration
- velocity cosine similarity

Interpretation:

- high velocity norm: large movement in weight space
- high acceleration: rapidly changing update direction or magnitude
- high velocity cosine: smooth trajectory
- low or negative velocity cosine: oscillatory or turning behavior

These metrics help distinguish optimizers that produce smooth geometric evolution from those that produce sharper turns or more abrupt spectral changes.

---

## 7. AdamW vs Muon: what should we expect?

This project is not meant to prove that one optimizer is universally better. Instead, it is designed to reveal that different optimizers can induce different internal geometries.

Possible observations:

- AdamW may concentrate energy into fewer dominant directions.
- Muon may produce different norm growth and effective-rank behavior.
- Both optimizers may reach similar validation loss while producing different spectra.
- Differences may appear more clearly in stable rank, subspace drift, and gradient-covariance proxies than in loss alone.

The main message is:

> Similar external performance does not imply similar internal training dynamics.

---

## 8. Limitations

These diagnostics should be interpreted carefully.

- The Fisher proxy is not the full empirical Fisher.
- Weight-space metrics are affected by parameterization and normalization layers.
- Small models may not reproduce all dynamics of large LLMs.
- Word-level tokenization is intentionally simple and educational, not state-of-the-art.
- SVD-based diagnostics can be expensive for larger matrices.
- Weight geometry and activation geometry are related but not identical.

The notebook is therefore best understood as an educational observatory, not as a complete theory of Transformer training.

---

## 9. Suggested future extensions

Potential directions:

- exact per-sample empirical Fisher blocks for selected matrices
- Hessian-vector product diagnostics
- Lanczos estimates of Hessian/Fisher spectra
- activation CKA between checkpoints
- layerwise comparison across attention and FFN blocks
- LoRA-specific spectral dynamics
- quantization-aware spectral diagnostics
- optimizer-state diagnostics for Adam moments and Muon updates
- comparison with SGD, Lion, Adafactor, Sophia, and Shampoo

---

## Additional references

- Amari, S. (1998). Natural gradient works efficiently in learning.
- Martens, J. and Grosse, R. (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature.
- Raghu, M. et al. (2017). SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability.
- Morcos, A. et al. (2018). Insights on Representational Similarity in Neural Networks with Canonical Correlation.
- Kornblith, S. et al. (2019). Similarity of Neural Network Representations Revisited.
- Martin, C. H. and Mahoney, M. W. (2019). Traditional and Heavy-Tailed Self Regularization in Neural Network Models.
- Martens, J. (2020). New insights and perspectives on the natural gradient method.
