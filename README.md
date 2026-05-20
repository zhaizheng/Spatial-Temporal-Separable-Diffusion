# Spatial–Temporal Separable Diffusion for Efficient Generative Modeling

Official implementation of **Spatial–Temporal Separable Diffusion for Efficient Generative Modeling** by Zheng Zhai and Zhuan Liang.

---

## Overview

Diffusion models have achieved remarkable success in image and video generation, but their computational and optimization complexity remains a major challenge. This work proposes a **spatial–temporal separable parameterization** for diffusion models that explicitly disentangles spatial representations and temporal dynamics.

The proposed predictor is formulated as

\[
g_{\theta,\phi}(x,t)=w_{\phi}(t)^\top f_{\theta}(x),
\]

where:

- \(f_{\theta}(x)\) extracts spatial representations,
- \(w_{\phi}(t)\) models temporal coefficients,
- the final prediction is obtained through their interaction.

This decomposition leads to:

- efficient computation,
- reduced parameter coupling,
- structured Hessian geometry,
- improved optimization properties,
- enhanced interpretability of diffusion dynamics.
