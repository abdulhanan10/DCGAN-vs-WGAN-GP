# DCGAN vs WGAN-GP — Tackling Mode Collapse in GANs

> **Course:** AI4009 — Generative AI | Assignment 3, Question 1  
> **Authors:** 22F-8762 · 22F-3275  
> **Framework:** PyTorch · Gradio · Kaggle Datasets

---

## Overview

This notebook implements and compares two GAN architectures to investigate and mitigate **mode collapse** — one of the most common failure modes in generative adversarial training:

- **DCGAN** — Deep Convolutional GAN with binary cross-entropy loss
- **WGAN-GP** — Wasserstein GAN with Gradient Penalty for stable training

Both models are trained on anime face and Pokémon sprite datasets and evaluated side-by-side on visual quality and diversity metrics.

---

## Architecture

### DCGAN
| Component | Details |
|---|---|
| Generator | 5-layer ConvTranspose2d · 100→64×64 · BatchNorm + ReLU · Tanh output |
| Discriminator | 5-layer Conv2d · LeakyReLU(0.2) · Sigmoid output |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam (lr=0.0002, β=(0.5, 0.999)) |

### WGAN-GP (Critic)
| Component | Details |
|---|---|
| Generator | Same architecture as DCGAN |
| Critic | Conv2d stack — no Sigmoid, raw score output |
| Loss | Wasserstein distance + Gradient Penalty (λ=10) |
| Critic updates per G step | 5 |

---

## Dataset

Downloaded automatically via `kagglehub`:
- [`jackemartin/pokemon-sprites`](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites)
- [`soumikrakshit/anime-faces`](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)

Images are resized to **64×64**, normalized to `[-1, 1]`. Subset size: 8,000 images.

---

## Training Details

| Config | Value |
|---|---|
| Image size | 64×64 |
| Noise vector (NZ) | 100 |
| Batch size | 64 |
| Epochs | 50 (each model) |
| Learning rate | 0.0002 |
| DCGAN labels | Real=0.9 (label smoothing), Fake=0.0 |
| WGAN-GP λ | 10 |
| Critic iterations | 5 per generator step |
| Mixed precision | AMP (GradScaler) |

Checkpoints saved every 100 steps and at end of every epoch to `checkpoints/`.

---

## Notebook Structure

```
1. Install Dependencies
2. Imports & Configuration
3. Data Preparation
4. Model Architectures (DCGenerator, DCDiscriminator, WGenerator, WCritic)
5. DCGAN Training Loop
6. DCGAN — Logs & Sample Visualizations
7. WGAN-GP Setup (Gradient Penalty, Critic)
8. WGAN-GP Training Loop
9. WGAN-GP — Logs & Sample Visualizations
10. Side-by-Side Comparison (DCGAN vs WGAN-GP)
11. Quantitative Evaluation (Pixel Diversity, FID, IS)
12. Gradio App Deployment
```

---

## Evaluation

| Metric | Description |
|---|---|
| Pixel Diversity | Mean pairwise pixel std across generated samples |
| FID (optional) | Fréchet Inception Distance via `torchmetrics` |
| Loss curves | Generator vs Discriminator/Critic per epoch |

---

## Gradio Demo

An interactive Gradio app is included with two tabs:

- **Single Model** — choose DCGAN or WGAN-GP, set number of images and seed
- **Compare Both** — generate from both models side-by-side on the same noise

```python
demo.launch(share=True)  # generates a public link
```

### How to deploy on Gradio (Hugging Face Spaces)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) → SDK: **Gradio**
2. Upload your notebook (converted to `app.py`) and saved model checkpoints
3. Add a `requirements.txt`:
   ```
   torch
   torchvision
   gradio
   numpy
   Pillow
   ```
4. In `app.py`, replace `demo.launch(share=True)` with just `demo.launch()`
5. Push to the Space — it auto-deploys

### How to deploy on Streamlit

1. Convert the Gradio app to a `app_streamlit.py`:
   ```python
   import streamlit as st
   import torch
   from torchvision.utils import make_grid
   from PIL import Image
   import numpy as np

   st.title("DCGAN vs WGAN-GP Image Generator")

   model_choice = st.selectbox("Choose model", ["DCGAN", "WGAN-GP"])
   num_images = st.slider("Number of images", 1, 16, 8)
   seed = st.slider("Random seed", 0, 9999, 42)

   if st.button("Generate"):
       torch.manual_seed(seed)
       # load netG_dc or netG_wgan from checkpoint
       # z = torch.randn(num_images, 100, 1, 1)
       # imgs = netG(z)
       st.image(pil_output)
   ```
2. Install Streamlit: `pip install streamlit`
3. Run locally: `streamlit run app_streamlit.py`
4. Deploy to [streamlit.io/cloud](https://streamlit.io/cloud) by connecting your GitHub repo

---

## Requirements

```
torch
torchvision
gradio
matplotlib
numpy
Pillow
scikit-image
kagglehub
torchmetrics  # optional, for FID
```

---

## Results

| Model | Pixel Diversity | Training Stability | Mode Collapse Risk |
|---|---|---|---|
| DCGAN | — | Moderate | Higher |
| WGAN-GP | — | High | Lower |

*Fill in diversity scores after training.*

---

## Key Takeaways

- WGAN-GP replaces the discriminator with a **critic** that outputs raw scores instead of probabilities, making the loss a proper distance metric
- The **gradient penalty** enforces the Lipschitz constraint without weight clipping, leading to more stable gradients
- DCGAN is faster to train but more prone to mode collapse on limited datasets
- Anti-blur tweaks (asymmetric LR: G=0.0001, D=0.0004) improve DCGAN sharpness significantly
