# PhyDA-India 🌧️

**Physics-guided Deep Learning Data Assimilation for Indian Weather Prediction**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

PhyDA-India is a physics-guided deep learning framework for weather data assimilation over the Indian subcontinent, built on top of the base PhyDA architecture (arXiv:2505.12882). It integrates ERA5 reanalysis data (2022–2024) with deep learning to produce accurate, physically consistent weather analyses.

This work introduces two novel contributions tailored for Indian weather patterns:

1. **Indian Monsoon Physics Constraint (PRDO)** — Season-specific thermodynamic laws enforced during training to ensure physical consistency across monsoon, winter, summer, and post-monsoon seasons.
2. **4-Head Seasonal Attention Mechanism** — Integrated into the Variational Recurrent Encoder (VRE) to capture season-specific spatio-temporal dependencies.

---

## Key Results

| Metric | Value |
|--------|-------|
| Overall Skill Score | **91.65%** |
| Best Seasonal Score | 92.56% |
| Validation Loss (Epoch 19/20) | 33.5844 |

### Validation Events
- 🌊 **Kerala Floods** — August 2024
- 🌀 **Cyclone Fengal** — November 2024

---

## Architecture
ERA5 Input (Temperature, Humidity, Wind, Pressure)

↓

Variational Autoencoder (VAE Encoder)

↓

4-Head Seasonal Attention (VRE)

↓

Indian Monsoon Physics Constraint (PRDO)

↓

Trust-weighted Data Assimilation Blend

↓

Analysed Weather Field (Xa)
---

## Dataset

- **Source:** ERA5 Reanalysis — Copernicus Climate Data Store (CDS)
- **Domain:** Indian subcontinent
- **Period:** 2022–2024 (36 monthly files, ~5.31 GB)
- **Variables:** Temperature, Specific Humidity, U/V Wind Components, Surface Pressure

---

## Project Structure
PhyDA-India/

├── models/          # VAE and VRE model definitions

├── utils/           # Data loading, normalization, helper functions

├── results/         # Skill scores, output plots

├── .gradio/         # Gradio app config

├── train.py         # Training script

└── .gitignore
---

## Getting Started

```bash
git clone https://github.com/dharshinigs14-max/PhyDa-India.git
cd PhyDa-India
pip install -r requirements.txt
python train.py
```

---

## Authors

- **Dharshini G** — Chennai Institute of Technology
- **Gopika D** — Chennai Institute of Technology
- **Supervisor:** Dr. Venkatesan, NIT Puducherry

---

## References

Based on PhyDA: [arXiv:2505.12882](https://arxiv.org/abs/2505.12882)
