---
title: "Investigating Vanishing Gradients and Early-Layer Trainability Across Network Depth in Deep MLPs on BloodMNIST"
layout: single
author_profile: true
--- 

<p align="center">
  <img src="{{ site.baseurl }}/images/gradient_vanishing.png"
       width="960"
       alt="Hepatic Risk Outcome Prediction"
       style="border-radius: 14px;"/>
</p>

---  

# Investigating Vanishing Gradients and Early-Layer Trainability Across Network Depth in Deep MLPs on BloodMNIST

A controlled empirical study of **depth-dependent weight-gradient attenuation, early-layer trainability, and activation-dependent optimization behavior** in multilayer perceptrons (MLPs) trained on BloodMNIST.

> **Project:** The Effects of Vanishing Gradients in Deep Neural Networks, individual empirical study  
> **Research area:** Deep learning and neural network optimization  
> **Dataset:** BloodMNIST (MedMNIST v2)  
> **Framework:** PyTorch  
> **Experimental scope:** Two controlled stages; 18 unique main runs across three random seeds

## Research overview

Training deep feedforward networks depends on gradients reaching the parameters of earlier layers. Under some architectures and training conditions, repeated transformations during backpropagation can yield very small early-layer gradients, potentially restricting parameter movement and slowing optimization.

This project investigates **how increasing MLP depth is associated with layer-wise weight-gradient magnitudes and early-layer trainability**, then examines how **Sigmoid, Tanh, and ReLU** behave at a fixed depth. BloodMNIST serves as a standardized image-classification testbed; the objective is to analyze neural-network optimization rather than develop a clinical image-classification system.

### Research questions

1. **Network depth:** How does increasing depth affect layer-wise weight-gradient magnitudes and the trainability of early layers in Sigmoid MLPs?
2. **Activation functions:** At a fixed depth and under a common initialization and training protocol, how do Sigmoid, Tanh, and ReLU differ in relative gradient attenuation, optimization progress, and validation performance?

## Experimental design

| Component | Configuration |
|---|---|
| Input | BloodMNIST RGB images, 28 × 28 pixels; flattened to 2,352 features |
| Architecture | Constant-width MLPs with 256 units per hidden layer and eight output logits |
| Stage A — Depth | D2, D5, D10, D20; Sigmoid activation |
| Stage B — Activation | Sigmoid, Tanh, ReLU; fixed D5 architecture |
| Initialization | Xavier Uniform (gain = 1); zero biases |
| Optimization | Plain SGD; learning rate 0.1; momentum 0; weight decay 0 |
| Training | Cross-entropy loss; 20 epochs; batch size 128 |
| Repetitions | Seeds 0, 1, 2; 18 unique main runs, reusing D5–Sigmoid across stages |
| Evaluation | Official BloodMNIST validation split; cross-entropy, accuracy, macro-F1 |

The official dataset partitions contain **11,959 training**, **1,712 validation**, and **3,421 test** images. Channel-wise normalization statistics were estimated from the training split only and applied to all partitions; no data augmentation or additional splitting was performed. The test split was loaded for data verification but **was not used to report classification performance**.

Stage A varied network depth under a fixed Sigmoid–Xavier–SGD configuration. D5 was then selected for Stage B because it exhibited early-layer gradient attenuation while retaining measurable optimization progress. Stage B varied only the hidden-layer activation function; it did not test activation-by-depth interactions.

## What I investigated

- **Gradient propagation across depth:** Measured root-mean-square (RMS) weight gradients for every hidden layer, comparing absolute magnitudes and first-to-last hidden-layer gradient ratios.
- **Early-layer trainability:** Tracked normalized first-layer weight displacement from initialization and training-loss progression.
- **Optimization progress:** Used a predefined mean training-loss threshold of **1.50**, selected following a short learning-rate pilot and fixed before the main experiments.
- **Activation-dependent behavior:** Compared Sigmoid, Tanh, and ReLU at D5 while holding initialization, optimizer, learning rate, data processing, and training duration constant.
- **Run-to-run variability:** Repeated each unique configuration across three random seeds and summarized results as **mean ± sample standard deviation**.
- **Validation performance:** Reported validation cross-entropy, accuracy, and macro-F1 alongside gradient and training measurements.

## Models and methods

`PyTorch` · `Configurable Deep MLPs` · `Backpropagation` · `Sigmoid / Tanh / ReLU` · `Xavier Uniform Initialization` · `SGD` · `Layer-wise Gradient RMS` · `Weight Displacement` · `Multi-Seed Evaluation`

## Experimental results

### Stage A — Network depth under Sigmoid activation

The following results summarize epoch-1 first-to-last hidden-layer **RMS weight-gradient ratios**, epoch-20 normalized first-layer weight displacement, and epoch-20 validation accuracy. Values are the mean across three seeds.

| Hidden layers | Gradient ratio (epoch 1) | First-layer displacement (epoch 20) | Validation accuracy (epoch 20) |
|---|---:|---:|---:|
| D2 | 0.4070 | 0.3279 | 81.23% |
| D5 | 0.006369 | 0.1090 | 46.63% |
| D10 | 4.837 × 10⁻⁶ | 2.740 × 10⁻⁵ | 18.20% |
| D20 | 2.711 × 10⁻¹² | 0.0000* | 18.20% |

The first-to-last hidden-layer gradient ratio decreased sharply with depth. D10 and D20 did not reach the predefined training-loss threshold of 1.50 within the 20-epoch training budget; D2 reached it at epoch 2, while D5 reached it after approximately 14–15 epochs.

*The D20 first-layer displacement was recorded as zero at the available numerical precision. This does not imply mathematically zero parameter updates.*

### Stage B — Activation comparison at D5

All three activation functions used the same D5 architecture, Xavier Uniform initialization, and SGD training protocol. Values are means across three seeds.

| Activation | Gradient ratio (epoch 1) | Validation accuracy (epoch 20) | Validation macro-F1 (epoch 20) |
|---|---:|---:|---:|
| Sigmoid | 0.006369 | 46.63% | 0.3377 |
| Tanh | 0.4750 | **86.21%** | **0.8441** |
| ReLU | 0.4954 | 81.70% | 0.7946 |

Tanh and ReLU exhibited substantially larger relative early-layer weight gradients, greater first-layer parameter displacement, and faster training-loss reduction than Sigmoid under the shared initialization rule. Tanh attained the highest mean validation accuracy and macro-F1, while ReLU had a slightly higher mean relative gradient ratio.

### Experimental figures

The notebook and technical report include three primary figures:

1. **Figure 1 — Gradient magnitude versus hidden-layer position:** Absolute and last-layer-normalized RMS weight gradients for D2, D5, D10, and D20 at epoch 1.
<p align="center">
  <img src="{{ site.baseurl }}/images/figure1_gradient_vs_layer.png"
       width="960"
       alt="Hepatic Risk Outcome Prediction"
       style="border-radius: 14px;"/>
</p>  

3. **Figure 2 — Gradient evolution over training:** A layer-by-epoch heatmap for the D20 Sigmoid network across 20 epochs.  
<p align="center">
  <img src="{{ site.baseurl }}/images/figure2_gradient_evolution_d20.png"
       width="960"
       alt="Hepatic Risk Outcome Prediction"
       style="border-radius: 14px;"/>
</p>
4. **Figure 3 — Training-loss trajectories:** Mean training cross-entropy loss for Stage A depths, including the predefined 1.50 threshold.  
<p align="center">
  <img src="{{ site.baseurl }}/images/figure3_training_loss_curves.png"
       width="960"
       alt="Hepatic Risk Outcome Prediction"
       style="border-radius: 14px;"/>
</p>

## What the project demonstrates

The experiments provide evidence that, **under the tested Sigmoid–Xavier–SGD conditions**, increasing MLP depth is associated with pronounced early-layer weight-gradient attenuation, smaller net first-layer parameter displacement, and slower optimization. The fixed-depth comparison additionally shows that activation choice can substantially change gradient and training behavior under a common initialization rule.

The analysis treats gradient attenuation and trainability as **empirically associated**, not as proof of a unique causal mechanism. In particular, RMS weight gradients are not direct measurements of activation-gradient propagation or network Jacobian properties. The results do not establish that one activation function is universally superior.

## Experimental workflow and reproducibility

1. **Data preparation:** Load official BloodMNIST splits; compute RGB normalization statistics from training images only; verify dimensions and class distributions.
2. **Model construction:** Implement and verify a configurable constant-width MLP with explicit weight and bias initialization.
3. **Training and validation:** Train with cross-entropy and plain SGD; evaluate the full validation split after each epoch.
4. **Gradient diagnostics:** Record weight-gradient RMS values immediately after backpropagation and before the optimizer step; log the first 100 training steps and the first 10 mini-batches of each epoch.
5. **Protocol pilot:** Check numerical stability at D2 and D20; fix the learning rate and training-loss threshold before the main experiments.
6. **Main experiments:** Execute Stage A and Stage B with the shared seed set; aggregate run-level results; generate figures and summary tables.

To reproduce the experiment, install the dependencies recorded in the repository's `requirements.txt`, open the notebook in `notebooks/`, and execute the cells in order. The MedMNIST package downloads BloodMNIST automatically into the locally configured `./data` directory. The notebook writes analysis outputs under `./results`.

The experiment was run with Python 3.13, PyTorch 2.14.1, and MedMNIST 3.0.2 on Apple Silicon using the MPS backend. Numerical results may vary with software versions, floating-point precision, and hardware backends.

**Project materials:** [Experimental notebook](https://github.com/uyentran9/gradient-vanishing/tree/main/notebooks) · [Technical report](https://github.com/uyentran9/gradient-vanishing/tree/main/reports) · [GitHub repository](https://github.com/uyentran9/gradient-vanishing)

## Limitations and future work

- The evidence comes from one low-resolution dataset and constant-width MLPs; the results should not be generalized to CNNs, Transformers, GNNs, or other datasets without further testing.
- Deeper networks also contain more parameters, so depth-related differences cannot be attributed independently of all changes in model capacity.
- Stage B compares activation functions only at D5 and uses a shared Xavier initialization; activation-by-depth and initialization sensitivity were not tested.
- Three random seeds and 20 training epochs provide a limited basis for quantifying variability and longer-term optimization behavior; no formal significance tests were conducted.
- Layer-wise RMS weight gradients and net weight displacement do not isolate the causal mechanisms of gradient attenuation or account for all individual parameter updates.

Potential extensions include controlled initialization comparisons, activation-saturation diagnostics, batch normalization or residual connections, alternative optimizers, and evaluations on additional architectures and datasets.

## Data and intended use

BloodMNIST is a public research dataset from **MedMNIST v2**, derived from peripheral blood-cell microscopy data. The work is an empirical study of neural-network optimization and is **not intended for clinical diagnosis or medical decision-making**. Full dataset attribution and related literature are provided in the technical report.  

## Author

**Victoria Tran (Uyen Tran)**  
Individual project



Thank you for visiting — and for being part of my journey 🌱
