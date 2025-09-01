# Geometry-Aware Learning (GAL)

This project implements a geometry-aware deep learning framework that trains deep networks layer by layer. The training objective balances intra-class compactness with inter-class separation to improve adversarial robustness.

## Features
- Local layer-wise training without global error signals.
- Geometry-aware loss (FDBLoss) encouraging separable representations.
- Utilities for adversarial attacks and evaluation.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib

Install dependencies with:
```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

## Usage
Training on MNIST with default settings:
```bash
python main.py
```
Models and logs are saved in the `model` directory and `experiment_log.txt` respectively.

## Project Structure
- `main.py` – end-to-end training script.
- `models_struc/` – network layer and readout definitions.
- `loss/` – geometry-aware loss and adversarial attack utilities.
- `train/` – training loop for individual layers.
- `utils.py` – data loading, preprocessing, visualization and evaluation helpers.

## References
The approach is inspired by geometry-aware learning principles that promote manifold smoothness and adversarial robustness.
