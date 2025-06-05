# QuantR

This repository contains a PyTorch example for predicting whether NVDA stock closes higher or lower on the next day. Historical price data is downloaded from Yahoo Finance using the `yfinance` library. Features include returns, moving averages and volume information. A lightweight Transformer model is trained to classify the next day's direction.

## Setup

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python src/train.py
```

This downloads the last five years of NVDA data and trains a small Transformer using sequences of technical features.
