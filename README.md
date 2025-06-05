# QuantR

This repository contains a PyTorch example for predicting whether NVDA stock closes higher or lower on the next day. Historical price data is downloaded from Yahoo Finance using the `yfinance` library. Features include returns, moving averages and volume information. A lightweight Transformer model is trained to classify the next day's direction.

## Setup

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Training

Run the training script and follow the prompts to train and/or backtest:

```bash
python src/train.py
```

The script downloads five years of NVDA data, trains a small Transformer on sequences of technical indicators and can optionally backtest the strategy. It automatically uses an NVIDIA GPU if one is available.
