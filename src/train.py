import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class StockSequenceDataset(Dataset):
    """Dataset returning a sequence of features and a binary label."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


class TransformerClassifier(nn.Module):
    """Small transformer for sequence classification."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.cls_head(x).squeeze(-1)


def download_data(symbol: str = "NVDA", period: str = "5y") -> pd.DataFrame:
    """Download historical data from Yahoo Finance."""
    return yf.download(symbol, period=period)


def create_sequences(df: pd.DataFrame, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sequences of features, labels, and future returns."""
    df = df.copy()
    df["Return"] = df["Adj Close"].pct_change()
    df["MA5"] = df["Adj Close"].rolling(window=5).mean()
    df["MA10"] = df["Adj Close"].rolling(window=10).mean()
    df["VolumeNorm"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
    df["Target"] = (df["Adj Close"].shift(-1) > df["Adj Close"]).astype(int)
    df["NextReturn"] = df["Adj Close"].pct_change().shift(-1)
    df = df.dropna()

    feature_cols = ["Return", "MA5", "MA10", "VolumeNorm"]
    data = df[feature_cols].values
    targets = df["Target"].values
    future_returns = df["NextReturn"].values

    sequences = []
    labels = []
    returns = []
    for i in range(len(df) - seq_len):
        sequences.append(data[i : i + seq_len])
        labels.append(targets[i + seq_len - 1])
        returns.append(future_returns[i + seq_len - 1])

    return np.array(sequences), np.array(labels), np.array(returns)


def scale_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X: np.ndarray, y: np.ndarray, seq_len: int, epochs: int = 20, batch_size: int = 64, lr: float = 1e-3, device: torch.device | str = "cpu") -> Tuple[nn.Module, StandardScaler]:
    """Train the transformer classifier and return the model and scaler."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, scaler = scale_data(X_train, X_test)

    train_ds = StockSequenceDataset(X_train, y_train)
    test_ds = StockSequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = TransformerClassifier(input_dim=X.shape[-1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                predicted = (outputs > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Test Acc: {acc:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_dim": X.shape[-1],
        "seq_len": seq_len,
    }, "model_checkpoint.pth")

    return model, scaler


def load_checkpoint(device: torch.device | str = "cpu") -> Tuple[nn.Module, StandardScaler, int]:
    """Load a saved model and scaler."""
    ckpt = torch.load("model_checkpoint.pth", map_location=device)
    model = TransformerClassifier(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    return model, scaler, ckpt["seq_len"]


def backtest(df: pd.DataFrame, model: nn.Module, scaler: StandardScaler, seq_len: int, device: torch.device | str = "cpu") -> None:
    """Simple backtesting by applying the model to historical data."""
    X, _, returns = create_sequences(df, seq_len=seq_len)
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X.shape)

    dataset = StockSequenceDataset(X_scaled, np.zeros(len(X_scaled)))
    loader = DataLoader(dataset, batch_size=64)

    preds = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).cpu().numpy()
            preds.extend(out)
    preds = np.array(preds)
    directions = np.where(preds > 0.5, 1.0, -1.0)
    strategy_returns = directions * returns[: len(directions)]
    cumulative = (1 + strategy_returns).cumprod()
    print(f"Backtest final cumulative return: {cumulative[-1]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stock direction transformer")
    parser.add_argument("--symbol", default="NVDA", help="Ticker symbol")
    parser.add_argument("--period", default="5y", help="Historical period to download")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_prompt = input("Train=True/False: ").strip().lower()
    backtest_prompt = input("BackTest=True/False: ").strip().lower()
    do_train = train_prompt in {"true", "t", "yes", "y"}
    do_backtest = backtest_prompt in {"true", "t", "yes", "y"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = download_data(symbol=args.symbol, period=args.period)

    if do_train:
        X, y, _ = create_sequences(df, seq_len=args.seq_len)
        model, scaler = train_model(
            X,
            y,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
    else:
        model, scaler, seq_len_loaded = load_checkpoint(device=device)
        args.seq_len = seq_len_loaded

    if do_backtest:
        backtest(df, model, scaler, seq_len=args.seq_len, device=device)


if __name__ == "__main__":
    main()
