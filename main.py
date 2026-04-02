import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
TARGET_COL = "Yield_kg_hectare"


import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------
# Logging Setup
# -----------------------------------------------------

def setup_logging(base_dir="logs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "run.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(f"Logging initialized in: {run_dir}")

    return run_dir


def save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def log_metrics_table(title: str, results: Dict[str, Dict[str, float]]):
    logging.info("==== " + title + " ====")
    for name, mets in results.items():
        logging.info(
            f"{name:15s} | "
            f"RMSE={mets['RMSE']:.2f} "
            f"MAE={mets['MAE']:.2f} "
            f"R2={mets['R2']:.3f} "
            f"MAPE={mets['MAPE']:.2f}"
        )


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Metrics
# -----------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2), "MAPE": mape(y_true, y_pred)}


# -----------------------------
# Model
# -----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Data utilities
# -----------------------------
@dataclass
class ClientData:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def clean_features(df, feature_cols, client_name):
    logging.info(f"[{client_name}] raw features: {len(feature_cols)}")

    # Replace inf with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    valid = []
    for c in feature_cols:
        non_nan = df[c].notna().sum()
        if non_nan >= 1:
            valid.append(c)
        else:
            logging.info(f"[{client_name}] drop all-NaN column: {c}")

    # Median imputation
    df[valid] = df[valid].fillna(df[valid].median())

    # remove zero variance
    final = []
    for c in valid:
        if df[c].std() > 1e-12:
            final.append(c)
        else:
            logging.info(f"[{client_name}] drop constant column: {c}")

    logging.info(f"[{client_name}] final features: {len(final)}")
    return df, final


def load_client_csv(csv_path: Path, harmonize_yield_units=True):
    df = pd.read_csv(csv_path)

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != TARGET_COL]

    df, feature_cols = clean_features(df, feature_cols, csv_path.stem)

    # --- DO NOT CRASH HERE ---
    if len(feature_cols) == 0:
        logging.warning(f"{csv_path.name}: no valid numeric features; using dummy feature")
        df["__dummy__"] = 0.0
        feature_cols = ["__dummy__"]

    # ---- TARGET ----
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    if harmonize_yield_units:
        y_mean = np.nanmean(y)
        if y_mean < 10:
            y = y * 1000
        elif y_mean < 200:
            y = y * 100

    mask = ~np.isnan(y)
    df = df[mask]
    y = y[mask]

    X = df[feature_cols].to_numpy(dtype=np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    for arr in [X_train, X_val, X_test]:
        np.nan_to_num(arr, copy=False)

    return ClientData(
        name=csv_path.stem,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        feature_names=feature_cols,
    )


def load_all_clients(dataset_dir: Path, harmonize_yield_units=True):
    csvs = sorted(dataset_dir.glob("wheat*_features.csv"))

    clients = [load_client_csv(p, harmonize_yield_units) for p in csvs]

    # ---- UNION OF FEATURES ----
    union = sorted(set().union(*[c.feature_names for c in clients]))
    logging.info(f"Union feature size: {len(union)}")

    aligned = []
    for c in clients:
        Xtr = np.zeros((c.X_train.shape[0], len(union)), dtype=np.float32)
        Xva = np.zeros((c.X_val.shape[0],   len(union)), dtype=np.float32)
        Xte = np.zeros((c.X_test.shape[0],  len(union)), dtype=np.float32)

        for i, f in enumerate(union):
            if f in c.feature_names:
                j = c.feature_names.index(f)
                Xtr[:, i] = c.X_train[:, j]
                Xva[:, i] = c.X_val[:, j]
                Xte[:, i] = c.X_test[:, j]

        aligned.append(
            ClientData(
                name=c.name,
                X_train=Xtr,
                y_train=c.y_train,
                X_val=Xva,
                y_val=c.y_val,
                X_test=Xte,
                y_test=c.y_test,
                feature_names=union,
            )
        )

    return aligned


# -----------------------------
# Training helpers
# -----------------------------
def train_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y.reshape(-1, 1)).to(device)

    n = X_t.shape[0]
    idx = torch.randperm(n, device=device)

    for _ in range(epochs):
        # reshuffle each epoch
        idx = idx[torch.randperm(n, device=device)]
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = X_t[batch_idx]
            yb = y_t[batch_idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    # return training loss on train set
    with torch.no_grad():
        pred = model(X_t)
        loss = loss_fn(pred, y_t).item()
    return float(loss)


@torch.no_grad()
def predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        y = model(X_t).cpu().numpy().reshape(-1)

    # HARD GUARD
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def get_model_weights(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_model_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(w) for k, w in zip(keys, weights)}
    model.load_state_dict(new_state, strict=True)


# -----------------------------
# Flower Client
# -----------------------------
class WheatClient(fl.client.NumPyClient):
    def __init__(self, cdata: ClientData, device: str = "cpu"):
        self.cdata = cdata
        self.device = device
        self.model = MLPRegressor(in_dim=self.cdata.X_train.shape[1]).to(self.device)

    def get_parameters(self, config):
        return get_model_weights(self.model)

    def fit(self, parameters, config):
        set_model_weights(self.model, parameters)
        epochs = int(config.get("local_epochs", 3))
        lr = float(config.get("lr", 1e-3))
        batch_size = int(config.get("batch_size", 32))

        train_loss = train_local(
            self.model,
            self.cdata.X_train,
            self.cdata.y_train,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=self.device,
        )
        num_examples = self.cdata.X_train.shape[0]
        metrics = {"train_loss": train_loss}
        return get_model_weights(self.model), num_examples, metrics

    def evaluate(self, parameters, config):
        set_model_weights(self.model, parameters)
        yhat = predict(self.model, self.cdata.X_val, device=self.device)
        mets = regression_metrics(self.cdata.y_val, yhat)
        # Flower expects (loss, num_examples, metrics)
        # Use RMSE as "loss" for reporting
        return mets["RMSE"], self.cdata.X_val.shape[0], mets


# -----------------------------
# Baselines
# -----------------------------
def run_local_baselines(clients: List[ClientData], device: str = "cpu") -> Dict[str, Dict[str, float]]:
    results = {}
    for c in clients:
        model = MLPRegressor(in_dim=c.X_train.shape[1]).to(device)
        train_local(model, c.X_train, c.y_train, epochs=30, lr=1e-3, batch_size=32, device=device)
        yhat = predict(model, c.X_test, device=device)
        results[c.name] = regression_metrics(c.y_test, yhat)
    return results


def run_centralized_baseline(clients: List[ClientData], device: str = "cpu") -> Dict[str, float]:
    X_train = np.concatenate([c.X_train for c in clients], axis=0)
    y_train = np.concatenate([c.y_train for c in clients], axis=0)
    X_test = np.concatenate([c.X_test for c in clients], axis=0)
    y_test = np.concatenate([c.y_test for c in clients], axis=0)

    model = MLPRegressor(in_dim=X_train.shape[1]).to(device)
    train_local(model, X_train, y_train, epochs=50, lr=1e-3, batch_size=64, device=device)
    yhat = predict(model, X_test, device=device)
    return regression_metrics(y_test, yhat)


# -----------------------------
# Federated runner
# -----------------------------
def run_federated(
    clients: List[ClientData],
    *,
    rounds: int = 80,
    local_epochs: int = 3,
    device: str = "cpu",
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        return WheatClient(clients[idx], device=device).to_client()

    def fit_config(server_round: int) -> Dict[str, str]:
        return {
            "local_epochs": str(local_epochs),
            "lr": "0.001",
            "batch_size": "32",
        }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(clients),
        min_evaluate_clients=len(clients),
        min_available_clients=len(clients),
        on_fit_config_fn=fit_config,
    )

    # NOTE: `flwr.simulation.start_simulation()` is deprecated and will
    # be removed in future versions. The recommended replacement is the
    # `flwr run` CLI. For now we temporarily raise the `flwr` logger level
    # to suppress the deprecation warning while still running the local
    # simulation programmatically.
    flwr_logger = logging.getLogger("flwr")
    prev_level = flwr_logger.level
    flwr_logger.setLevel(logging.ERROR)
    try:
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(clients),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    finally:
        flwr_logger.setLevel(prev_level)

    # Evaluate final global model on:
    # - pooled test
    # - per-client test
    # We reconstruct a model and load the final aggregated parameters from history.
    # Flower history does not always store final parameters; easiest is to rerun a final round fetch:
    # Instead, we run a "central evaluation" by training a model from last distributed parameters:
    #
    # Practical approach: run one more server-side evaluation by asking one client for parameters after training:
    # Not available here reliably; so we repeat a lightweight FL: store params via custom strategy would be ideal.
    #
    # For simplicity in a prototype: we approximate by taking the last "distributed" parameters from one client
    # by re-running a single client init and using its current weights (post-simulation not accessible).
    #
    # Better: use a custom strategy to keep latest parameters.
    #
    # We'll implement custom strategy now:
    return {}, {}  # replaced below


class FedAvgWithParams(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            params, _ = aggregated
            self.latest_parameters = params
        return aggregated


def run_federated_with_params(
    clients: List[ClientData],
    *,
    rounds: int = 80,
    local_epochs: int = 3,
    device: str = "cpu",
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        return WheatClient(clients[idx], device=device).to_client()

    def fit_config(server_round: int) -> Dict[str, str]:
        return {
            "local_epochs": str(local_epochs),
            "lr": "0.001",
            "batch_size": "32",
        }

    strategy = FedAvgWithParams(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(clients),
        min_evaluate_clients=len(clients),
        min_available_clients=len(clients),
        on_fit_config_fn=fit_config,
    )

    # Suppress Flower deprecation warning (see note above).
    flwr_logger = logging.getLogger("flwr")
    prev_level = flwr_logger.level
    flwr_logger.setLevel(logging.ERROR)
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(clients),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    finally:
        flwr_logger.setLevel(prev_level)

    if strategy.latest_parameters is None:
        raise RuntimeError("No aggregated parameters captured from FL run")

    # Convert Flower Parameters -> list[np.ndarray]
    ndarrays = fl.common.parameters_to_ndarrays(strategy.latest_parameters)

    # Evaluate pooled + per-client
    in_dim = clients[0].X_train.shape[1]
    global_model = MLPRegressor(in_dim=in_dim).to(device)
    set_model_weights(global_model, ndarrays)

    # pooled test
    X_test = np.concatenate([c.X_test for c in clients], axis=0)
    y_test = np.concatenate([c.y_test for c in clients], axis=0)
    yhat = predict(global_model, X_test, device=device)
    pooled = regression_metrics(y_test, yhat)

    per_client = {}
    for c in clients:
        yhat_c = predict(global_model, c.X_test, device=device)
        per_client[c.name] = regression_metrics(c.y_test, yhat_c)

    return pooled, per_client


# -----------------------------
# Pretty printing
# -----------------------------
def print_table(title: str, rows: List[Tuple[str, Dict[str, float]]]) -> None:
    print("\n" + title)
    cols = ["RMSE", "MAE", "R2", "MAPE"]
    header = f"{'Name':<20} " + " ".join([f"{c:>12}" for c in cols])
    print(header)
    print("-" * len(header))
    for name, mets in rows:
        print(
            f"{name:<20} "
            + " ".join([f"{mets[c]:12.4f}" for c in cols])
        )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_harmonize_yield", action="store_true")
    args = parser.parse_args()

    # ---- INIT LOGGING ----
    run_dir = setup_logging()

    config = vars(args)
    save_json(config, run_dir / "config.json")

    set_seed(SEED)

    harmonize = not args.no_harmonize_yield

    logging.info("Loading clients...")
    clients = load_all_clients(Path(args.dataset_dir), harmonize)

    logging.info(f"Clients: {len(clients)}")
    logging.info(f"Features: {clients[0].X_train.shape[1]}")
    logging.info(f"Harmonize yield: {harmonize}")

    # ---- LOCAL BASELINES ----
    logging.info("Running LOCAL baselines...")
    local_res = run_local_baselines(clients, device=args.device)

    log_metrics_table("LOCAL MODELS", local_res)
    save_json(local_res, run_dir / "metrics_local.json")

    # ---- CENTRALIZED ----
    logging.info("Running CENTRALIZED model...")
    cent = run_centralized_baseline(clients, device=args.device)

    log_metrics_table("CENTRALIZED", {"centralized": cent})
    save_json(cent, run_dir / "metrics_centralized.json")

    # ---- FEDERATED ----
    logging.info("Running FEDERATED learning...")
    pooled_fl, per_client_fl = run_federated_with_params(
        clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        device=args.device,
    )

    log_metrics_table("FEDERATED POOLED", {"fedavg": pooled_fl})
    log_metrics_table("FEDERATED PER CLIENT", per_client_fl)

    save_json(pooled_fl, run_dir / "metrics_federated_pooled.json")
    save_json(per_client_fl, run_dir / "metrics_federated_clients.json")

    logging.info("Experiment finished.")


if __name__ == "__main__":
    main()
