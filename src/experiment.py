import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models import DTMLModel, LSTMModel, MASTERModel, RAGFormerModel, TransformerModel
from src.reproducibility import apply_reproducibility_defaults


BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_REGISTRY = {
    "ragformer": RAGFormerModel,
    "lstm": LSTMModel,
    "dtml": DTMLModel,
    "transformer": TransformerModel,
    "master": MASTERModel,
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_taxwar_fsi_dataframe() -> pd.DataFrame:
    df_index = pd.read_csv(BASE_DIR / "data" / "taxwar" / "index.csv")
    df_index["Date"] = pd.to_datetime(df_index["Date"], errors="coerce").dt.to_period("M")

    df_fsi = pd.read_csv(BASE_DIR / "data" / "fsi" / "fsi18_optimized.csv", parse_dates=["Date"])
    df_fsi["Date"] = df_fsi["Date"].dt.to_period("M")

    df = pd.merge(df_index, df_fsi[["Date", "FSI_Raw"]], on="Date", how="inner")
    return df.sort_values("Date").reset_index(drop=True)


def split_mask(series: pd.Series, start: str, end: str) -> pd.Series:
    start_p = pd.Period(start, freq="M")
    end_p = pd.Period(end, freq="M")
    return (series >= start_p) & (series <= end_p)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": rmse,
    }


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_columns: List[str]
    seq_len: int
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    features_scaled: np.ndarray
    targets_scaled: np.ndarray
    train_feature_pool: np.ndarray
    split_indices: Dict[str, np.ndarray]


class WindowDataset(Dataset):
    def __init__(self, features_scaled: np.ndarray, targets_scaled: np.ndarray, target_indices: np.ndarray, seq_len: int):
        self.features_scaled = features_scaled
        self.targets_scaled = targets_scaled
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_idx = int(self.target_indices[idx])
        start_idx = target_idx - self.seq_len
        window = self.features_scaled[start_idx:target_idx]
        target = self.targets_scaled[target_idx]
        return torch.from_numpy(window).float(), torch.from_numpy(target).float()


def prepare_data(config: Dict, model_name: str) -> PreparedData:
    df = load_taxwar_fsi_dataframe()
    feature_columns = [c for c in df.columns if c not in ["Date", "FSI_Raw"]]
    model_cfg = config["models"][model_name]
    seq_len = int(model_cfg.get("seq_len", config["splits"]["history_window_months"]))

    train_mask = split_mask(df["Date"], config["splits"]["train"]["start"], config["splits"]["train"]["end"])
    val_mask = split_mask(df["Date"], config["splits"]["validation"]["start"], config["splits"]["validation"]["end"])
    test_mask = split_mask(df["Date"], config["splits"]["test"]["start"], config["splits"]["test"]["end"])

    feature_values = np.nan_to_num(df[feature_columns].values.astype(np.float32), nan=0.0)
    target_values = np.nan_to_num(df["FSI_Raw"].values.astype(np.float32).reshape(-1, 1), nan=0.0)

    scaler_X = StandardScaler().fit(feature_values[train_mask.values])
    scaler_y = StandardScaler().fit(target_values[train_mask.values])

    features_scaled = scaler_X.transform(feature_values).astype(np.float32)
    targets_scaled = scaler_y.transform(target_values).astype(np.float32)

    all_target_indices = np.arange(len(df))
    eligible = all_target_indices >= seq_len

    split_indices = {
        "train": all_target_indices[eligible & train_mask.values],
        "validation": all_target_indices[eligible & val_mask.values],
        "test": all_target_indices[eligible & test_mask.values],
    }

    return PreparedData(
        df=df,
        feature_columns=feature_columns,
        seq_len=seq_len,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        features_scaled=features_scaled,
        targets_scaled=targets_scaled,
        train_feature_pool=features_scaled[train_mask.values],
        split_indices=split_indices,
    )


def build_model(model_name: str, config: Dict, input_size: int) -> nn.Module:
    model_cfg = dict(config["models"][model_name])
    model_cfg["input_size"] = input_size

    if model_name == "dtml":
        model_cfg.setdefault("n_time", int(model_cfg.get("seq_len", 12)))
        model_cfg.pop("seq_len", None)
    else:
        model_cfg.pop("seq_len", None)

    return MODEL_REGISTRY[model_name](**model_cfg)


def attach_retrieval_pool(model: nn.Module, pool: np.ndarray, device: torch.device) -> None:
    if hasattr(model, "set_historical_data"):
        pool_tensor = torch.from_numpy(pool).float().to(device)
        if hasattr(model, "k_neighbors"):
            model.k_neighbors = max(1, min(int(model.k_neighbors), int(pool_tensor.shape[0])))
        model.set_historical_data(pool_tensor)


def create_dataloaders(prepared: PreparedData, batch_size: int) -> Dict[str, DataLoader]:
    loaders = {}
    for split_name, indices in prepared.split_indices.items():
        dataset = WindowDataset(prepared.features_scaled, prepared.targets_scaled, indices, prepared.seq_len)
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loaders


def run_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device, train: bool, grad_clip_norm: float = 0.0) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    preds = []
    targets = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = model(batch_x)
            output = output.view(-1, 1)
            loss = criterion(output, batch_y)

            if train:
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        preds.append(output.detach().cpu().numpy())
        targets.append(batch_y.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0).reshape(-1)
    y_true = np.concatenate(targets, axis=0).reshape(-1)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def predict_split(model: nn.Module, loader: DataLoader, device: torch.device, scaler_y: StandardScaler) -> Dict[str, np.ndarray]:
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            output = model(batch_x).view(-1, 1).cpu().numpy()
            preds.append(output)
            targets.append(batch_y.cpu().numpy())

    pred_scaled = np.concatenate(preds, axis=0)
    true_scaled = np.concatenate(targets, axis=0)
    pred = scaler_y.inverse_transform(pred_scaled).reshape(-1)
    true = scaler_y.inverse_transform(true_scaled).reshape(-1)
    return {"y_pred": pred, "y_true": true}


def save_artifacts(
    model: nn.Module,
    model_name: str,
    prepared: PreparedData,
    config: Dict,
    run_name: str,
    metrics: Dict[str, Dict[str, float]],
    feature_columns: List[str],
) -> Dict[str, str]:
    saved_models_dir = BASE_DIR / "saved_models"
    scalers_dir = BASE_DIR / "scalers"
    metrics_dir = BASE_DIR / "artifacts" / "metrics"

    saved_models_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scaler_X_path = scalers_dir / f"scaler_X_{run_name}.pkl"
    scaler_y_path = scalers_dir / f"scaler_y_{run_name}.pkl"
    checkpoint_path = saved_models_dir / f"{run_name}.pth"
    metrics_path = metrics_dir / f"{run_name}.json"

    joblib.dump(prepared.scaler_X, scaler_X_path)
    joblib.dump(prepared.scaler_y, scaler_y_path)

    model_cfg = dict(config["models"][model_name])
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "model_params": {k: v for k, v in model_cfg.items() if k != "seq_len"},
        "input_size": int(len(feature_columns)),
        "seq_len": int(prepared.seq_len),
        "feature_columns": feature_columns,
        "scaler_X_path": str(scaler_X_path.relative_to(BASE_DIR)),
        "scaler_y_path": str(scaler_y_path.relative_to(BASE_DIR)),
        "metrics": metrics,
        "config_path": "configs/reproducibility.json",
    }
    torch.save(checkpoint, checkpoint_path)

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "checkpoint_path": str(checkpoint_path),
        "scaler_X_path": str(scaler_X_path),
        "scaler_y_path": str(scaler_y_path),
        "metrics_path": str(metrics_path),
    }


def bootstrap_experiment() -> Dict:
    return apply_reproducibility_defaults()
