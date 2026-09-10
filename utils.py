import csv
import os
import random
from typing import Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda"):
    return torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, value, n: int = 1):
        self.val = float(value)
        self.sum += self.val * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_metric: float,
    path: str,
    extra: Optional[Dict] = None,
):
    ensure_dir(os.path.dirname(path))
    state = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
    map_location: str = "cpu",
    load_optimizer: bool = True,
):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    model_state = state.get("model", state)
    incompatible = model.load_state_dict(model_state, strict=strict)
    if not strict:
        print("Missing keys:", list(incompatible.missing_keys))
        print("Unexpected keys:", list(incompatible.unexpected_keys))
    if optimizer is not None and load_optimizer and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except ValueError as exc:
            print("Optimizer state skipped:", exc)
    return int(state.get("epoch", 0)), float(state.get("best_metric", 0.0))


class CSVLogger:
    def __init__(self, csv_path: str, fieldnames):
        self.csv_path = csv_path
        self.fieldnames = list(fieldnames)
        ensure_dir(os.path.dirname(csv_path))
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def write(self, row: Dict):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
