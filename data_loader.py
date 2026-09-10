"""HSI/MSI dataset loader for the clean S2Diff-MH baseline.

Important: LR-HSI is NOT generated here.  Innovation 1 owns the observation
operator, so training/evaluation obtains LR-HSI from ProgressiveDegradation.
This prevents dataset/model degradation mismatch.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from srf_utils import (
    build_srf_weights,
    hsi_to_msi_numpy,
    load_hsi_wavelengths,
    sensor_protocol,
)

try:
    import scipy.io as scio
except ImportError:
    scio = None
try:
    import h5py
except ImportError:
    h5py = None
try:
    import hdf5storage
except ImportError:
    hdf5storage = None


DATASET_SPECS = {
    "PaviaU": {"file": "PaviaU.mat", "keys": ["paviaU", "PaviaU", "img", "data"]},
    "Houston13": {"file": "Houston13.mat", "keys": ["Houston13", "Houston_HSI", "data", "img"]},
    "Chikusei": {"file": "Chikusei.mat", "keys": ["chikusei", "Chikusei", "img", "data"]},
}


def _extract_cube(value):
    if isinstance(value, dict):
        for item in value.values():
            out = _extract_cube(item)
            if out is not None:
                return out
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            out = _extract_cube(item)
            if out is not None:
                return out
        return None
    if not isinstance(value, np.ndarray):
        return None
    arr = value
    while isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.reshape(-1)[0])
    if isinstance(arr, np.ndarray) and arr.dtype.names:
        for field in arr.dtype.names:
            out = _extract_cube(arr[field])
            if out is not None:
                return out
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
        return arr
    return None


def _fix_hsi_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 3:
        raise ValueError(f"HSI must be 3-D, got {arr.shape}")
    # Typical MATLAB v7.3 readers may expose CxHxW or CxWxH.
    if arr.shape[0] <= 256 and arr.shape[1] > 256 and arr.shape[2] > 256:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[1] <= 256 and arr.shape[0] > 256 and arr.shape[2] > 256:
        arr = np.transpose(arr, (0, 2, 1))
    return arr.astype(np.float32)


def read_hsi_mat(path: str, candidate_keys: List[str]) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    errors = []
    if hdf5storage is not None:
        try:
            mapping = hdf5storage.loadmat(path)
            for key in candidate_keys + list(mapping.keys()):
                if key in mapping:
                    cube = _extract_cube(mapping[key])
                    if cube is not None:
                        return _fix_hsi_shape(cube)
        except Exception as exc:
            errors.append(f"hdf5storage: {exc}")
    if scio is not None:
        try:
            mapping = scio.loadmat(path)
            for key in candidate_keys + list(mapping.keys()):
                if key in mapping:
                    cube = _extract_cube(mapping[key])
                    if cube is not None:
                        return _fix_hsi_shape(cube)
        except Exception as exc:
            errors.append(f"scipy.io: {exc}")
    if h5py is not None:
        try:
            with h5py.File(path, "r") as handle:
                found = []
                def visitor(name, obj):
                    if not found and isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                        found.append(np.asarray(obj))
                handle.visititems(visitor)
                if found:
                    return _fix_hsi_shape(found[0])
        except Exception as exc:
            errors.append(f"h5py: {exc}")
    raise RuntimeError("Unable to load HSI cube. " + " | ".join(errors))


def normalize_hsi(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def crop_to_scale(img: np.ndarray, scale_ratio: int) -> np.ndarray:
    h, w, _ = img.shape
    return img[: h // scale_ratio * scale_ratio, : w // scale_ratio * scale_ratio]


def _center_test_rect(h: int, w: int, test_size: int) -> Tuple[int, int, int, int]:
    top = max((h - test_size) // 2, 0)
    left = max((w - test_size) // 2, 0)
    return top, left, min(top + test_size, h), min(left + test_size, w)


def _intersects(a, b) -> bool:
    t1, l1, b1, r1 = a
    t2, l2, b2, r2 = b
    return not (r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1)


def _patch_coords(h, w, patch_size, stride, test_rect, split):
    if split == "test":
        top, left, bottom, right = test_rect
        if bottom - top < patch_size or right - left < patch_size:
            top, left = max((h - patch_size) // 2, 0), max((w - patch_size) // 2, 0)
        return [(top, left)]
    coords = []
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            rect = (top, left, top + patch_size, left + patch_size)
            if not _intersects(rect, test_rect):
                coords.append((top, left))
    return coords


class HSIHSRDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        srf_weights: np.ndarray,
        patch_size: int,
        stride: int,
        split: str,
        test_size: int = 128,
        augment: bool = True,
    ):
        self.image = image
        self.srf_weights = srf_weights
        self.patch_size = int(patch_size)
        self.split = split
        self.augment = bool(augment and split == "train")
        h, w, _ = image.shape
        self.coords = _patch_coords(
            h, w, self.patch_size, int(stride), _center_test_rect(h, w, test_size), split
        )
        if not self.coords:
            raise RuntimeError("No training patches were generated; reduce patch/test size")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        top, left = self.coords[index]
        gt = self.image[top : top + self.patch_size, left : left + self.patch_size].copy()
        if self.augment:
            if random.random() < 0.5:
                gt = np.flip(gt, axis=0)
            if random.random() < 0.5:
                gt = np.flip(gt, axis=1)
            if random.random() < 0.5:
                gt = np.rot90(gt, k=random.randint(1, 3), axes=(0, 1))
            gt = np.ascontiguousarray(gt)
        hr_msi = hsi_to_msi_numpy(gt, self.srf_weights)
        return {
            "gt": torch.from_numpy(gt).permute(2, 0, 1).contiguous().float(),
            "hr_msi": torch.from_numpy(hr_msi).permute(2, 0, 1).contiguous().float(),
        }


def build_loaders(cfg):
    if cfg.dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset={cfg.dataset!r}")
    spec = DATASET_SPECS[cfg.dataset]
    image = normalize_hsi(read_hsi_mat(os.path.join(cfg.data_root, spec["file"]), spec["keys"]))
    image = crop_to_scale(image, cfg.scale_ratio)
    protocol = sensor_protocol(cfg.dataset)
    wavelengths = load_hsi_wavelengths(protocol["wavelength_path"], image.shape[2])
    weights, band_names = build_srf_weights(
        protocol["srf_path"], wavelengths, protocol["bands"], interp_kind=cfg.srf_interp
    )
    train_set = HSIHSRDataset(
        image, weights, cfg.patch_size, cfg.stride, "train", cfg.test_size, augment=True
    )
    test_set = HSIHSRDataset(
        image, weights, cfg.test_size, cfg.test_size, "test", cfg.test_size, augment=False
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    info = {
        "n_bands": int(image.shape[2]),
        "n_msi_bands": int(weights.shape[0]),
        "srf_weights": weights,
        "srf_band_names": band_names,
        "shape": tuple(image.shape),
    }
    return train_loader, test_loader, info
