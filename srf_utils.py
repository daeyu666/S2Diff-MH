"""Sensor spectral-response utilities for S2Diff-MH."""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

IKONOS_4_BANDS = [
    "IKONOS Blue",
    "IKONOS Green",
    "IKONOS Red",
    "IKONOS NIR",
]

WV2_ALL_8_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
    "WV2 RedEdge",
    "WV2 NIR1",
    "WV2 NIR2",
]


def load_hsi_wavelengths(wavelength_path: str, n_bands: int) -> np.ndarray:
    if not os.path.exists(wavelength_path):
        raise FileNotFoundError(f"Cannot find wavelength file: {wavelength_path}")
    ext = os.path.splitext(wavelength_path)[1].lower()
    if ext == ".npy":
        values = np.load(wavelength_path)
    elif ext in (".txt", ".dat"):
        values = np.loadtxt(wavelength_path)
    elif ext == ".csv":
        df = pd.read_csv(wavelength_path)
        selected = None
        for key in ("wavelength", "wave", "wl", "lambda", "center"):
            for col in df.columns:
                if key in col.lower():
                    selected = col
                    break
            if selected is not None:
                break
        values = df[selected or df.columns[0]].values
    else:
        raise ValueError(f"Unsupported wavelength file type: {ext}")
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != int(n_bands):
        raise ValueError(
            f"Wavelength number mismatch: got {values.size}, HSI has {n_bands} bands"
        )
    if np.nanmax(values) < 10:
        values = values * 1000.0
    return values.astype(np.float32)


def estimate_band_widths(wavelengths: np.ndarray) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    if wavelengths.size == 1:
        return np.ones_like(wavelengths)
    edges = np.zeros(wavelengths.size + 1, dtype=np.float32)
    edges[1:-1] = 0.5 * (wavelengths[:-1] + wavelengths[1:])
    edges[0] = wavelengths[0] - 0.5 * (wavelengths[1] - wavelengths[0])
    edges[-1] = wavelengths[-1] + 0.5 * (wavelengths[-1] - wavelengths[-2])
    return np.maximum(edges[1:] - edges[:-1], 1e-6).astype(np.float32)


def interp_srf_to_hsi_wavelengths(
    srf_wavelengths: np.ndarray,
    response_values: np.ndarray,
    hsi_wavelengths: np.ndarray,
    interp_kind: str = "pchip",
) -> np.ndarray:
    srf_wavelengths = np.asarray(srf_wavelengths, dtype=np.float32)
    response_values = np.asarray(response_values, dtype=np.float32)
    hsi_wavelengths = np.asarray(hsi_wavelengths, dtype=np.float32)
    order = np.argsort(srf_wavelengths)
    srf_wavelengths = srf_wavelengths[order]
    response_values = response_values[order]
    sampled = None
    if interp_kind == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator
            sampled = PchipInterpolator(
                srf_wavelengths,
                response_values,
                extrapolate=False,
            )(hsi_wavelengths)
        except Exception:
            sampled = None
    if sampled is None:
        if interp_kind not in ("pchip", "linear"):
            raise ValueError(f"Unsupported interp_kind: {interp_kind}")
        sampled = np.interp(
            hsi_wavelengths,
            srf_wavelengths,
            response_values,
            left=0.0,
            right=0.0,
        )
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(sampled, 0.0).astype(np.float32)


def build_srf_weights(
    srf_path: str,
    hsi_wavelengths: np.ndarray,
    selected_bands: Iterable[str],
    interp_kind: str = "pchip",
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, List[str]]:
    if not os.path.exists(srf_path):
        raise FileNotFoundError(f"Cannot find SRF file: {srf_path}")
    df = pd.read_csv(srf_path)
    if "WL(nm)" not in df.columns:
        raise ValueError("SRF file must contain column: WL(nm)")
    srf_wavelengths = df["WL(nm)"].values.astype(np.float32)
    hsi_wavelengths = np.asarray(hsi_wavelengths, dtype=np.float32).reshape(-1)
    widths = estimate_band_widths(hsi_wavelengths)
    weights = []
    names = []
    for band in selected_bands:
        if band not in df.columns:
            raise ValueError(f"SRF file does not contain band column: {band}")
        response = interp_srf_to_hsi_wavelengths(
            srf_wavelengths,
            df[band].values.astype(np.float32),
            hsi_wavelengths,
            interp_kind=interp_kind,
        )
        raw = response * widths
        total = float(raw.sum())
        if total < eps:
            raise ValueError(
                f"SRF band {band} has no overlap with HSI wavelength range "
                f"{hsi_wavelengths.min():.2f}-{hsi_wavelengths.max():.2f} nm"
            )
        weights.append((raw / (total + eps) if normalize else raw).astype(np.float32))
        names.append(str(band))
    return np.stack(weights, axis=0).astype(np.float32), names


def hsi_to_msi_numpy(hsi: np.ndarray, srf_weights: np.ndarray, clip: bool = True) -> np.ndarray:
    if hsi.ndim != 3 or srf_weights.ndim != 2:
        raise ValueError("hsi must be HxWxC and srf_weights must be MxC")
    if hsi.shape[2] != srf_weights.shape[1]:
        raise ValueError("HSI/SRF band mismatch")
    msi = np.tensordot(hsi, srf_weights.T, axes=([2], [0])).astype(np.float32)
    return np.clip(msi, 0.0, 1.0) if clip else msi


def sensor_protocol(dataset: str):
    if dataset == "PaviaU":
        return {
            "bands": IKONOS_4_BANDS,
            "srf_path": "./data/srf/ikonos_relative_spectral_response.csv",
            "wavelength_path": "./data/wavelengths/PaviaU_nominal_430_860.txt",
        }
    if dataset in ("Chikusei", "Houston13"):
        return {
            "bands": WV2_ALL_8_BANDS,
            "srf_path": "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv",
            "wavelength_path": f"./data/wavelengths/{dataset}.txt",
        }
    raise ValueError(f"No fixed sensor protocol for dataset={dataset!r}")
