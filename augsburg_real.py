"""Augsburg-Real data preparation and protocol helpers.

The real branch keeps the existing synthetic Augsburg benchmark untouched.
It uses

* reference: MDAS EeteS_EnMAP_10m (sensor-realistic 10 m EnMAP-like HSI),
* LR-HSI:    MDAS EeteS_EnMAP_30m (sensor-realistic 30 m EnMAP-like HSI),
* MSI:       real Sentinel-2 L2A, native-10 m B2/B3/B4/B8 only.

Only metadata-based georeferencing is performed. No content-based image
registration is used during preparation, so residual real cross-sensor
misregistration remains for Real-C CDRDI.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from srf_utils import build_srf_weights


S2_NATIVE10_BANDS = ("B2", "B3", "B4", "B8")
S2_CANONICAL_12_BANDS = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
)

AUGSBURG_REAL_SPLITS = {
    "train": {
        "gt": "sr_deep_model_data/EeteS_EnMAP_10m_deep_train.tif",
        "lr": "sr_deep_model_data/EeteS_EnMAP_30m_deep_train.tif",
    },
    "validation": {
        "gt": "sr_deep_model_data/EeteS_EnMAP_10m_deep_valid.tif",
        "lr": "sr_deep_model_data/EeteS_EnMAP_30m_deep_valid.tif",
    },
    "test": {
        "gt": "sub_area_1/EeteS_EnMAP_10m_sub_area1.tif",
        "lr": "sub_area_1/EeteS_EnMAP_30m_sub_area1.tif",
    },
}


def find_augsburg_root(data_root: str) -> str:
    candidates = [
        os.path.join(data_root, "Augsburg", "Augsburg_data_4_publication"),
        os.path.join(data_root, "Augsburg_data_4_publication"),
        os.path.join(data_root, "Augsburg"),
        data_root,
    ]
    for root in candidates:
        if os.path.exists(os.path.join(root, "band_242_meta_info.hdr")):
            return os.path.abspath(root)
    raise FileNotFoundError(
        "Augsburg root must contain band_242_meta_info.hdr; checked: "
        + ", ".join(candidates)
    )


def resolve_real_s2_path(root: str, explicit: str = "") -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            os.path.join(root, "entire_city", "Sentinel-2.tif"),
            os.path.join(root, "entire_city", "Sentinel_2.tif"),
            os.path.join(root, "Sentinel-2.tif"),
            os.path.join(root, "Sentinel_2.tif"),
        ]
    )
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Cannot find real Sentinel-2 GeoTIFF. Pass --real_s2_path explicitly. "
        "Checked: " + ", ".join(candidates)
    )


def read_enmap_wavelengths(hdr_path: str) -> np.ndarray:
    import re

    if not os.path.exists(hdr_path):
        raise FileNotFoundError(hdr_path)
    text = open(hdr_path, "r", encoding="utf-8", errors="ignore").read()
    match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, flags=re.I | re.S)
    if not match:
        raise ValueError(f"No wavelength={{...}} field found in {hdr_path}")
    values = [
        float(x)
        for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", match.group(1))
    ]
    arr = np.asarray(values, dtype=np.float32)
    if arr.size != 242:
        raise ValueError(f"Expected 242 EnMAP wavelengths, got {arr.size}")
    if float(arr.max()) < 10.0:
        arr *= 1000.0
    return arr


def _require_rasterio():
    try:
        import rasterio
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise ImportError(
            "Augsburg-Real geospatial preparation requires rasterio; "
            "install repository requirements first"
        ) from exc
    return rasterio, Resampling, reproject


def _normalise_band_token(value: object) -> str:
    text = str(value or "").upper().replace(" ", "").replace("_", "").replace("-", "")
    for token in ("B8A", "B11", "B12", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"):
        if token in text:
            return token
    return ""


def resolve_s2_band_indexes(
    dataset,
    requested: Sequence[str] = S2_NATIVE10_BANDS,
) -> Tuple[List[int], List[str]]:
    """Resolve 1-based raster indexes for B2/B3/B4/B8."""
    labels: Dict[str, int] = {}
    descriptions = list(dataset.descriptions or ())
    for index in range(1, dataset.count + 1):
        candidates: List[object] = []
        if index - 1 < len(descriptions):
            candidates.append(descriptions[index - 1])
        candidates.extend(dataset.tags(index).values())
        for value in candidates:
            token = _normalise_band_token(value)
            if token:
                labels.setdefault(token, index)

    requested = tuple(str(x).upper() for x in requested)
    if all(name in labels for name in requested):
        return [labels[name] for name in requested], list(requested)

    if dataset.count == 12:
        fallback = {name: i + 1 for i, name in enumerate(S2_CANONICAL_12_BANDS)}
        return [fallback[name] for name in requested], list(requested)

    raise ValueError(
        "Could not resolve Sentinel-2 band labels from GeoTIFF metadata and "
        f"fallback requires a 12-band MDAS stack; count={dataset.count}"
    )


def infer_s2_platform(dataset) -> str:
    values: List[object] = list(dataset.tags().values())
    for index in range(1, min(dataset.count, 4) + 1):
        values.extend(dataset.tags(index).values())
    text = " ".join(str(v).upper() for v in values)
    if "SENTINEL-2B" in text or "S2B" in text:
        return "S2B"
    if "SENTINEL-2A" in text or "S2A" in text:
        return "S2A"
    return "UNKNOWN"


def _target_profile(reference_path: str):
    rasterio, _, _ = _require_rasterio()
    with rasterio.open(reference_path) as ref:
        if ref.crs is None or ref.transform is None:
            raise ValueError(f"Reference GeoTIFF lacks CRS/transform: {reference_path}")
        return ref.crs, ref.transform, ref.width, ref.height


def _reproject_multiband(
    source_path: str,
    *,
    target_crs,
    target_transform,
    target_width: int,
    target_height: int,
    indexes: Optional[Sequence[int]] = None,
    scale: float = 10000.0,
    resampling: str = "bilinear",
) -> np.ndarray:
    rasterio, Resampling, reproject = _require_rasterio()
    mode = getattr(Resampling, str(resampling))
    with rasterio.open(source_path) as src:
        selected = list(indexes) if indexes is not None else list(range(1, src.count + 1))
        out = np.zeros((len(selected), target_height, target_width), dtype=np.float32)
        for out_index, src_index in enumerate(selected):
            reproject(
                source=rasterio.band(src, int(src_index)),
                destination=out[out_index],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=mode,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
    if scale <= 0.0:
        raise ValueError("reflectance scale must be >0")
    out = out / float(scale)
    return np.moveaxis(out, 0, -1).astype(np.float32)


def _reproject_mask(
    source_path: str,
    *,
    target_crs,
    target_transform,
    target_width: int,
    target_height: int,
    invalid_classes: Sequence[int],
) -> np.ndarray:
    rasterio, Resampling, reproject = _require_rasterio()
    with rasterio.open(source_path) as src:
        source = src.read(1)
        target = np.full((target_height, target_width), 0, dtype=np.int16)
        reproject(
            source=source,
            destination=target,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=0,
        )
    return ~np.isin(target, np.asarray(list(invalid_classes), dtype=target.dtype))


def build_s2_srf_weights(
    root: str,
    *,
    srf_path: str,
    band_columns: Sequence[str],
    interp_kind: str = "pchip",
) -> Tuple[np.ndarray, np.ndarray]:
    wavelengths = read_enmap_wavelengths(os.path.join(root, "band_242_meta_info.hdr"))
    weights, _ = build_srf_weights(
        srf_path,
        wavelengths,
        band_columns,
        interp_kind=interp_kind,
    )
    return weights.astype(np.float32), wavelengths.astype(np.float32)


def prepare_augsburg_real_cache(
    *,
    data_root: str,
    cache_root: str,
    real_s2_path: str = "",
    srf_path: str,
    srf_band_columns: Sequence[str],
    scl_path: str = "",
    s2_reflectance_scale: float = 10000.0,
    enmap_reflectance_scale: float = 10000.0,
    invalid_scl_classes: Sequence[int] = (0, 1, 3, 8, 9, 10, 11),
    overwrite: bool = False,
) -> Dict[str, object]:
    """Prepare metadata-aligned real-S2/EnMAP arrays for all official splits."""
    rasterio, _, _ = _require_rasterio()
    from affine import Affine

    root = find_augsburg_root(data_root)
    real_s2_path = resolve_real_s2_path(root, real_s2_path)
    os.makedirs(cache_root, exist_ok=True)

    with rasterio.open(real_s2_path) as s2_src:
        s2_indexes, s2_names = resolve_s2_band_indexes(s2_src)
        platform = infer_s2_platform(s2_src)
        s2_source_count = int(s2_src.count)

    srf_weights, wavelengths = build_s2_srf_weights(
        root,
        srf_path=srf_path,
        band_columns=srf_band_columns,
    )
    np.save(os.path.join(cache_root, "srf_weights.npy"), srf_weights)
    np.save(os.path.join(cache_root, "hsi_wavelengths.npy"), wavelengths)

    summary: Dict[str, object] = {
        "root": root,
        "real_s2_path": real_s2_path,
        "s2_platform": platform,
        "s2_source_count": s2_source_count,
        "s2_band_indexes_1based": s2_indexes,
        "s2_band_names": s2_names,
        "srf_path": os.path.abspath(srf_path),
        "srf_band_columns": list(srf_band_columns),
        "splits": {},
    }

    for split, relative in AUGSBURG_REAL_SPLITS.items():
        split_dir = os.path.join(cache_root, split)
        os.makedirs(split_dir, exist_ok=True)
        metadata_path = os.path.join(split_dir, "meta.json")
        required = [
            os.path.join(split_dir, "gt.npy"),
            os.path.join(split_dir, "lr_hsi.npy"),
            os.path.join(split_dir, "hr_msi.npy"),
            os.path.join(split_dir, "valid_mask.npy"),
            metadata_path,
        ]
        if not overwrite and all(os.path.exists(p) for p in required):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            summary["splits"][split] = meta
            continue

        gt_path = os.path.join(root, relative["gt"])
        lr_path = os.path.join(root, relative["lr"])
        if not os.path.exists(gt_path) or not os.path.exists(lr_path):
            raise FileNotFoundError(f"Missing Augsburg split files: {gt_path} / {lr_path}")

        hr_crs, hr_transform, hr_width, hr_height = _target_profile(gt_path)
        hr_width3 = (hr_width // 3) * 3
        hr_height3 = (hr_height // 3) * 3
        lr_width = hr_width3 // 3
        lr_height = hr_height3 // 3
        lr_transform = hr_transform * Affine.scale(3, 3)

        gt = _reproject_multiband(
            gt_path,
            target_crs=hr_crs,
            target_transform=hr_transform,
            target_width=hr_width3,
            target_height=hr_height3,
            scale=enmap_reflectance_scale,
            resampling="bilinear",
        )
        lr_hsi = _reproject_multiband(
            lr_path,
            target_crs=hr_crs,
            target_transform=lr_transform,
            target_width=lr_width,
            target_height=lr_height,
            scale=enmap_reflectance_scale,
            resampling="bilinear",
        )
        hr_msi = _reproject_multiband(
            real_s2_path,
            target_crs=hr_crs,
            target_transform=hr_transform,
            target_width=hr_width3,
            target_height=hr_height3,
            indexes=s2_indexes,
            scale=s2_reflectance_scale,
            resampling="bilinear",
        )

        lr_valid = (
            np.isfinite(lr_hsi).all(axis=2)
            & (lr_hsi >= -0.05).all(axis=2)
            & (lr_hsi <= 1.5).all(axis=2)
        )
        lr_valid_hr = np.repeat(
            np.repeat(lr_valid, 3, axis=0), 3, axis=1
        )[:hr_height3, :hr_width3]
        valid = (
            np.isfinite(gt).all(axis=2)
            & np.isfinite(hr_msi).all(axis=2)
            & (gt >= -0.05).all(axis=2)
            & (gt <= 1.5).all(axis=2)
            & (hr_msi >= -0.05).all(axis=2)
            & (hr_msi <= 1.5).all(axis=2)
            & lr_valid_hr
        )
        if scl_path:
            valid &= _reproject_mask(
                scl_path,
                target_crs=hr_crs,
                target_transform=hr_transform,
                target_width=hr_width3,
                target_height=hr_height3,
                invalid_classes=invalid_scl_classes,
            )

        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        lr_hsi = np.nan_to_num(lr_hsi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        hr_msi = np.nan_to_num(hr_msi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        valid = valid.astype(np.uint8)

        np.save(os.path.join(split_dir, "gt.npy"), gt)
        np.save(os.path.join(split_dir, "lr_hsi.npy"), lr_hsi)
        np.save(os.path.join(split_dir, "hr_msi.npy"), hr_msi)
        np.save(os.path.join(split_dir, "valid_mask.npy"), valid)

        meta = {
            "split": split,
            "gt_source": gt_path,
            "lr_source": lr_path,
            "s2_source": real_s2_path,
            "hr_shape": list(gt.shape),
            "lr_shape": list(lr_hsi.shape),
            "msi_shape": list(hr_msi.shape),
            "scale_ratio": 3,
            "valid_fraction": float(valid.mean()),
            "s2_platform": platform,
            "s2_band_indexes_1based": s2_indexes,
            "s2_band_names": s2_names,
            "metadata_alignment_only": True,
            "content_registration": False,
            "scl_mask_used": bool(scl_path),
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        summary["splits"][split] = meta

    with open(os.path.join(cache_root, "protocol.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _grid_coords(h: int, w: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    return [
        (top, left)
        for top in range(0, h - patch + 1, stride)
        for left in range(0, w - patch + 1, stride)
    ]


def _partition_tiles(h: int, w: int, max_patch: int) -> List[Tuple[int, int, int, int]]:
    """Cover an x3-compatible region exactly once with non-overlapping tiles."""
    if max_patch < 3 or max_patch % 3:
        raise ValueError("Augsburg-Real eval_patch_size must be divisible by 3")
    if h % 3 or w % 3:
        raise ValueError("Augsburg-Real evaluation region must be divisible by 3")
    rows = []
    top = 0
    while top < h:
        ph = min(max_patch, h - top)
        if ph % 3:
            raise RuntimeError("evaluation tile height lost x3 compatibility")
        left = 0
        while left < w:
            pw = min(max_patch, w - left)
            if pw % 3:
                raise RuntimeError("evaluation tile width lost x3 compatibility")
            rows.append((top, left, ph, pw))
            left += pw
        top += ph
    return rows


class AugsburgRealDataset(Dataset):
    """Patch dataset over prepared Augsburg-Real arrays."""

    def __init__(
        self,
        cache_root: str,
        split: str,
        *,
        train_patch_size: int = 96,
        train_stride: int = 48,
        eval_patch_size: int = 192,
        min_valid_fraction: float = 0.80,
        augment: bool = True,
    ):
        split = "validation" if split == "val" else split
        if split not in ("train", "validation", "test"):
            raise ValueError(split)
        self.split = split
        split_dir = os.path.join(cache_root, split)
        self.gt = np.load(os.path.join(split_dir, "gt.npy"), mmap_mode="r")
        self.lr_hsi = np.load(os.path.join(split_dir, "lr_hsi.npy"), mmap_mode="r")
        self.hr_msi = np.load(os.path.join(split_dir, "hr_msi.npy"), mmap_mode="r")
        self.valid = np.load(os.path.join(split_dir, "valid_mask.npy"), mmap_mode="r")
        if self.gt.shape[:2] != self.hr_msi.shape[:2] or self.valid.shape != self.gt.shape[:2]:
            raise ValueError("Augsburg-Real HR arrays are not co-gridded")
        if self.gt.shape[0] != self.lr_hsi.shape[0] * 3 or self.gt.shape[1] != self.lr_hsi.shape[1] * 3:
            raise ValueError("Augsburg-Real HR/LR arrays are not exact x3 pairs")

        patch = int(train_patch_size if split == "train" else eval_patch_size)
        stride = int(train_stride if split == "train" else eval_patch_size)
        if patch % 3 or stride % 3:
            raise ValueError("Augsburg-Real patch_size and stride must be divisible by 3")
        self.patch_size = patch
        self.augment = bool(augment and split == "train")
        self.samples: List[Tuple[int, int, int, int]] = []
        if split == "train":
            candidates = [
                (top, left, patch, patch)
                for top, left in _grid_coords(
                    self.gt.shape[0], self.gt.shape[1], patch, stride
                )
            ]
        else:
            candidates = _partition_tiles(
                self.gt.shape[0], self.gt.shape[1], patch
            )
        for top, left, ph, pw in candidates:
            mask = self.valid[top:top + ph, left:left + pw]
            if float(mask.mean()) >= float(min_valid_fraction):
                self.samples.append((top, left, ph, pw))
        if not self.samples:
            raise RuntimeError(
                f"No Augsburg-Real {split} patches meet min_valid_fraction={min_valid_fraction}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        top, left, ph, pw = self.samples[index]
        lt, ll = top // 3, left // 3
        lph, lpw = ph // 3, pw // 3
        gt = np.asarray(self.gt[top:top+ph, left:left+pw]).copy()
        hr_msi = np.asarray(self.hr_msi[top:top+ph, left:left+pw]).copy()
        lr_hsi = np.asarray(self.lr_hsi[lt:lt+lph, ll:ll+lpw]).copy()
        mask = np.asarray(self.valid[top:top+ph, left:left+pw]).copy()

        if self.augment:
            if np.random.rand() < 0.5:
                gt = np.flip(gt, 0)
                hr_msi = np.flip(hr_msi, 0)
                lr_hsi = np.flip(lr_hsi, 0)
                mask = np.flip(mask, 0)
            if np.random.rand() < 0.5:
                gt = np.flip(gt, 1)
                hr_msi = np.flip(hr_msi, 1)
                lr_hsi = np.flip(lr_hsi, 1)
                mask = np.flip(mask, 1)
            k = int(np.random.randint(0, 4))
            if k:
                gt = np.rot90(gt, k, axes=(0, 1))
                hr_msi = np.rot90(hr_msi, k, axes=(0, 1))
                lr_hsi = np.rot90(lr_hsi, k, axes=(0, 1))
                mask = np.rot90(mask, k, axes=(0, 1))

        gt = np.ascontiguousarray(gt)
        hr_msi = np.ascontiguousarray(hr_msi)
        lr_hsi = np.ascontiguousarray(lr_hsi)
        mask = np.ascontiguousarray(mask.astype(np.float32))
        return {
            "gt": torch.from_numpy(gt).permute(2, 0, 1).float(),
            "lr_hsi": torch.from_numpy(lr_hsi).permute(2, 0, 1).float(),
            "hr_msi": torch.from_numpy(hr_msi).permute(2, 0, 1).float(),
            "valid_mask": torch.from_numpy(mask).unsqueeze(0).float(),
        }


def build_augsburg_real_loaders(
    cache_root: str,
    *,
    train_patch_size: int = 96,
    train_stride: int = 48,
    eval_patch_size: int = 192,
    min_valid_fraction: float = 0.80,
    batch_size: int = 2,
    num_workers: int = 0,
):
    train = AugsburgRealDataset(
        cache_root,
        "train",
        train_patch_size=train_patch_size,
        train_stride=train_stride,
        eval_patch_size=eval_patch_size,
        min_valid_fraction=min_valid_fraction,
        augment=True,
    )
    val = AugsburgRealDataset(
        cache_root,
        "validation",
        train_patch_size=train_patch_size,
        train_stride=train_stride,
        eval_patch_size=eval_patch_size,
        min_valid_fraction=min_valid_fraction,
        augment=False,
    )
    test = AugsburgRealDataset(
        cache_root,
        "test",
        train_patch_size=train_patch_size,
        train_stride=train_stride,
        eval_patch_size=eval_patch_size,
        min_valid_fraction=min_valid_fraction,
        augment=False,
    )

    def make(dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    info = {
        "n_bands": int(train.gt.shape[2]),
        "n_msi_bands": int(train.hr_msi.shape[2]),
        "scale_ratio": 3,
        "stages": [1, 2, 3],
        "srf_weights": np.load(os.path.join(cache_root, "srf_weights.npy")),
        "hsi_wavelengths": np.load(os.path.join(cache_root, "hsi_wavelengths.npy")),
        "train_samples": len(train),
        "validation_samples": len(val),
        "test_samples": len(test),
    }
    return (
        make(train, batch_size, True),
        make(val, 1, False),
        make(test, 1, False),
        info,
    )
