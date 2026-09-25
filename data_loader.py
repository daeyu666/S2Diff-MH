"""Unified HSI/MSI dataset loader and benchmark split protocols.

LR-HSI is intentionally NOT generated here in S2Diff-MH. Innovation 1 owns the
observation operator, so every sample returned here contains only clean HR-HSI
GT and the registered HR-MSI generated from the frozen sensor SRF.

Final benchmark protocols:
- PaviaU: center 128x128 test, top-left 128x128 validation, remaining area train.
- Houston2013: center 128x128 test, top-left 128x128 validation, remaining area train.
- Chikusei: center-crop 2304x2048; rows 0:128 test (16 non-overlapping
  128x128 patches), rows 128:256 validation (16 patches), rows 256:2304 train.
- CAVE: deterministic 16 train / 4 validation / 12 test scenes (20/12
  train-pool/test convention with 20% of the train pool held out for validation).
- Botswana: center 128x128 test, top-left 128x128 validation, remaining area train.
- Augsburg synthetic x4: official MDAS geographic train/validation/test files;
  EnMAP 10m is treated as HR-HSI and LR-HSI is generated later by Innovation 1.

The train split is spatially/scene disjoint from validation and test.  Test
must not be used for checkpoint selection.
"""

from __future__ import annotations

import functools
import glob
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from srf_utils import build_srf_weights, hsi_to_msi_numpy, load_hsi_wavelengths, sensor_protocol

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
    "PaviaU": {"kind": "mat", "file": "PaviaU.mat", "keys": ["paviaU", "PaviaU", "img", "data"]},
    "Houston13": {"kind": "mat", "file": "Houston13.mat", "keys": ["Houston13", "Houston_HSI", "data", "img"]},
    "Chikusei": {"kind": "mat", "file": "Chikusei.mat", "keys": ["chikusei", "Chikusei", "img", "data"]},
    "Botswana": {"kind": "mat", "file": "Botswana.mat", "keys": ["Botswana", "botswana", "img", "data"]},
    "CAVE": {"kind": "cave", "dir": "CAVE"},
    "Augsburg": {"kind": "augsburg", "dir": "Augsburg"},
}


CAVE_TRAIN_SCENES = [
    "balloons", "beads", "cd", "chart_and_stuffed_toy", "clay", "cloth",
    "egyptian_statue", "face", "fake_and_real_beers", "fake_and_real_food",
    "fake_and_real_lemon_slices", "fake_and_real_lemons",
    "fake_and_real_peppers", "fake_and_real_strawberries",
    "fake_and_real_sushi", "fake_and_real_tomatoes",
]
CAVE_VALIDATION_SCENES = ["feathers", "flowers", "glass_tiles", "hairs"]
CAVE_TEST_SCENES = [
    "jelly_beans", "oil_painting", "paints", "photo_and_face", "pompoms",
    "real_and_fake_apples", "real_and_fake_peppers", "sponges", "stuffed_toys",
    "superballs", "thread_spools", "watercolors",
]


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


def _fix_hsi_shape(arr: np.ndarray, expected_bands: int | None = None) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 3:
        raise ValueError(f"HSI must be 3-D, got {arr.shape}")
    if expected_bands is not None:
        axes = [i for i, size in enumerate(arr.shape) if size == int(expected_bands)]
        if len(axes) == 1 and axes[0] != 2:
            arr = np.moveaxis(arr, axes[0], 2)
            return arr.astype(np.float32)
    if arr.shape[0] <= 256 and arr.shape[1] > 256 and arr.shape[2] > 256:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[1] <= 256 and arr.shape[0] > 256 and arr.shape[2] > 256:
        arr = np.transpose(arr, (0, 2, 1))
    return arr.astype(np.float32)


def read_hsi_mat(path: str, candidate_keys: Sequence[str]) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    errors = []
    if hdf5storage is not None:
        try:
            mapping = hdf5storage.loadmat(path)
            for key in list(candidate_keys) + list(mapping.keys()):
                if key in mapping:
                    cube = _extract_cube(mapping[key])
                    if cube is not None:
                        return _fix_hsi_shape(cube)
        except Exception as exc:
            errors.append(f"hdf5storage: {exc}")
    if scio is not None:
        try:
            mapping = scio.loadmat(path)
            for key in list(candidate_keys) + list(mapping.keys()):
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
    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def crop_to_scale(img: np.ndarray, scale_ratio: int) -> np.ndarray:
    h, w, _ = img.shape
    return img[: h // scale_ratio * scale_ratio, : w // scale_ratio * scale_ratio]


def _center_rect(h: int, w: int, size: int) -> Tuple[int, int, int, int]:
    if h < size or w < size:
        raise ValueError(f"Image {(h,w)} smaller than requested {size}x{size}")
    top = (h - size) // 2
    left = (w - size) // 2
    return top, left, top + size, left + size


def _intersects(a, b) -> bool:
    t1, l1, b1, r1 = a
    t2, l2, b2, r2 = b
    return not (r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1)


def _grid_coords(h: int, w: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    return [
        (top, left)
        for top in range(0, h - patch + 1, stride)
        for left in range(0, w - patch + 1, stride)
    ]


def _nonoverlap_coords(h: int, w: int, patch: int) -> List[Tuple[int, int]]:
    return _grid_coords(h, w, patch, patch)


def _single_scene_coords(
    dataset: str,
    h: int,
    w: int,
    patch_size: int,
    stride: int,
    split: str,
    test_size: int,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    test_rect = _center_rect(h, w, test_size)
    val_rect = (0, 0, test_size, test_size)
    if _intersects(val_rect, test_rect):
        val_rect = (0, w - test_size, test_size, w)
    if split == "test":
        return [(test_rect[0], test_rect[1])], val_rect, test_rect
    if split in ("validation", "val"):
        return [(val_rect[0], val_rect[1])], val_rect, test_rect
    coords = []
    for top, left in _grid_coords(h, w, patch_size, stride):
        rect = (top, left, top + patch_size, left + patch_size)
        if not _intersects(rect, val_rect) and not _intersects(rect, test_rect):
            coords.append((top, left))
    if not coords:
        raise RuntimeError(f"No train patches for {dataset}")
    return coords, val_rect, test_rect


def _center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w, _ = img.shape
    if h < target_h or w < target_w:
        raise ValueError(f"Cannot center-crop {(h,w)} to {(target_h,target_w)}")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return img[top:top + target_h, left:left + target_w, :]


def _prepare_chikusei(img: np.ndarray) -> np.ndarray:
    # Literature protocol used by recent Chikusei HSI-SR work.
    return _center_crop(img, 2304, 2048)


def _chikusei_coords(
    patch_size: int,
    stride: int,
    split: str,
    test_size: int,
) -> List[Tuple[int, int]]:
    if test_size != 128:
        raise ValueError("Fixed Chikusei benchmark uses test_size=128")
    if split == "test":
        return [(0, left) for left in range(0, 2048, 128)]
    if split in ("validation", "val"):
        return [(128, left) for left in range(0, 2048, 128)]
    coords = []
    for top in range(256, 2304 - patch_size + 1, stride):
        for left in range(0, 2048 - patch_size + 1, stride):
            coords.append((top, left))
    return coords


class HSIHSRDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        srf_weights: np.ndarray,
        patch_size: int,
        coords: Sequence[Tuple[int, int]],
        split: str,
        augment: bool = True,
        dataset_name: str = "",
    ):
        self.image = image
        self.srf_weights = np.asarray(srf_weights, dtype=np.float32)
        self.patch_size = int(patch_size)
        self.coords = list(coords)
        self.split = split
        self.dataset_name = dataset_name
        self.augment = bool(augment and split == "train")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        top, left = self.coords[index]
        gt = self.image[top:top+self.patch_size, left:left+self.patch_size].copy()
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
            "gt": torch.from_numpy(gt).permute(2,0,1).contiguous().float(),
            "hr_msi": torch.from_numpy(hr_msi).permute(2,0,1).contiguous().float(),
        }


def _canonical_scene_name(path: str) -> str:
    name = os.path.basename(os.path.normpath(path)).lower()
    name = re.sub(r"_ms$", "", name)
    return name


def _find_cave_scene_dirs(root: str) -> Dict[str, str]:
    candidates = set()
    for pattern in ("**/*_ms_01.png", "**/*_ms_01.PNG"):
        for file_path in glob.glob(os.path.join(root, pattern), recursive=True):
            candidates.add(os.path.dirname(file_path))
    mapping = {_canonical_scene_name(path): path for path in sorted(candidates)}
    expected = set(CAVE_TRAIN_SCENES + CAVE_VALIDATION_SCENES + CAVE_TEST_SCENES)
    missing = sorted(expected - set(mapping))
    if missing:
        raise FileNotFoundError(
            "CAVE scenes not found under %s: %s" % (root, ", ".join(missing))
        )
    return mapping


@functools.lru_cache(maxsize=4)
def _load_cave_scene(scene_dir: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("CAVE PNG loading requires Pillow") from exc
    files = []
    for idx in range(1, 32):
        matches = glob.glob(os.path.join(scene_dir, f"*_ms_{idx:02d}.png"))
        matches += glob.glob(os.path.join(scene_dir, f"*_ms_{idx:02d}.PNG"))
        if not matches:
            raise FileNotFoundError(f"Missing CAVE band {idx:02d} in {scene_dir}")
        files.append(sorted(matches)[0])
    bands = [np.asarray(Image.open(path), dtype=np.float32) for path in files]
    cube = np.stack(bands, axis=2)
    # Official CAVE reflectance PNGs are 16-bit.
    if cube.max() > 1.0:
        cube = cube / 65535.0
    return np.clip(cube, 0.0, 1.0).astype(np.float32)


class CAVEDataset(Dataset):
    def __init__(
        self,
        scene_dirs: Dict[str, str],
        scene_names: Sequence[str],
        srf_weights: np.ndarray,
        split: str,
        patch_size: int,
        stride: int,
        augment: bool,
    ):
        self.scene_dirs = scene_dirs
        self.scene_names = list(scene_names)
        self.srf_weights = np.asarray(srf_weights, dtype=np.float32)
        self.split = split
        self.patch_size = int(patch_size)
        self.augment = bool(augment and split == "train")
        self.samples: List[Tuple[str, int, int, int]] = []
        for name in self.scene_names:
            if split == "train":
                for top, left in _grid_coords(512, 512, self.patch_size, int(stride)):
                    self.samples.append((name, top, left, self.patch_size))
            else:
                self.samples.append((name, 0, 0, 512))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, top, left, size = self.samples[index]
        image = _load_cave_scene(self.scene_dirs[name])
        gt = image[top:top+size, left:left+size].copy()
        if self.augment:
            if random.random() < 0.5:
                gt = np.flip(gt, axis=0)
            if random.random() < 0.5:
                gt = np.flip(gt, axis=1)
            if random.random() < 0.5:
                gt = np.rot90(gt, k=random.randint(1,3), axes=(0,1))
            gt = np.ascontiguousarray(gt)
        hr_msi = hsi_to_msi_numpy(gt, self.srf_weights)
        return {
            "gt": torch.from_numpy(gt).permute(2,0,1).contiguous().float(),
            "hr_msi": torch.from_numpy(hr_msi).permute(2,0,1).contiguous().float(),
        }


def _read_envi_wavelengths(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, flags=re.I | re.S)
    if not match:
        raise ValueError(f"No wavelength={{...}} field found in {path}")
    values = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", match.group(1))]
    arr = np.asarray(values, dtype=np.float32)
    if arr.size != 242:
        raise ValueError(f"Expected 242 EnMAP wavelengths, got {arr.size}")
    if arr.max() < 10:
        arr *= 1000.0
    return arr


def _read_tiff_cube(path: str, expected_bands: int = 242) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import tifffile
        arr = tifffile.imread(path)
    except ImportError as exc:
        raise ImportError("Augsburg GeoTIFF loading requires tifffile") from exc
    arr = _fix_hsi_shape(arr, expected_bands=expected_bands)
    # MDAS EnMAP/HySpex reflectance products use scale 1e4 in the reference code.
    arr = arr.astype(np.float32)
    if np.nanmax(arr) > 2.0:
        arr = arr / 10000.0
    return np.clip(np.nan_to_num(arr), 0.0, 1.0).astype(np.float32)


def _find_augsburg_root(data_root: str) -> str:
    candidates = [
        os.path.join(data_root, "Augsburg", "Augsburg_data_4_publication"),
        os.path.join(data_root, "Augsburg"),
        os.path.join(data_root, "Augsburg_data_4_publication"),
        data_root,
    ]
    for root in candidates:
        if os.path.exists(os.path.join(root, "band_242_meta_info.hdr")):
            return root
    raise FileNotFoundError(
        "Augsburg root must contain band_242_meta_info.hdr; checked: " + ", ".join(candidates)
    )


def _make_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def _build_standard_single_scene(cfg, image: np.ndarray, weights: np.ndarray):
    image = normalize_hsi(image)
    image = crop_to_scale(image, cfg.scale_ratio)
    if cfg.dataset == "Chikusei":
        image = _prepare_chikusei(image)
        train_coords = _chikusei_coords(cfg.patch_size, cfg.stride, "train", cfg.test_size)
        val_coords = _chikusei_coords(cfg.test_size, cfg.test_size, "validation", cfg.test_size)
        test_coords = _chikusei_coords(cfg.test_size, cfg.test_size, "test", cfg.test_size)
        val_rect = (128, 0, 256, 2048)
        test_rect = (0, 0, 128, 2048)
    else:
        train_coords, val_rect, test_rect = _single_scene_coords(
            cfg.dataset, image.shape[0], image.shape[1], cfg.patch_size, cfg.stride, "train", cfg.test_size
        )
        val_coords, _, _ = _single_scene_coords(
            cfg.dataset, image.shape[0], image.shape[1], cfg.test_size, cfg.test_size, "validation", cfg.test_size
        )
        test_coords, _, _ = _single_scene_coords(
            cfg.dataset, image.shape[0], image.shape[1], cfg.test_size, cfg.test_size, "test", cfg.test_size
        )
    train_set = HSIHSRDataset(image, weights, cfg.patch_size, train_coords, "train", True, cfg.dataset)
    val_set = HSIHSRDataset(image, weights, cfg.test_size, val_coords, "validation", False, cfg.dataset)
    test_set = HSIHSRDataset(image, weights, cfg.test_size, test_coords, "test", False, cfg.dataset)
    return train_set, val_set, test_set, image, val_rect, test_rect


def _build_cave(cfg):
    root = os.path.join(cfg.data_root, "CAVE")
    if not os.path.isdir(root):
        root = cfg.data_root
    scene_dirs = _find_cave_scene_dirs(root)
    protocol = sensor_protocol("CAVE")
    wavelengths = load_hsi_wavelengths(protocol["wavelength_path"], 31)
    weights, band_names = build_srf_weights(protocol["srf_path"], wavelengths, protocol["bands"], interp_kind=cfg.srf_interp)
    train_set = CAVEDataset(scene_dirs, CAVE_TRAIN_SCENES, weights, "train", cfg.patch_size, cfg.stride, True)
    val_set = CAVEDataset(scene_dirs, CAVE_VALIDATION_SCENES, weights, "validation", 512, 512, False)
    test_set = CAVEDataset(scene_dirs, CAVE_TEST_SCENES, weights, "test", 512, 512, False)
    info = {
        "n_bands": 31,
        "n_msi_bands": 3,
        "srf_weights": weights,
        "srf_band_names": band_names,
        "shape": (512,512,31),
        "protocol": "CAVE deterministic 16 train / 4 validation / 12 test scenes",
        "train_samples": len(train_set), "validation_samples": len(val_set), "test_samples": len(test_set),
    }
    return train_set, val_set, test_set, info


def _build_augsburg(cfg):
    root = _find_augsburg_root(cfg.data_root)
    hdr = os.path.join(root, "band_242_meta_info.hdr")
    wavelengths = _read_envi_wavelengths(hdr)
    protocol = sensor_protocol("Augsburg")
    weights, band_names = build_srf_weights(protocol["srf_path"], wavelengths, protocol["bands"], interp_kind=cfg.srf_interp)
    files = {
        "train": os.path.join(root, "sr_deep_model_data", "EeteS_EnMAP_10m_deep_train.tif"),
        "validation": os.path.join(root, "sr_deep_model_data", "EeteS_EnMAP_10m_deep_valid.tif"),
        "test": os.path.join(root, "sub_area_1", "EeteS_EnMAP_10m_sub_area1.tif"),
    }
    images = {k: _read_tiff_cube(v, 242) for k,v in files.items()}
    # Training uses overlapping 64x64 patches; validation/test cover the official
    # geographic regions by non-overlapping 128x128 tiles.
    train_coords = _grid_coords(images["train"].shape[0], images["train"].shape[1], cfg.patch_size, cfg.stride)
    val_coords = _nonoverlap_coords(images["validation"].shape[0], images["validation"].shape[1], cfg.test_size)
    test_coords = _nonoverlap_coords(images["test"].shape[0], images["test"].shape[1], cfg.test_size)
    train_set = HSIHSRDataset(images["train"], weights, cfg.patch_size, train_coords, "train", True, "Augsburg")
    val_set = HSIHSRDataset(images["validation"], weights, cfg.test_size, val_coords, "validation", False, "Augsburg")
    test_set = HSIHSRDataset(images["test"], weights, cfg.test_size, test_coords, "test", False, "Augsburg")
    info = {
        "n_bands": 242,
        "n_msi_bands": 4,
        "srf_weights": weights,
        "srf_band_names": band_names,
        "hsi_wavelengths": wavelengths,
        "shape": tuple(images["train"].shape),
        "protocol": "MDAS official geographic train/validation/sub_area_1 test; synthetic x4",
        "train_samples": len(train_set), "validation_samples": len(val_set), "test_samples": len(test_set),
    }
    return train_set, val_set, test_set, info


def build_datasets(cfg):
    if cfg.dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset={cfg.dataset!r}")
    if cfg.dataset == "CAVE":
        return _build_cave(cfg)
    if cfg.dataset == "Augsburg":
        return _build_augsburg(cfg)

    spec = DATASET_SPECS[cfg.dataset]
    image = read_hsi_mat(os.path.join(cfg.data_root, spec["file"]), spec["keys"])
    protocol = sensor_protocol(cfg.dataset)
    wavelengths = load_hsi_wavelengths(protocol["wavelength_path"], image.shape[2])
    weights, band_names = build_srf_weights(
        protocol["srf_path"], wavelengths, protocol["bands"], interp_kind=cfg.srf_interp
    )
    train_set, val_set, test_set, image, val_rect, test_rect = _build_standard_single_scene(cfg, image, weights)
    info = {
        "n_bands": int(image.shape[2]),
        "n_msi_bands": int(weights.shape[0]),
        "srf_weights": weights,
        "srf_band_names": band_names,
        "hsi_wavelengths": wavelengths,
        "shape": tuple(image.shape),
        "validation_rect": val_rect,
        "test_rect": test_rect,
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "test_samples": len(test_set),
        "protocol": (
            "Chikusei centered 2304x2048; top 128-row test strip, next 128-row validation strip"
            if cfg.dataset == "Chikusei"
            else f"{cfg.dataset} center128 test + top-left128 validation"
        ),
    }
    return train_set, val_set, test_set, info


def build_train_val_test_loaders(cfg):
    train_set, val_set, test_set, info = build_datasets(cfg)
    return (
        _make_loader(train_set, cfg.batch_size, True, cfg.num_workers, False),
        _make_loader(val_set, 1, False, cfg.num_workers, False),
        _make_loader(test_set, 1, False, cfg.num_workers, False),
        info,
    )


def build_loaders(cfg):
    """Backward-compatible train/test API.

    The train set still excludes/respects the fixed validation split.  New
    training code should call build_train_val_test_loaders() so checkpoint
    selection never touches the final test set.
    """
    train_loader, _, test_loader, info = build_train_val_test_loaders(cfg)
    return train_loader, test_loader, info
