from pathlib import Path


def test_repository_core_is_free_of_abandoned_v4_alignment_modules():
    root = Path(__file__).resolve().parents[1]
    forbidden_files = [
        *root.glob("models/predictor_v4*.py"),
        *root.glob("train_v4*.py"),
        *root.glob("diagnose_alignment_v4*.py"),
        root / "degradations" / "misalignment.py",
    ]
    assert not any(path.exists() for path in forbidden_files)


def test_config_does_not_expose_old_alignment_arguments():
    root = Path(__file__).resolve().parents[1]
    text = (root / "config.py").read_text(encoding="utf-8").lower()
    for token in (
        "alignment_global",
        "alignment_local",
        "train_msi_translation",
        "train_msi_rotation",
        "predictor_version v4",
    ):
        assert token not in text
