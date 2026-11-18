import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_s2diff_benchmark_entry", ROOT / "benchmark_paviau.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_benchmark_loads_its_own_fixed_predictor(monkeypatch):
    # A same-named older module is already present, e.g. from another checkout.
    unrelated = ModuleType("test_fixed_deformation")
    unrelated.build_predictor = lambda args, info, process, device: None
    monkeypatch.setitem(sys.modules, "test_fixed_deformation", unrelated)
    monkeypatch.syspath_prepend(str(ROOT))

    predictor, pipeline = benchmark.load_benchmark_predictor(ROOT)
    assert predictor is not unrelated.build_predictor
    assert Path(predictor.__code__.co_filename) == ROOT / "test_fixed_deformation.py"
    assert Path(pipeline.__file__).parent == ROOT


def test_benchmark_identifies_stale_local_predictor(tmp_path):
    entry = tmp_path / "test_fixed_deformation.py"
    entry.write_text("def build_predictor(args, info, process, device): pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires a newer .*test_fixed_deformation.py"):
        benchmark.load_benchmark_predictor(tmp_path)
