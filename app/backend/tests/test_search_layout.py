"""Exercise renamed search paths using temporary, synthetic inputs only."""
from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

from hydra import compose, initialize_config_dir
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SEARCH = ROOT / "battwise_cddi_search"


def test_application_paths_and_problem_import():
    from app.backend import settings

    assert settings.EVO_DIR == SEARCH
    assert settings.DIANCHI_DIR == ROOT / "battery_compute"
    assert settings.EVO_GPT_PATH.is_file()
    assert (settings.EVO_PROMPTS_DIR / "seed_func.txt").is_file()
    assert settings.DIANCHI_MMD_TOOLS_NEW.is_file()
    module = importlib.import_module("battwise_cddi_search.problems.battery_cddi")
    assert Path(module.__file__).parent == settings.EVO_GPT_PATH.parent
    candidate = importlib.import_module("battwise_cddi_search.problems.battery_cddi.gpt")
    function = next(getattr(candidate, name) for name in dir(candidate) if name.startswith("calculate_mmd_v"))
    assert np.isfinite(function(np.ones((4, 2)), np.zeros((4, 2))))


@pytest.mark.parametrize("config", sorted((SEARCH / "cfg").glob("config_*.yaml")), ids=lambda p: p.stem)
def test_search_configs_resolve(config):
    with initialize_config_dir(version_base=None, config_dir=str(SEARCH / "cfg")):
        cfg = compose(config_name=config.stem)
    assert cfg.problem.problem_name == "battery_cddi"
    assert (SEARCH / "problems" / cfg.problem.problem_name / "eval.py").is_file()
    assert (SEARCH / "prompts" / cfg.problem.problem_name / "seed_func.txt").is_file()


@pytest.mark.parametrize("case", ["2", "4"], ids=["first_200s", "all_200"])
def test_evaluator_with_synthetic_curves(tmp_path, monkeypatch, case):
    evaluator = SEARCH / "problems/battery_cddi/eval.py"
    monkeypatch.setattr(sys, "argv", [str(evaluator), str(tmp_path), "train", case, "0"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    config = runpy.run_path(str(evaluator))
    data = Path(config["DATA_FOLDER"])
    assert data.parent == tmp_path / "problems/battery_data"
    data.mkdir(parents=True)
    generated = tmp_path / "baselines/eoh/gpt_code/gpt_code_0.py"
    generated.parent.mkdir(parents=True)
    # Seed only the temporary harness; never edit the frozen program itself.
    generated.write_text("import numpy as np\nnp.random.seed(0)\n" +
                         (ROOT / "programs/cddi/calculate_mmd_v2.py").read_text(encoding="utf-8"),
                         encoding="utf-8")

    # Excel bytes are detected by pandas even for the evaluator's .xls names.
    def workbook(frame, name):
        buffer = io.BytesIO()
        frame.to_excel(buffer, index=False, engine="openpyxl")
        (data / name).write_bytes(buffer.getvalue())

    t = np.linspace(0.0, 1.0, 200)
    reference = np.column_stack([1.5 - 0.5 * t, 3.2 + 0.8 * t])
    workbook(pd.DataFrame(reference, columns=["Current", "Voltage"]), config["MODEL_FILENAME"])
    values = []
    namespace = runpy.run_path(str(generated))
    for cell in range(1, 5):
        curve = reference + np.column_stack([cell * 0.03 * (1 + t), cell * 0.02 * t**2])
        workbook(pd.DataFrame(curve, columns=["Current", "Voltage"]), f"Cell{cell}.xlsx")
        values.append(namespace["calculate_mmd_v2"](curve, reference))
    assert np.isfinite(values).all() and np.ptp(values) > 0
    # A synthetic target equal to the computed scalars gives a known correlation.
    workbook(pd.DataFrame({"Cell_num": range(1, 5), "synthetic_descriptor": values}), config["CHAR_FILENAME"])
    result = subprocess.run(
        [sys.executable, "-B", str(evaluator), str(tmp_path), "train", case, "0"],
        cwd=SEARCH, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "computation complete: 4 samples" in result.stdout
    objective = re.search(r"Error code: (\S+)", result.stdout)
    assert objective, result.stdout
    assert float(objective[1]) == pytest.approx(-1.0), result.stdout


@pytest.mark.parametrize("filename", ["best_code_battery_cddi.txt", "best_code_battery_mmd.txt"])
def test_seed_reader_accepts_new_and_historical_names(tmp_path, monkeypatch, filename):
    from app.backend.services import eoh_seeds

    monkeypatch.setattr(eoh_seeds, "EVO_DIR", tmp_path)
    folder = tmp_path / "outputs/output/synthetic"
    folder.mkdir(parents=True)
    code = "def calculate_mmd_v2(xs, xt):\n    return 0.25\n"
    (folder / filename).write_text(code, encoding="utf-8")
    seeds = eoh_seeds._gather_candidates(["synthetic"])
    assert any(seed.code == code.strip() and seed.source == "synthetic/" + filename for seed in seeds)
