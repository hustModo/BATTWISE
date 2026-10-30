"""Project-wide settings and path constants for the BATTWISE backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.backend.naming import SYSTEM_NAME


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parent

EVO_DIR = REPO_ROOT / "battwise_cddi_search"
DIANCHI_DIR = REPO_ROOT / "battery_compute"
FIRST200_DIR = REPO_ROOT / "First_200_select"
AII200_DIR = DIANCHI_DIR / "AII_200_new"
STATIS_DATA_PATH = AII200_DIR / "Statis data.xlsx"

EVO_PROMPTS_DIR = EVO_DIR / "prompts" / "battery_cddi"
EVO_GPT_PATH = EVO_DIR / "problems" / "battery_cddi" / "gpt.py"
DIANCHI_MMD_TOOLS_NEW = DIANCHI_DIR / "mmd_tools_new.py"
# Pristine baseline that is NEVER modified by the pipeline. We restore from
# this file whenever mmd_tools_new.py is detected to be polluted (e.g. a
# regression-test left behind a hanging stub). See `ensure_mmd_tools_healthy`
# below for the detection heuristic.
DIANCHI_MMD_TOOLS_BASELINE = DIANCHI_DIR / "mmd_tools_new.baseline.py"

BUILTIN_DATASETS = {
    "builtin-first-200s": {
        "name": "Built-in First 200s",
        "type": "first_200s",
        "folder": DIANCHI_DIR / "first_200s",
        "model_file": "Model_first_200s.xls",
        "char_file": "Charging_curve_characteristics.xlsx",
        "indicator_file": "Statis data.xlsx",
        "indicator_path": STATIS_DATA_PATH,
    },
    # Refinement reference dataset: full-charge 200-step CV/CC curves for each cell.
    # Used by RefinementAgent for sub-cluster analysis when the user picks a cluster.
    "builtin-all-200": {
        "name": "Built-in All 200 (Refinement)",
        "type": "all_200",
        "folder": AII200_DIR,
        "model_file": "Model_200.xls",
        "char_file": "Charging_curve_characteristics.xlsx",
        # All-200 refinement must use curve-derived features, not the
        # First-200 electrochemical indicator workbook. CDDI values computed
        # on the All-200 curves are merged into this feature table by Cell_num.
        "indicator_file": "Charging_curve_characteristics.xlsx",
        "indicator_path": STATIS_DATA_PATH,
    },
}


class Settings(BaseSettings):
    """Centralized configuration with .env fallback.

    Resolution order for LLM credentials at runtime:
        Settings panel (encrypted on-disk) > .env > battwise_cddi_search utils.API_CONFIGS
    """

    app_name: str = SYSTEM_NAME
    debug: bool = True

    workspace_dir: Path = BACKEND_ROOT / "workspace"
    datasets_dir: Path = BACKEND_ROOT / "workspace" / "datasets"
    sessions_dir: Path = BACKEND_ROOT / "workspace" / "sessions"
    config_dir: Path = BACKEND_ROOT / "workspace" / "config"
    log_dir: Path = BACKEND_ROOT / "workspace" / "logs"

    # Pinned to a durable DashScope model. `qwen-turbo*`, `*-latest`,
    # `qwen-flash`, `qwen3-coder-plus`, `qwen-max-latest` are scheduled for
    # deprecation on 2026-05-30, so don't use them as defaults. Code in
    # `services/llm_client.py::_promote_if_deprecated` rewrites stale names
    # at read-time as a safety net.
    # Pinned default is the DashScope-current Plus tier (qwen3.6-plus); see
    # `llm_client.PINNED_MODEL` for the canonical choice.
    llm_model: str = "qwen3.6-plus"
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_temperature: float = 1.0
    llm_timeout: int = 60
    llm_max_retries: int = 3
    # Short critical-path timeouts. Long 60s calls are acceptable for
    # algorithm design / EOH, but not for interactive routing and chat.
    llm_router_timeout: int = 8
    llm_chat_timeout: int = 15
    # Lightweight Designer code-generation timeout. This is longer than chat
    # because the model must emit Python code, but still bounded so a failed
    # CDDI attempt cannot leave the UI spinning for 5+ minutes.
    llm_designer_timeout: int = 45
    # DashScope Qwen3 models can spend a long time in thinking/reasoning mode
    # before emitting code. Keep it off for this interactive app; the Designer
    # prompt already asks for direct runnable Python.
    llm_enable_thinking: bool = False

    designer_max_repair_attempts: int = 5
    designer_sandbox_cells: int = 3

    allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    max_upload_mb: int = 1024
    session_ttl_hours: int = 72

    model_config = SettingsConfigDict(
        env_prefix="BATRECYCLER_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        for path in (
            self.workspace_dir,
            self.datasets_dir,
            self.sessions_dir,
            self.config_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def session_workspace(self, session_id: str) -> Path:
        path = self.sessions_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def ensure_mmd_tools_healthy() -> None:
    """Restore `mmd_tools_new.py` from the pristine baseline if it's polluted.

    Why this exists:
        The Compute regression test (``test_compute_timeout``) injects an
        intentionally-hanging ``MMD_cal_new`` to verify our SIGKILL escape
        hatch. If that test crashes mid-way (Ctrl-C, pytest --tb=...) the
        cleanup `finally:` block doesn't run, leaving a hanging stub on
        disk. The next Compute run via the UI then re-uses that stub and
        hangs forever — exactly what the user reported repeatedly.

    Detection heuristics (any one triggers a restore):
        * file is missing
        * file is suspiciously small (< 200 bytes — the real baseline is
          ~1.8 KB)
        * file contains 'Intentionally hangs' or 'while True' without any
          'np.exp' (real Gaussian MMD must call ``np.exp`` on the kernel
          matrix; the hang stub doesn't)

    Called once at backend startup (and harmless to call repeatedly).
    """
    if not DIANCHI_MMD_TOOLS_BASELINE.exists():
        return  # nothing to restore from

    if not DIANCHI_MMD_TOOLS_NEW.exists():
        DIANCHI_MMD_TOOLS_NEW.write_bytes(DIANCHI_MMD_TOOLS_BASELINE.read_bytes())
        return

    try:
        contents = DIANCHI_MMD_TOOLS_NEW.read_text(encoding="utf-8")
    except OSError:
        DIANCHI_MMD_TOOLS_NEW.write_bytes(DIANCHI_MMD_TOOLS_BASELINE.read_bytes())
        return

    is_polluted = (
        len(contents) < 200
        or "Intentionally hangs" in contents
        or ("while True" in contents and "np.exp" not in contents)
    )
    if is_polluted:
        DIANCHI_MMD_TOOLS_NEW.write_bytes(DIANCHI_MMD_TOOLS_BASELINE.read_bytes())


settings = Settings()
settings.ensure_directories()
# NOTE: `ensure_mmd_tools_healthy()` is NOT called here on purpose — it
# would also fire in worker subprocesses (they re-import settings) and
# wipe out the LLM-injected MMD function before the worker can load it,
# breaking both production runs and the timeout regression test. The
# health-check is wired into the FastAPI startup hook in `main.py` so
# it runs exactly once per backend boot.
