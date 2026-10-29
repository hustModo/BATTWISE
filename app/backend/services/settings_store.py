"""Encrypted on-disk store for runtime LLM settings.

The encryption key lives next to the encrypted blob (mode 0600).
This is sufficient for single-host deployments where filesystem ACLs are the
boundary; not a substitute for a real KMS.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from app.backend.settings import settings


CONFIG_FILE = settings.config_dir / "llm.json.enc"
SECRET_FILE = settings.config_dir / "secret.key"


@dataclass
class LLMConfig:
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 1.0

    def masked(self) -> dict:
        d = asdict(self)
        key = d.pop("api_key", "") or ""
        d["api_key_masked"] = (
            f"{key[:4]}****{key[-4:]}" if len(key) >= 8 else ("****" if key else "")
        )
        d["has_key"] = bool(key)
        return d


def _load_or_create_secret() -> bytes:
    settings.config_dir.mkdir(parents=True, exist_ok=True)
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes()
    key = Fernet.generate_key()
    SECRET_FILE.write_bytes(key)
    try:
        os.chmod(SECRET_FILE, 0o600)
    except OSError:
        pass
    return key


def _fernet() -> Fernet:
    return Fernet(_load_or_create_secret())


def load() -> Optional[LLMConfig]:
    if not CONFIG_FILE.exists():
        return None
    try:
        token = CONFIG_FILE.read_bytes()
        plain = _fernet().decrypt(token)
        data = json.loads(plain.decode("utf-8"))
        return LLMConfig(**data)
    except (InvalidToken, ValueError, TypeError):
        return None


def save(cfg: LLMConfig) -> None:
    settings.config_dir.mkdir(parents=True, exist_ok=True)
    plain = json.dumps(asdict(cfg), ensure_ascii=False).encode("utf-8")
    token = _fernet().encrypt(plain)
    CONFIG_FILE.write_bytes(token)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def clear() -> None:
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
