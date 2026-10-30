"""LLM client with three-level credential fallback.

Resolution order:
    1. Settings panel (encrypted JSON in workspace/config/llm.json.enc)
    2. Environment variables (.env loaded via pydantic-settings)
    3. battwise_cddi_search utils.utils.API_CONFIGS (built-in dev fallback)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from app.backend.services import settings_store
from app.backend.settings import REPO_ROOT, settings

logger = logging.getLogger(__name__)


@dataclass
class ResolvedConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float
    source: str  # settings | env | builtin


# --- Pinned model (hoisted before _from_builtin uses it) -----------------
# Per user directive (2026-05-27): pick one durable model and lock it in.
# Don't keep auto-probing; don't ship aliases that Alibaba is sunsetting
# on 2026-05-30 (`*-latest`, `qwen-turbo*`, `qwen3-coder-plus`,
# `qwen-flash`, `qwen-max-latest`).
#
# Choice: `qwen3.6-plus` — DashScope's CURRENT default Plus tier
# (super­seded `qwen3.5-plus` in 2026-04). Fast (≈1-3 s short message),
# tier-priced, 1M-token context, agentic upgrade over 3.5-plus at the
# same cost. NOT `qwen3.7-max` (3× output cost, slower, overkill for the
# chat / router / lightweight-designer paths we use it for).
#
# `qwen3.7-plus` does NOT exist on DashScope as of this commit — only
# `qwen3.7-max` shipped (2026-05-22). When/if 3.7-plus lands, bump this
# AND extend _DEPRECATED_MODEL_PATTERNS to auto-migrate 3.6-plus users.
PINNED_MODEL = "qwen3.6-plus"

_DEPRECATED_MODEL_PATTERNS = (
    re.compile(r"-latest$", re.I),
    re.compile(r"^qwen-turbo", re.I),
    re.compile(r"^qwen-flash$", re.I),
    re.compile(r"^qwen3-coder-plus$", re.I),
    re.compile(r"^qwen-max-latest$", re.I),
)


def is_deprecated_model(name: Optional[str]) -> bool:
    """True iff this model id is on the May-30 sunset list."""
    if not name:
        return False
    return any(p.search(name) for p in _DEPRECATED_MODEL_PATTERNS)


def _promote_if_deprecated(name: str) -> str:
    """Inline rewrite for read-time use: deprecated alias → PINNED_MODEL."""
    if is_deprecated_model(name):
        logger.warning(
            "Model %r is on the 2026-05-30 deprecation list; using %r instead.",
            name, PINNED_MODEL,
        )
        return PINNED_MODEL
    return name


def _from_builtin() -> Optional[ResolvedConfig]:
    """Last-resort fallback: import battwise_cddi_search API_CONFIGS without triggering its client init."""
    try:
        import sys

        evo_path = str(REPO_ROOT / "battwise_cddi_search")
        if evo_path not in sys.path:
            sys.path.append(evo_path)
        from utils.utils import API_CONFIGS  # type: ignore

        # Prefer the Qwen DashScope provider if present — it ships the
        # pinned (durable, non-deprecated) model. Fall back to DeepSeek
        # only when there's no Qwen key around.
        for provider_name in ("qwen", "dashscope", "aliyun", "deepseek"):
            provider = API_CONFIGS.get(provider_name)
            if not provider:
                continue
            keys = provider.get("api_keys", [])
            if not keys:
                continue
            return ResolvedConfig(
                model=PINNED_MODEL if provider_name != "deepseek" else "deepseek-v3.2",
                base_url=provider.get("base_url", ""),
                api_key=keys[0],
                temperature=1.0,
                source=f"builtin:{provider_name}",
            )
        return None
    except Exception as exc:
        logger.debug("Builtin LLM fallback unavailable: %s", exc)
        return None


def resolve_config() -> ResolvedConfig:
    panel = settings_store.load()
    if panel and panel.api_key:
        return ResolvedConfig(
            model=_promote_if_deprecated(panel.model or settings.llm_model),
            base_url=panel.base_url or settings.llm_base_url,
            api_key=panel.api_key,
            temperature=panel.temperature,
            source="settings",
        )
    if settings.llm_api_key:
        return ResolvedConfig(
            model=_promote_if_deprecated(settings.llm_model),
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            source="env",
        )
    builtin = _from_builtin()
    if builtin:
        return builtin
    raise RuntimeError(
        "No LLM credentials available. Configure via Settings panel, .env, "
        "or battwise_cddi_search utils API_CONFIGS."
    )


def has_credentials() -> bool:
    try:
        resolve_config()
        return True
    except RuntimeError:
        return False


def _build_client(cfg: ResolvedConfig, *, timeout: Optional[float] = None) -> OpenAI:
    return OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url or None,
        timeout=timeout if timeout is not None else settings.llm_timeout,
    )


def _should_send_thinking_toggle(model_name: str, cfg: ResolvedConfig) -> bool:
    """DashScope Qwen3 models support ``enable_thinking`` in ``extra_body``."""
    name = (model_name or "").lower()
    source = (cfg.source or "").lower()
    base = (cfg.base_url or "").lower()
    return (
        name.startswith("qwen3")
        and ("dashscope" in base or "qwen" in source or "dashscope" in source or "aliyun" in source)
    )


def _apply_extra_body(
    kwargs: Dict[str, Any],
    *,
    model_name: str,
    cfg: ResolvedConfig,
    enable_thinking: Optional[bool],
) -> bool:
    """Attach provider-specific options. Returns the effective thinking flag."""
    effective = settings.llm_enable_thinking if enable_thinking is None else enable_thinking
    if _should_send_thinking_toggle(model_name, cfg):
        kwargs["extra_body"] = {"enable_thinking": bool(effective)}
    return bool(effective)


def chat(
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    purpose: str = "llm",
    enable_thinking: Optional[bool] = None,
) -> str:
    cfg = resolve_config()
    client = _build_client(cfg, timeout=timeout)
    retries = max_retries if max_retries is not None else settings.llm_max_retries
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        started = time.time()
        try:
            model_name = model or cfg.model
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": min(
                    temperature if temperature is not None else cfg.temperature,
                    1.5,
                ),
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            thinking = _apply_extra_body(
                kwargs,
                model_name=model_name,
                cfg=cfg,
                enable_thinking=enable_thinking,
            )
            response = client.chat.completions.create(
                **kwargs,
            )
            elapsed_ms = int((time.time() - started) * 1000)
            logger.info(
                "LLM call ok purpose=%s model=%s source=%s elapsed_ms=%s timeout=%s thinking=%s",
                purpose,
                model_name,
                cfg.source,
                elapsed_ms,
                timeout if timeout is not None else settings.llm_timeout,
                thinking,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            elapsed_ms = int((time.time() - started) * 1000)
            logger.warning(
                "LLM call failed purpose=%s attempt=%s/%s model=%s elapsed_ms=%s timeout=%s error=%s",
                purpose,
                attempt,
                retries,
                model or cfg.model,
                elapsed_ms,
                timeout if timeout is not None else settings.llm_timeout,
                exc,
            )
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


async def chat_async(
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    purpose: str = "llm",
    enable_thinking: Optional[bool] = None,
) -> str:
    """Async completion with a real wall-clock timeout.

    Do NOT implement this by wrapping the sync ``chat()`` in
    ``asyncio.to_thread``: the caller gets control back, but the worker
    thread can keep running and later log a misleading "LLM call ok" with
    elapsed time far beyond the configured timeout. Using ``AsyncOpenAI`` lets
    ``asyncio.wait_for`` cancel the underlying request cleanly.
    """
    cfg = resolve_config()
    retries = max_retries if max_retries is not None else settings.llm_max_retries
    actual_timeout = float(timeout) if timeout is not None else float(settings.llm_timeout)
    hard_deadline = actual_timeout + 3.0 if timeout is not None else None
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        started = time.time()
        client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url or None,
            timeout=actual_timeout,
            max_retries=0,
        )
        try:
            model_name = model or cfg.model
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": min(
                    temperature if temperature is not None else cfg.temperature,
                    1.5,
                ),
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            thinking = _apply_extra_body(
                kwargs,
                model_name=model_name,
                cfg=cfg,
                enable_thinking=enable_thinking,
            )
            coro = client.chat.completions.create(**kwargs)
            response = (
                await asyncio.wait_for(coro, timeout=hard_deadline)
                if hard_deadline is not None
                else await coro
            )
            elapsed_ms = int((time.time() - started) * 1000)
            logger.info(
                "LLM async call ok purpose=%s model=%s source=%s elapsed_ms=%s timeout=%s thinking=%s",
                purpose,
                model_name,
                cfg.source,
                elapsed_ms,
                actual_timeout,
                thinking,
            )
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError as exc:
            last_err = exc
            elapsed_ms = int((time.time() - started) * 1000)
            logger.warning(
                "LLM async hard-timeout purpose=%s attempt=%s/%s model=%s elapsed_ms=%s timeout=%s deadline=%s",
                purpose,
                attempt,
                retries,
                model or cfg.model,
                elapsed_ms,
                actual_timeout,
                hard_deadline,
            )
            if attempt < retries:
                await asyncio.sleep(min(2 ** attempt, 10))
        except Exception as exc:
            last_err = exc
            elapsed_ms = int((time.time() - started) * 1000)
            logger.warning(
                "LLM async call failed purpose=%s attempt=%s/%s model=%s elapsed_ms=%s timeout=%s error=%s",
                purpose,
                attempt,
                retries,
                model or cfg.model,
                elapsed_ms,
                actual_timeout,
                exc,
            )
            if attempt < retries:
                await asyncio.sleep(min(2 ** attempt, 10))
        finally:
            try:
                await client.close()
            except Exception:
                pass
    raise RuntimeError(f"LLM async call failed after {retries} attempts: {last_err}")


async def chat_stream(
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    total_timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    purpose: str = "stream",
    enable_thinking: Optional[bool] = None,
) -> AsyncIterator[str]:
    """Stream completion chunks from the LLM as they arrive.

    Yields incremental string deltas (already extracted from
    ``chunk.choices[0].delta.content``). Empty / role-only chunks are
    silently skipped. Caller is responsible for accumulating the full
    text if needed.

    ``timeout`` is an *idle / first-token* timeout, not a total response
    budget. This distinction matters for streaming: a 30-second answer is
    fine if tokens keep arriving, but the UI should not wait 30 seconds for
    the first token or after the stream stalls. ``total_timeout`` is a
    separate hard cap to prevent truly endless streams.
    """
    cfg = resolve_config()
    idle_timeout = float(timeout) if timeout is not None else float(settings.llm_timeout)
    hard_timeout = (
        float(total_timeout)
        if total_timeout is not None
        else max(45.0, idle_timeout * 4.0)
    )
    client = AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url or None,
        timeout=idle_timeout,
    )
    model_name = model or cfg.model
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": min(
            temperature if temperature is not None else cfg.temperature,
            1.5,
        ),
        "stream": True,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    thinking = _apply_extra_body(
        kwargs,
        model_name=model_name,
        cfg=cfg,
        enable_thinking=enable_thinking,
    )
    started = time.time()
    chars = 0
    try:
        stream = await client.chat.completions.create(**kwargs)
        hard_deadline = started + hard_timeout
        while True:
            hard_remaining = hard_deadline - time.time()
            if hard_remaining <= 0:
                raise asyncio.TimeoutError(
                    f"chat_stream total budget exhausted (total_timeout={hard_timeout}s)"
                )
            try:
                chunk = await asyncio.wait_for(
                    stream.__anext__(),
                    timeout=min(idle_timeout, hard_remaining),
                )
            except StopAsyncIteration:
                break
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                chars += len(delta)
                yield delta
        elapsed_ms = int((time.time() - started) * 1000)
        logger.info(
            "LLM stream ok purpose=%s model=%s source=%s elapsed_ms=%s idle_timeout=%s total_timeout=%s chars=%s thinking=%s",
            purpose, model_name, cfg.source,
            elapsed_ms, idle_timeout, hard_timeout, chars, thinking,
        )
    except asyncio.TimeoutError as exc:
        elapsed_ms = int((time.time() - started) * 1000)
        logger.warning(
            "LLM stream timeout purpose=%s model=%s elapsed_ms=%s idle_timeout=%s total_timeout=%s chars=%s",
            purpose, model or cfg.model, elapsed_ms, idle_timeout, hard_timeout, chars,
        )
        raise TimeoutError(
            f"LLM stream stalled or exceeded budget (idle={idle_timeout:.0f}s, total={hard_timeout:.0f}s, purpose={purpose})"
        ) from exc
    except Exception as exc:
        elapsed_ms = int((time.time() - started) * 1000)
        logger.warning(
            "LLM stream failed purpose=%s model=%s elapsed_ms=%s error=%s",
            purpose, model or cfg.model, elapsed_ms, exc,
        )
        raise
    finally:
        # AsyncOpenAI uses an httpx.AsyncClient under the hood; closing it
        # releases the connection back to the pool. Non-fatal if it errors.
        try:
            await client.close()
        except Exception:
            pass


# Candidates for the (rare) `/api/settings/llm/probe` endpoint. Kept short
# and durable on purpose — every entry here must still exist after the
# 2026-05-30 deprecation wave. Order = tie-breaker priority (fast → flagship).
DEFAULT_PROBE_MODELS: List[str] = [
    "qwen3.5-flash",   # cheapest+fastest, OK for routing/chat
    "qwen3.6-plus",    # current pinned default
    "qwen3.5-plus",    # previous-gen Plus, still alive
    "qwen3-max",       # last-gen flagship, still alive
    "qwen3.7-max",     # newest flagship (2026-05-22); slow + expensive
    "deepseek-v3.2",
    "deepseek-v3",
]


def ensure_model_pinned() -> Dict[str, Any]:
    """One-shot migration: rewrite saved/env model to PINNED_MODEL if it
    matches a known-deprecated alias. Idempotent — safe to call at every
    backend startup.

    Returns a short status dict for logging.
    """
    panel = settings_store.load()
    if panel and is_deprecated_model(panel.model):
        old = panel.model
        new_cfg = settings_store.LLMConfig(
            model=PINNED_MODEL,
            base_url=panel.base_url,
            api_key=panel.api_key,
            temperature=panel.temperature,
        )
        settings_store.save(new_cfg)
        return {"migrated": True, "from": old, "to": PINNED_MODEL, "source": "settings"}
    return {"migrated": False, "current": (panel.model if panel else None)}


def _probe_one(
    cfg: ResolvedConfig,
    model: str,
    timeout: float,
) -> Dict[str, Any]:
    """Single model latency probe. Returns {model, ok, latency_ms, sample|error}."""
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None, timeout=timeout)
    started = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "ping"},
                {"role": "user", "content": "Reply with exactly the token: pong"},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        latency = int((time.time() - started) * 1000)
        sample = ""
        try:
            sample = (resp.choices[0].message.content or "").strip()[:60]
        except Exception:
            pass
        return {"model": model, "ok": True, "latency_ms": latency, "sample": sample}
    except Exception as exc:
        return {
            "model": model,
            "ok": False,
            "latency_ms": int((time.time() - started) * 1000),
            "error": str(exc)[:240],
        }


def probe_models(
    *,
    candidates: Optional[List[str]] = None,
    timeout: float = 12.0,
    max_workers: int = 6,
) -> List[Dict[str, Any]]:
    """Run a 1-shot latency probe across many models in parallel.

    All probes share the resolved base_url + api_key. Results are returned
    sorted with successful + fastest first; failures grouped at the end.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg = resolve_config()
    models = list(candidates) if candidates else list(DEFAULT_PROBE_MODELS)
    # de-dupe while preserving order
    seen: set[str] = set()
    models = [m for m in models if not (m in seen or seen.add(m))]

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_probe_one, cfg, m, timeout): m for m in models}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"model": futs[f], "ok": False, "error": str(exc)})

    results.sort(
        key=lambda r: (
            0 if r.get("ok") else 1,
            r.get("latency_ms", 10**9),
        )
    )
    return results


def ping(
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Test connectivity, optionally with override credentials."""
    if api_key:
        cfg = ResolvedConfig(
            model=model or settings.llm_model,
            base_url=base_url or settings.llm_base_url,
            api_key=api_key,
            temperature=1.0,
            source="override",
        )
    else:
        cfg = resolve_config()
    client = _build_client(cfg)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": "ping"},
                {"role": "user", "content": "Reply with the single token: pong"},
            ],
            temperature=0.0,
        )
        sample = ""
        try:
            sample = (resp.choices[0].message.content or "").strip()[:80]
        except Exception:
            pass
        return {
            "ok": True,
            "latency_ms": int((time.time() - start) * 1000),
            "source": cfg.source,
            "model": cfg.model,
            "sample": sample,
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": int((time.time() - start) * 1000),
            "error": str(exc),
            "model": cfg.model,
        }
