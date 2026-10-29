"""Router Agent: intent classifier + pipeline gatekeeper.

Routing strategy (top-down):

1. **Cancel fast-path** — a tight regex catches "stop / cancel / 停 / 算了"
   in any language. Must be instant, doesn't go through the LLM.
2. **Greeting fast-path** — if the entire message is a short greeting
   ("hi", "你好"), route straight to ChatAgent without an LLM call.
3. **LLM smart router** (``services.llm_router``) — uses qwen3.5-flash
   plus a tool schema and the running conversation summary
   (``services.conversation_memory``) to classify intent, extract
   parameters, and optionally ask a clarifying question (slot filling).
4. **Legacy regex heuristic** — runs if the LLM is unreachable or returns
   below the confidence threshold, so the system never blocks on a
   network hiccup.

The old keyword-matching code (``_heuristic``, ``_extract_eoh_params``,
``_extract_cluster_params`` ...) is still in this file and is exercised
by the fallback path. Don't delete it.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List

from app.backend.agents.base import AgentBase
from app.backend.naming import REFLECTIVE_SEARCH_MODE
from app.backend.services import conversation_memory, llm_router

# Legacy intent vocabulary the regex heuristic emits. The new LLM router
# emits a richer "tool" vocabulary that ``llm_router.decision_to_pipeline_dict``
# maps back onto these strings, so downstream pipeline branches keep working.
INTENTS = (
    "start",
    "design_mmd",
    "compute_mmd",
    "cluster",
    "select_cluster",
    "refine_all200",
    "cancel",
    "chitchat",
)


# Conversational cancel — user can stop a running Designer / Compute /
# Cluster mid-flight. We check this BEFORE any other intent so messages
# like "stop the clustering" don't get re-routed to ClusterAgent and
# trigger a fresh run instead of stopping the running one.
_CANCEL_PATTERN = re.compile(
    r"^\s*("
    r"(?:please\s+)?(?:stop|cancel|abort|halt|kill|terminate|nevermind|never\s+mind|quit)\b"
    r"|enough\b|that'?s\s+enough"
    r"|(?:请|帮我|麻烦|麻烦你|请帮我)?\s*"
    r"(?:停(?:止|下|一下)?|结束|终止|算了|不跑了|不要了|够了|取消(?:一下)?|中止|打断)"
    r")",
    re.I,
)

# Cluster-count parsing — extract a target k from natural-language requests.
# Examples that should set target_k=6:
#   "Cluster into 6 groups."
#   "Run clustering with k=6"
#   "split into 6 clusters"
#   "Run kmeans clustering with 6 clusters."
# We match these BEFORE the generic `cluster` pattern (otherwise they would
# be eaten as a plain cluster request, which is still functionally correct
# but loses the k hint). The matched group is exposed via decision["target_k"].
_CLUSTER_K_PATTERNS = [
    re.compile(r"\bk\s*=\s*(\d+)\b", re.I),
    re.compile(r"\binto\s+(\d+)\s+(?:clusters?|groups?|categories|classes)\b", re.I),
    re.compile(r"\b(\d+)\s+(?:clusters?|groups?|categories|classes)\b", re.I),
]


# NOTE: order matters — `select_cluster` (with explicit cluster #N) must come
# BEFORE the generic `cluster` pattern, otherwise "cluster 3" would land on
# `cluster` first. Also: use `cluster\w*` so we catch `clustering` / `clusters`
# without matching unrelated words. The `\bclust` prefix would be ambiguous; we
# anchor on the full root form.
_KEYWORD_PATTERNS = [
    ("design_mmd", re.compile(r"\b(design|generate|create|new|re-?design|regenerate)\s+(an?\s+)?(?:cddi|ddia|mmd)\b", re.I)),
    ("design_mmd", re.compile(r"\b(?:cddi|ddia|mmd)\s+(function|formula|algorithm|kernel)\b", re.I)),
    ("design_mmd", re.compile(r"\b(?:run|start|launch|begin)\s+(?:BATTWISE\s+)?Reflective\s+(?:Metric\s+)?Search\b", re.I)),
    ("design_mmd", re.compile(r"(?:设计|生成|重新设计)\s*(?:CDDI|DDIA|MMD)|(?:启动|运行|开始)\s*(?:BATTWISE\s*)?反思(?:式指标)?搜索", re.I)),
    # Reflective Search / legacy CDDI imperative forms — let the user say
    # "run full EOH", "start CDDI", "再跑一次 EOH" → DesignerAgent (full_eoh).
    ("design_mmd", re.compile(r"\b(run|start|launch|begin|re-?run)\s+(full\s+)?(eoh|cddi|ddia)\b", re.I)),
    ("design_mmd", re.compile(r"\b(full\s+eoh|ddia\s+evolution)\b", re.I)),
    ("design_mmd", re.compile(r"(?:跑|启动|开始|开跑|重跑|再跑)(?:一次|一下)?\s*(?:full\s+)?(?:eoh|cddi|ddia)", re.I)),
    ("compute_mmd", re.compile(r"\b(run|compute|calculate|evaluate)\s+(?:cddi|ddia|mmd)\b", re.I)),
    ("compute_mmd", re.compile(r"(?:计算|运行)\s*(?:CDDI|DDIA|MMD)", re.I)),
    ("compute_mmd", re.compile(r"\bcompute\s+(the\s+)?(full\s+)?(batch|cells)\b", re.I)),
    ("select_cluster", re.compile(r"\b(cluster|group)\s*#?\s*(\d+)\b", re.I)),
    ("select_cluster", re.compile(r"\binterested\s+in\b", re.I)),
    ("refine_all200", re.compile(r"\b(all[\s-]?200|refine|refinement|second\s+stage|all200)\b", re.I)),
    # `clusters?` / `clustering` / `kmeans` / `group cells`; also catch
    # "run clustering" / "do clustering" / "ready to run clustering" etc.
    ("cluster", re.compile(r"\b(clusters?|clustering|kmeans|k-means|group\s+cells?)\b", re.I)),
    ("cluster", re.compile(r"\b(run|do|start|begin|ready\s+to\s+run)\b.{0,30}\bcluster", re.I)),
    # CJK forms — \b doesn't help between CJK chars, so anchor on the
    # literal verbs/nouns. "聚类" alone (run clustering); "重新聚类" (re-run);
    # "做聚类"/"开始聚类". Don't include "类别数" here — that's just a
    # parameter mention, not necessarily a request to cluster.
    ("cluster", re.compile(r"(?:聚类|做聚类|开始聚类|重新聚类|跑(?:一次)?聚类|分类|分组)")),
    ("start", re.compile(r"\b(start|begin|kick\s+off|let'?s\s+go)\b", re.I)),
]


def _extract_target_k(text: str) -> int | None:
    """Parse a desired cluster count `k` from user text (e.g. 'k=6', 'into 6 groups')."""
    for pat in _CLUSTER_K_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                k = int(m.group(1))
            except (ValueError, IndexError):
                continue
            # Reject implausible counts to avoid e.g. "Compute MMD for 71 cells"
            # being interpreted as k=71. Domain-reasonable range is 2..12.
            if 2 <= k <= 12:
                return k
    return None


_MULTI_CLUSTER_PATTERN = re.compile(
    r"\bclusters?\b[^\d]{0,8}((?:\d+[,\s/and]*)+)", re.I,
)


# ----------------------------------------------------------------------
# EOH parameter parsing.
#
# Lets the user say things like
#   "Run Reflective Search for 5 generations, pop 6"
#   "EOH 跑 3 代 每代 6 个，批 4"
#   "max_fe=5 pop_size=6"
# and have those flow into DesignerAgent's overrides. Only three knobs are
# user-controllable on purpose (per user spec); everything else stays at the
# light-weight defaults baked into DesignerAgent.FULL_EOH_LIGHT_DEFAULTS.
# ----------------------------------------------------------------------

_EOH_PARAM_PATTERNS = {
    "max_fe": [
        re.compile(r"\bmax[_\s-]?fe\s*=\s*(\d+)\b", re.I),
        re.compile(r"\b(\d+)\s*(?:gens?|generations?)\b", re.I),
        re.compile(r"(\d+)\s*代"),
    ],
    "pop_size": [
        re.compile(r"\bpop(?:_?size)?\s*=\s*(\d+)\b", re.I),
        re.compile(r"\bpopulation\s+(?:of\s+)?(\d+)\b", re.I),
        re.compile(r"\b(\d+)\s*(?:individuals?|candidates?)\s+per\s+gen", re.I),
        re.compile(r"(?:每代|每一代|个体数?)\s*(\d+)\s*个?"),
        re.compile(r"(\d+)\s*个体"),
    ],
    "batch_size": [
        re.compile(r"\bbatch(?:_?size)?\s*=\s*(\d+)\b", re.I),
        re.compile(r"\bbatch\s+(?:of\s+)?(\d+)\b", re.I),
        re.compile(r"(?:批(?:大小|次)?)\s*=?\s*(\d+)"),
    ],
}

# Plausible ranges to reject false positives like "5 cells" being read as
# pop_size=5. Generous enough to allow real demo tuning but tight enough to
# kick out misreads.
_EOH_PARAM_RANGES = {
    "max_fe": (1, 30),
    "pop_size": (2, 32),
    "batch_size": (1, 32),
}


def _extract_eoh_params(text: str) -> Dict[str, int]:
    """Pull max_fe / pop_size / batch_size from natural-language input.

    Returns only the keys actually mentioned, so callers can merge them on
    top of defaults without clobbering.
    """
    found: Dict[str, int] = {}
    for key, patterns in _EOH_PARAM_PATTERNS.items():
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            try:
                v = int(m.group(1))
            except (ValueError, IndexError):
                continue
            lo, hi = _EOH_PARAM_RANGES[key]
            if lo <= v <= hi:
                found[key] = v
                break  # first plausible hit wins
    return found


# Lightweight CDDI Designer knobs are separate from reflective-search params:
# max_attempts controls the sandbox repair loop; sandbox_cells controls how
# many cells are used for the quick validation pass.
_DESIGNER_PARAM_PATTERNS = {
    "repair_attempts": [
        re.compile(r"\brepair[_\s-]?attempts?\s*=\s*(\d+)\b", re.I),
        re.compile(r"\bmax[_\s-]?attempts?\s*=\s*(\d+)\b", re.I),
        re.compile(r"\battempts?\s*=\s*(\d+)\b", re.I),
        re.compile(r"(?:修复|修正|尝试|嘗試)(?:次数|次數)?\s*=?\s*(\d+)"),
        re.compile(r"(\d+)\s*次(?:修复|修正|尝试|嘗試)"),
    ],
    "sandbox_cells": [
        re.compile(r"\bsandbox[_\s-]?cells?\s*=\s*(\d+)\b", re.I),
        re.compile(r"\bsandbox\s*=\s*(\d+)\b", re.I),
        re.compile(r"\bvalidation[_\s-]?cells?\s*=\s*(\d+)\b", re.I),
        re.compile(r"(?:沙盒|验证|驗證|测试|測試)(?:cell|cells|电池|電池)?\s*=?\s*(\d+)"),
        re.compile(r"(\d+)\s*(?:个|個)?(?:cell|cells|电池|電池).{0,8}(?:沙盒|验证|驗證|测试|測試)"),
    ],
}

_DESIGNER_PARAM_RANGES = {
    "repair_attempts": (1, 12),
    "sandbox_cells": (1, 20),
}


def _extract_designer_params(text: str) -> Dict[str, int]:
    """Pull lightweight CDDI Designer params from natural-language input."""
    found: Dict[str, int] = {}
    for key, patterns in _DESIGNER_PARAM_PATTERNS.items():
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            try:
                v = int(m.group(1))
            except (ValueError, IndexError):
                continue
            lo, hi = _DESIGNER_PARAM_RANGES[key]
            if lo <= v <= hi:
                found[key] = v
                break
    return found


# ----------------------------------------------------------------------
# Small-talk / question detection. Without this the router would catch
# "what is an MMD function?" as `design_mmd` because the substring "mmd"
# matches one of the imperative patterns. That kicks off a full
# expensive Designer run for a casual question, which the user
# reported as "对话很死板" (the chat is stiff — anything I say triggers
# the pipeline). We now route obvious questions / greetings to the
# ChatAgent BEFORE the imperative keyword sweep.
# ----------------------------------------------------------------------

# Punctuation question marks (CJK + ASCII).
_QUESTION_MARK_RE = re.compile(r"[?？]")

# CJK interrogatives / hedges. Order-independent — any of these inside
# the message is a strong "this is a question" signal in Chinese.
_CJK_QUESTION_RE = re.compile(
    r"(什么|什麼|怎么|怎麼|为什么|為什麼|是不是|可以吗|能不能|可不可以"
    r"|吗|嗎|呢|啥|哪里|哪裡|多少|多大|多久|介绍|介紹|解释|解釋"
    r"|说一下|說一下|讲一下|講一下|科普|科普一下|你是谁|你是誰"
    r"|你能做什么|你會什么|你会什么|状态怎么样|狀態怎麼樣|做到哪一步"
    r"|现在到哪|現在到哪|在干嘛|在幹嘛)"
)

# Leading English question words ("what is...", "how do I..."). We
# anchor to the start of the line so that imperative sentences like
# "Compute MMD and then tell me what happened" still go through the
# imperative path.
_EN_QUESTION_RE = re.compile(
    r"^\s*(what|why|how|when|where|who|which|can|could|would|should|is|are|do|does|did|tell|explain|describe)\b",
    re.I,
)

# Pure greetings / chit-chat openers — even without a question mark
# these should go to chat, not to a downstream agent.
#
# IMPORTANT: `\b` doesn't act as a word boundary between two CJK
# characters (Python `re` defines \b as [a-zA-Z0-9_] vs not-that),
# so we use a non-greedy match without \b on the CJK alternatives.
# The English alternatives DO need \b to avoid matching e.g. "his"
# inside "his question".
_GREETING_RE = re.compile(
    r"^\s*("
    r"(?:hi|hello|hey|yo|good\s+(?:morning|afternoon|evening)"
    r"|thanks|thank\s+you|cheers|nice|cool|lol|haha|nb)\b"
    r"|你好|您好|嗨|哈喽|哈囉|在吗|在嗎|在不在|早上好|早安|晚上好|晚安"
    r"|谢谢|謝謝|感谢|感謝|多谢|多謝|牛|厉害|不错|可以|哈哈|嘿嘿|聊聊|聊天"
    r")",
    re.I,
)

_SIMPLE_START_WORK_RE = re.compile(
    r"\b(start|begin|go|kick\s+off)\b"
    r"|开始|启动|开工|干活|跑起来|做起来",
    re.I,
)

_EXPLICIT_WORKFLOW_RE = re.compile(
    r"\b(workflow|pipeline|full\s+flow|end[-\s]?to[-\s]?end|compute|cluster|refine)\b"
    r"|完整流程|全流程|整个流程|整套流程|一整套|全部流程|计算|聚类|分组|细分|精细化|回收决策",
    re.I,
)

# Imperative verbs that mean "do the work, not chat about it". We only
# fall through to the keyword router when one of these is present;
# otherwise a question wins.
#
# Deliberately EXCLUDED from this list:
#   * "cluster" / "clustering"   — usually a noun ("which cluster?")
#   * "refine"                    — could be a request OR a question
# Those words still drive the keyword sweep below; we just don't let
# them suppress the smalltalk detector. Pure verbs ("run", "compute",
# "design") stay in.
_IMPERATIVE_RE = re.compile(
    r"\b(run|compute|calculate|evaluate|design|generate|create|regenerate"
    r"|do|start|begin|kick\s+off)\b"
    r"|执行|开始|开跑|跑一下|计算|设计|生成|重新生成"
    r"|(?:帮我|请|麻烦).{0,12}(?:聚类|分组|计算|设计|生成|refine|细分)",
    re.I,
)


def _is_smalltalk(text: str) -> bool:
    """Return True iff this looks like a question or greeting that the
    Designer/Compute agents have no business handling.

    Heuristic, but deliberately conservative — if any imperative verb
    appears anywhere in the text we DO NOT short-circuit to chat; the
    downstream keyword matcher can still pull out the real intent.
    """
    if _IMPERATIVE_RE.search(text):
        return False
    if _QUESTION_MARK_RE.search(text):
        return True
    if _CJK_QUESTION_RE.search(text):
        return True
    if _EN_QUESTION_RE.search(text):
        return True
    if _GREETING_RE.search(text):
        return True
    return False


def _extract_selected_clusters(text: str) -> list[int]:
    """Pull every cluster id mentioned in a phrase like
    'interested in clusters 0, 3 and 5'. Returns sorted unique ids.

    Returns [] if no plural form matches.
    """
    m = _MULTI_CLUSTER_PATTERN.search(text)
    if not m:
        return []
    ids = [int(n) for n in re.findall(r"\d+", m.group(1))]
    return sorted(set(ids))


_EOH_CONTEXT_RE = re.compile(
    r"\b(eoh|evolution|full\s+eoh|generations?|max[_\s-]?fe|pop[_\s-]?size|batch[_\s-]?size)\b"
    r"|代数|代|个体|批次|进化|演化",
    re.I,
)

_DESIGNER_PARAM_CONTEXT_RE = re.compile(
    r"\b(lightweight|repair[_\s-]?attempts?|max[_\s-]?attempts?|attempts?|sandbox[_\s-]?cells?|sandbox|validation[_\s-]?cells?)\b"
    r"|轻量|轻量化|修复|修正|尝试|嘗試|沙盒|验证|驗證|测试|測試",
    re.I,
)


# ----------------------------------------------------------------------
# Cluster-stage parameter parsing.
#
# Lets the user say things like
#   "cluster with k 3 to 8, pop 100, iter 60, feedback dbi"
#   "聚类 k=3..8 种群=100 迭代=60 反馈=DBI"
#   "use variance as feedback"
# and have those flow into ClusterAgent overrides. Distinct from the EOH
# parser above — same keyword "pop/iter" but routed by intent==cluster.
# ----------------------------------------------------------------------

_CLUSTER_K_RANGE_PATTERNS = [
    re.compile(r"\bk\s*(?:=|in|from)?\s*(\d+)\s*(?:\.\.|-|—|to|至|到)\s*(\d+)\b", re.I),
    re.compile(r"\bmin[_\s-]?k\s*=\s*(\d+).*?max[_\s-]?k\s*=\s*(\d+)\b", re.I),
    re.compile(r"(?:类别数|聚类数)\s*(?:从|=)?\s*(\d+)\s*(?:到|-|~|至)\s*(\d+)"),
]
_CLUSTER_POP_PATTERNS = [
    re.compile(r"\b(?:ga[_\s-]?)?pop(?:_?size)?\s*[=\s]\s*(\d+)\b", re.I),
    re.compile(r"\bpopulation\s+(?:of\s+)?(\d+)\b", re.I),
    re.compile(r"(?:种群|群体)\s*(?:=|大小)?\s*(\d+)"),
]
_CLUSTER_ITER_PATTERNS = [
    re.compile(r"\b(?:max[_\s-]?)?iter(?:ations?|s)?\s*[=\s]\s*(\d+)\b", re.I),
    re.compile(r"\b(\d+)\s+(?:iter|iterations?|generations?)\b", re.I),
    re.compile(r"(?:迭代|轮次)(?:数|次数)?\s*=?\s*(\d+)"),
]
_CLUSTER_METRIC_PATTERNS = [
    re.compile(r"\b(?:feedback|metric|criterion)\s*(?:=|:)?\s*(dbi|variance|var|capacity|combined|both)\b", re.I),
    re.compile(r"\buse\s+(dbi|variance|var|capacity|combined|both)\s+(?:as\s+)?(?:feedback|metric|criterion)?\b", re.I),
    re.compile(r"(?:反馈|指标|评价)(?:=|为|用)\s*(dbi|variance|var|capacity|combined|both|方差|容量|综合)", re.I),
]
_CLUSTER_WEIGHT_PATTERNS = {
    "ocv": [
        re.compile(r"\bocv\b\s*(?:weight|权重)?\s*(?:=|:|为)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
        re.compile(r"(?:开路电压|OCV)\s*(?:权重)?\s*(?:=|:|为)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    ],
    "ir": [
        re.compile(r"\bir\b\s*(?:weight|权重)?\s*(?:=|:|为)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
        re.compile(r"(?:内阻|IR)\s*(?:权重)?\s*(?:=|:|为)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    ],
    "ddia": [
        re.compile(r"\b(?:cddi|ddi|ddia)\b\s*(?:weight|权重)?\s*(?:=|:|为)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
        re.compile(r"(?:结果|容量差异|CDDI|DDI|DDIA)\s*(?:权重)?\s*(?:=|:|为)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    ],
}
_CLUSTER_WEIGHT_RATIO_PATTERNS = [
    re.compile(
        r"(?:ocv|开路电压)\s*[/：:]\s*(?:ir|内阻)\s*[/：:]\s*(?:cddi|ddi|ddia|结果|容量差异)"
        r"\s*(?:=|为|:)?\s*([0-9.]+)\s*[/：:]\s*([0-9.]+)\s*[/：:]\s*([0-9.]+)",
        re.I,
    ),
    re.compile(
        r"(?:权重|weights?)\s*(?:ocv|开路电压)\s*[,，/]\s*(?:ir|内阻)\s*[,，/]\s*(?:cddi|ddi|ddia|结果|容量差异)"
        r"\s*(?:=|为|:)?\s*([0-9.]+)\s*[,，/:：]\s*([0-9.]+)\s*[,，/:：]\s*([0-9.]+)",
        re.I,
    ),
]

# Plausible ranges. k is hard-clamped to the same 2..12 we already use in
# ClusterAgent. pop_size sane for GA-KMeans = 20..500; iter = 10..500.
_CLUSTER_RANGES = {
    "k": (2, 12),
    "pop_size": (20, 500),
    "max_iter": (10, 500),
}


def _normalize_metric(raw: str) -> str:
    """Map user-supplied metric label → canonical key clustering_runner understands."""
    raw = raw.strip().lower()
    if raw in ("dbi",):
        return "dbi"
    if raw in ("variance", "var", "capacity", "方差", "容量"):
        return "variance"
    if raw in ("combined", "both", "综合"):
        return "combined"
    return "combined"


def _parse_weight(raw: str) -> float | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if not (0 < v < 10**6):
        return None
    return max(0.05, min(20.0, v))


def _extract_cluster_weights(text: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for pat in _CLUSTER_WEIGHT_RATIO_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        vals = [_parse_weight(x) for x in m.groups()[:3]]
        if all(v is not None for v in vals):
            return {"ocv": vals[0], "ir": vals[1], "ddia": vals[2]}  # type: ignore[dict-item]

    for key, patterns in _CLUSTER_WEIGHT_PATTERNS.items():
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            v = _parse_weight(m.group(1))
            if v is not None:
                weights[key] = v
                break
    return weights


def _extract_cluster_params(text: str) -> Dict[str, Any]:
    """Pull min_k/max_k, pop_size, max_iter, feedback_metric from a message.

    Returns only the keys actually mentioned. Validation is loose on purpose
    — ClusterAgent does its own clamping.
    """
    out: Dict[str, Any] = {}

    # k range first (so "k=4-8" doesn't get scooped up by a single-k extractor).
    for pat in _CLUSTER_K_RANGE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            lo, hi = int(m.group(1)), int(m.group(2))
        except (ValueError, IndexError):
            continue
        if lo > hi:
            lo, hi = hi, lo
        lo_clamp = max(_CLUSTER_RANGES["k"][0], lo)
        hi_clamp = min(_CLUSTER_RANGES["k"][1], hi)
        if lo_clamp <= hi_clamp:
            out["min_k"] = lo_clamp
            out["max_k"] = hi_clamp
        break

    for key, patterns, range_key in (
        ("pop_size", _CLUSTER_POP_PATTERNS, "pop_size"),
        ("max_iter", _CLUSTER_ITER_PATTERNS, "max_iter"),
    ):
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            try:
                v = int(m.group(1))
            except (ValueError, IndexError):
                continue
            lo, hi = _CLUSTER_RANGES[range_key]
            if lo <= v <= hi:
                out[key] = v
                break

    for pat in _CLUSTER_METRIC_PATTERNS:
        m = pat.search(text)
        if m:
            out["feedback_metric"] = _normalize_metric(m.group(1))
            break

    weights = _extract_cluster_weights(text)
    if weights:
        out["feature_weights"] = weights

    return out


_CLUSTER_PARAM_CONTEXT_RE = re.compile(
    r"\b(pop|iter|generation|feedback|metric|dbi|variance|min[_\s-]?k|max[_\s-]?k|weights?|ocv|ir|cddi|ddi|ddia)\b"
    r"|种群|迭代|反馈|指标|方差|类别数|权重|开路电压|内阻|容量差异",
    re.I,
)


def _heuristic(text: str) -> Dict[str, Any]:
    # FIRST: stop / cancel — must short-circuit before EVERYTHING ELSE,
    # otherwise "stop the clustering" routes to cluster intent and
    # triggers a fresh run instead of halting the running one.
    if _CANCEL_PATTERN.search(text):
        return {"intent": "cancel", "matched": "cancel-heuristic"}
    # Then: catch obvious questions / greetings before the keyword
    # sweep below. Without this, "what is mmd?" would match the
    # `design_mmd` pattern via the literal "mmd" substring and trigger
    # a Designer run for a casual question (user complaint:
    # "对话很死板, 我闲聊他也不支持").
    if _is_smalltalk(text):
        return {"intent": "chitchat", "matched": "smalltalk-heuristic"}
    for intent, pat in _KEYWORD_PATTERNS:
        m = pat.search(text)
        if m:
            extra: Dict[str, Any] = {}
            if intent == "select_cluster":
                # First try multi-cluster ("clusters 0, 3 and 5"); if none,
                # fall back to the single-cluster id captured by the
                # regex group.
                multi_ids = _extract_selected_clusters(text)
                if len(multi_ids) >= 2:
                    extra["selected_clusters"] = multi_ids
                    extra["selected_cluster"] = multi_ids[0]
                elif m.groups():
                    try:
                        cluster_idx = int(next(g for g in m.groups() if g and g.isdigit()))
                        extra["selected_cluster"] = cluster_idx
                        extra["selected_clusters"] = [cluster_idx]
                    except (StopIteration, ValueError):
                        pass
            if intent == "cluster":
                k = _extract_target_k(text)
                if k is not None:
                    extra["target_k"] = k
                # Power-user knobs: only parse when the message actually
                # talks about params, so a plain "run clustering" doesn't
                # pick up incidental numbers.
                if _CLUSTER_PARAM_CONTEXT_RE.search(text):
                    cparams = _extract_cluster_params(text)
                    if cparams:
                        extra["cluster_params"] = cparams
            # EOH params apply to design_mmd / start path. We gate on either
            # the intent being design-ish OR an explicit EOH context word
            # (so plain "design MMD" doesn't auto-pick up stray digits).
            if intent in ("design_mmd", "start"):
                if _DESIGNER_PARAM_CONTEXT_RE.search(text):
                    dparams = _extract_designer_params(text)
                    if dparams:
                        extra["designer_params"] = dparams
                if _EOH_CONTEXT_RE.search(text):
                    params = _extract_eoh_params(text)
                    if params:
                        extra["eoh_params"] = params
            return {"intent": intent, "matched": m.group(0), **extra}
    return {"intent": None}


def _next_agent(intent: str) -> str:
    return {
        "start": "designer",
        "design_mmd": "designer",
        "compute_mmd": "compute",
        "cluster": "cluster",
        "select_cluster": "refine",
        "refine_all200": "refine",
        "cancel": "cancel",  # handled in pipeline, no real agent dispatched
        "chitchat": "chat",
    }.get(intent, "chat")


class RouterAgent(AgentBase):
    agent_id = "router"
    node_id = "router"

    # If the entire message is at most this many characters AND matches the
    # greeting regex, we skip the LLM call entirely. Anything longer might
    # contain a question or imperative embedded in a casual hello.
    _GREETING_MAX_LEN = 30

    async def run(self, user_text: str) -> Dict[str, Any]:  # type: ignore[override]
        await self.started()
        await self.thinking(f"Routing: {user_text!r}")

        # Fast-path 1: cancel — never let an LLM hiccup delay a stop request.
        if _CANCEL_PATTERN.search(user_text):
            decision = {
                "intent": "cancel",
                "next_agent": "cancel",
                "source": "regex-cancel",
            }
        # Fast-path 2: trivial greeting → chat, no LLM round-trip.
        elif self._is_pure_greeting(user_text):
            decision = {
                "intent": "chitchat",
                "next_agent": "chat",
                "source": "greeting-fast-path",
            }
        # Fast-path 3: "开始干活 / start based on First-200s" should open
        # the Designer parameter gate, not let the LLM inflate it into a
        # multi-step plan. Explicit "完整流程 / compute+cluster" wording still
        # goes to the LLM router and may become plan_workflow.
        elif self._is_simple_start_work(user_text):
            decision = {
                "intent": "start",
                "next_agent": "designer",
                "source": "start-work-fast-path",
            }
            dparams = _extract_designer_params(user_text)
            if dparams:
                decision["designer_params"] = dparams
            eparams = _extract_eoh_params(user_text)
            if eparams:
                decision["eoh_params"] = eparams
            mode = self._extract_mode_hint(user_text)
            if mode:
                decision["proposed_mode"] = mode
        # Fast-path 3: ordinary questions / small-talk go straight to
        # ChatAgent. This is still LLM-powered (ChatAgent may call the chat
        # model), but avoids the slow serial chain:
        #   Router LLM classify -> Chat LLM answer
        # for messages like "你能帮我做什么？". Imperative/action messages
        # still fall through to the router LLM.
        elif self._is_obvious_chat(user_text):
            decision = {
                "intent": "chitchat",
                "next_agent": "chat",
                "source": "chat-question-fast-path",
            }
        else:
            decision = await self._smart_route(user_text)

        await self.emit(
            "router_decision",
            intent=decision["intent"],
            next_agent=decision["next_agent"],
            reason=decision.get("reasoning") or decision.get("reason"),
            source=decision.get("source"),
            tool=decision.get("tool"),
            confidence=decision.get("confidence"),
            needs_clarification=decision.get("needs_clarification"),
            clarification=decision.get("clarification"),
            designer_params=decision.get("designer_params"),
            cluster_params=decision.get("cluster_params"),
            eoh_params=decision.get("eoh_params"),
            target_k=decision.get("target_k"),
        )
        await self.finished(result=decision)
        return decision

    # ------------------------------------------------------------------ helpers

    @classmethod
    def _is_pure_greeting(cls, text: str) -> bool:
        """True iff the *entire* message is a short greeting with no imperatives.

        Stricter than ``_is_smalltalk`` — that one catches questions too and
        we want those to flow through the LLM router so the LLM can decide
        between "explain X" and "do X" naturally.
        """
        stripped = (text or "").strip()
        if not stripped or len(stripped) > cls._GREETING_MAX_LEN:
            return False
        if _IMPERATIVE_RE.search(stripped):
            return False
        return bool(_GREETING_RE.match(stripped))

    @classmethod
    def _is_obvious_chat(cls, text: str) -> bool:
        """Route non-action chat/questions directly to ChatAgent.

        This does NOT hard-code the answer. It only skips the routing LLM
        when the message is already clearly a question or small-talk and has
        no imperative verb. ChatAgent remains free to call the LLM and answer
        intelligently with session context.
        """
        stripped = (text or "").strip()
        if not stripped or len(stripped) > 240:
            return False
        return _is_smalltalk(stripped)

    @classmethod
    def _is_simple_start_work(cls, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if _EXPLICIT_WORKFLOW_RE.search(stripped):
            return False
        return bool(_SIMPLE_START_WORK_RE.search(stripped))

    @staticmethod
    def _extract_mode_hint(text: str) -> str | None:
        if re.search(r"\bfull\s*eoh\b|full\s*evolution|reflective\s+(?:metric\s+)?search|反思(?:式指标)?搜索|完整\s*eoh|全量\s*eoh", text, re.I):
            return REFLECTIVE_SEARCH_MODE
        if re.search(r"\blightweight\b|轻量|轻量化", text, re.I):
            return "lightweight"
        return None

    async def _smart_route(self, user_text: str) -> Dict[str, Any]:
        snapshot = self._build_snapshot()
        summary = conversation_memory.get_summary(self.session)
        history = self._recent_history()

        try:
            llm_decision = await llm_router.classify_async(
                user_text=user_text,
                session_snapshot=snapshot,
                summary=summary,
                recent_history=history,
            )
        except Exception as exc:
            await self.log(f"smart router exception: {exc}", level="warning")
            llm_decision = None

        if llm_decision is not None and llm_decision.confidence >= llm_router.MIN_CONFIDENCE:
            await self.thinking(
                f"LLM router → {llm_decision.tool} "
                f"(conf={llm_decision.confidence:.2f}"
                f"{', needs clarification' if llm_decision.needs_clarification else ''})"
            )
            return llm_router.decision_to_pipeline_dict(llm_decision)

        # Low-confidence or unreachable → regex fallback.
        await self.log(
            "LLM router unavailable or low confidence — using regex fallback",
            level="info",
        )
        return self._heuristic_decision(user_text)

    def _heuristic_decision(self, user_text: str) -> Dict[str, Any]:
        heuristic = _heuristic(user_text)
        if not heuristic.get("intent"):
            return {
                "intent": "chitchat",
                "next_agent": "chat",
                "source": "fallback-chat",
            }
        intent = heuristic["intent"]
        out: Dict[str, Any] = {
            "intent": intent,
            "next_agent": _next_agent(intent),
            "source": "heuristic-fallback",
            "matched": heuristic.get("matched"),
        }
        for key in (
            "selected_cluster",
            "selected_clusters",
            "target_k",
            "eoh_params",
            "cluster_params",
        ):
            if key in heuristic:
                out[key] = heuristic[key]
        return out

    def _build_snapshot(self) -> Dict[str, Any]:
        s = self.session
        return {
            "active_agent": s.active_agent,
            "pipeline_state": dict(s.pipeline_state),
            "mode": s.mode,
            "active_dataset_id": s.active_dataset_id,
            "selected_clusters": list(s.selected_clusters or []),
            "selected_cluster": s.selected_cluster,
            "sticky_eoh_params": s.context.get("eoh_user_params"),
            "sticky_cluster_params": s.context.get("cluster_user_params"),
            "pending_design_confirmation": bool(
                s.context.get("pending_design_confirmation")
            ),
            "recent_artifacts": [a.get("name") for a in s.artifacts[-5:]],
            "message_count": len(s.messages),
        }

    def _recent_history(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for msg in self.session.messages[-8:]:
            role = msg.get("role")
            text = msg.get("text") or ""
            if role in ("user", "assistant") and text:
                out.append({"role": role, "content": text})
        return out
