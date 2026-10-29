"""Session storage with JSON persistence and pipeline state tracking."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.backend.naming import LIGHTWEIGHT_MODE, normalize_mode
from app.backend.settings import settings


PIPELINE_NODES = ["router", "designer", "compute", "cluster", "refine"]


@dataclass
class Artifact:
    name: str
    kind: str  # csv | xlsx | py | png | json
    path: str  # absolute filesystem path
    size: int = 0
    created_at: float = field(default_factory=time.time)
    preview: Any = None


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    title: str = "New Session"
    mode: str = LIGHTWEIGHT_MODE
    active_dataset_id: str = "builtin-first-200s"
    active_agent: Optional[str] = None
    pipeline_state: Dict[str, str] = field(
        default_factory=lambda: {node: "idle" for node in PIPELINE_NODES}
    )
    # Single-cluster selection used by older "I'm interested in Cluster X"
    # flows. Kept for backwards compatibility — the Refine pipeline now
    # prefers `selected_clusters` (a list) so the user can refine several
    # clusters together. When a single cluster is selected, both fields
    # are kept in sync.
    selected_cluster: Optional[int] = None
    # Multi-cluster selection (e.g. user ticked clusters 0, 3, 5 in the
    # Decisions tab and asked to refine them together). Empty list means
    # "no multi-selection active" — fall back to `selected_cluster`.
    selected_clusters: List[int] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Persisted sessions created before the BATTWISE rebrand used
        # ``full_eoh``. Normalize at construction so every downstream agent
        # sees the new canonical mode without rewriting historical files.
        self.mode = normalize_mode(self.mode)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add_message(
        self,
        role: str,
        text: str,
        agent: Optional[str] = None,
        *,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        msg = {
            "id": message_id or uuid.uuid4().hex,
            "role": role,
            "agent": agent,
            "text": text,
            "t": time.time(),
        }
        self.messages.append(msg)
        self.updated_at = msg["t"]
        return msg

    def add_artifact(self, artifact: Artifact) -> Dict[str, Any]:
        for existing in self.artifacts:
            if existing["name"] == artifact.name:
                self.artifacts.remove(existing)
                break
        d = asdict(artifact)
        self.artifacts.append(d)
        self.updated_at = time.time()
        return d

    def set_node(self, node: str, state: str) -> None:
        if node in self.pipeline_state:
            self.pipeline_state[node] = state
            self.updated_at = time.time()


class SessionStore:
    """In-memory store with disk persistence under workspace/sessions/{id}/session.json."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    def _file(self, session_id: str) -> Path:
        return settings.session_workspace(session_id) / "session.json"

    async def create(self) -> Session:
        session_id = uuid.uuid4().hex[:8]
        session = Session(session_id=session_id)
        async with self._lock:
            self._sessions[session_id] = session
        await self._persist(session)
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
        path = self._file(session_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = Session(**data)
                async with self._lock:
                    self._sessions[session_id] = session
                return session
            except Exception:
                return None
        return None

    async def list(self) -> List[Session]:
        async with self._lock:
            return list(self._sessions.values())

    async def update(self, session: Session) -> None:
        async with self._lock:
            self._sessions[session.session_id] = session
        await self._persist(session)

    async def _persist(self, session: Session) -> None:
        path = self._file(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


store = SessionStore()
