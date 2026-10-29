"""Refinement Agent.

Workflow (per `docs/PLAN.md §四.5`):
    1. User picks a parent cluster (via "I'm interested in Cluster X" or by
       clicking a cluster card).
    2. Agent reads the previous ``Cluster_label.xlsx`` to find which Cells
       belong to that parent cluster.
    3. If no All-200 dataset is ready, emit a ``user_prompt_required`` event
       asking the user to either upload an All-200 acquisition or confirm
       use of the built-in ``builtin-all-200`` reference (development mode).
    4. Once ready, run a sub-pipeline:
         - inject the latest LLM-designed CDDI function into ``mmd_tools_new``
         - compute CDDI on the All-200 curves for those Cells only
         - re-cluster the subset on OCV, IR, and All-200 CDDI only
         - emit sub-cluster overlay charts and a sub-Cluster_label artefact
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.backend.agents.base import AgentBase
from app.backend.naming import is_reflective_search_mode
from app.backend.agents.designer_agent import DesignerAgent
from app.backend.orchestrator.session_store import Artifact, store
from app.backend.services import (
    cluster_merge,
    cluster_input_builder,
    clustering_runner,
    data_resolver,
    mmd_designer,
    mmd_runner,
)
from app.backend.services.clustering_runner import Individual
from app.backend.settings import settings

ITER_PROGRESS_STRIDE = 5
PROGRESS_STRIDE = 1  # All-200 is much smaller than first-200, so emit every cell


def _parse_clusters(user_text: str, fallback: List[int]) -> List[int]:
    """Extract one OR more cluster ids from user_text.

    Recognised patterns (case-insensitive):
        - "refine cluster 3"                        -> [3]
        - "I'm interested in cluster 1"             -> [1]
        - "refine clusters 0, 3 and 5"              -> [0, 3, 5]
        - "clusters 0 / 3 / 5"                      -> [0, 3, 5]
        - "refine_clusters=[0,3,5]"                 -> [0, 3, 5]

    If no explicit ids are found in the text, returns ``fallback`` (which
    is typically ``session.selected_clusters`` or [session.selected_cluster]).
    """
    if not user_text:
        return list(fallback)

    # Plural pattern: "clusters X, Y[, Z][ and W]" — capture every number
    # in the next ~80 chars after the word "clusters".
    plural = re.search(
        r"\bclusters?\b[^\d]{0,8}((?:\d+[,\s/and]*)+)", user_text, flags=re.I
    )
    if plural:
        ids = [int(m) for m in re.findall(r"\d+", plural.group(1))]
        if ids:
            return sorted(set(ids))

    # Single pattern: "cluster #N" / "cluster N"
    single = re.search(r"\bcluster\s*#?\s*(\d+)\b", user_text, flags=re.I)
    if single:
        try:
            return [int(single.group(1))]
        except ValueError:
            pass

    return list(fallback)


def _read_cells_for_clusters(
    cluster_xlsx: Path, cluster_ids: List[int]
) -> Dict[int, List[int]]:
    """Map each requested cluster id -> list of Cell_num values.

    Returns a dict keyed by the original (unsorted) cluster ids so the
    caller can keep the user's selection order if needed. Cluster ids that
    don't exist in the file map to an empty list.
    """
    df = pd.read_excel(cluster_xlsx, engine="calamine")
    if "Cluster_label" not in df.columns or "Cell_num" not in df.columns:
        raise KeyError("Cluster_label.xlsx must contain Cluster_label and Cell_num columns")
    out: Dict[int, List[int]] = {}
    for cid in cluster_ids:
        sub = df[df["Cluster_label"].astype(int) == int(cid)]
        out[int(cid)] = [int(c) for c in sub["Cell_num"].dropna().astype(int).tolist()]
    return out


def _cluster_label(ids: List[int]) -> str:
    """Human-readable tag for one OR many parent clusters.

    Single id  -> "Cluster 3"
    Multiple   -> "Clusters 0/3/5"  (sorted, joined by /)
    """
    if len(ids) == 1:
        return f"Cluster {ids[0]}"
    return "Clusters " + "/".join(str(i) for i in sorted(set(ids)))


def _scope_decision(cluster: int, cap_mean: Optional[float]) -> Dict[str, Any]:
    """Lightweight re-grade per sub-cluster — uses the same thresholds as
    ClusterAgent but tags with a 'sub_' prefix so the UI can render them
    inside the refinement panel."""
    if cap_mean is None:
        choice = "needs_review"
        reason = "capacity stats unavailable"
    elif cap_mean >= 1.9:
        choice = "direct_reuse"
        reason = f"Sub-cluster {cluster}: mean capacity {cap_mean:.3f} — recommend direct reuse."
    elif cap_mean >= 1.7:
        choice = "second_life"
        reason = f"Sub-cluster {cluster}: mean capacity {cap_mean:.3f} — recommend stationary storage / 2nd-life."
    else:
        choice = "material_recycle"
        reason = f"Sub-cluster {cluster}: mean capacity {cap_mean:.3f} — recommend material recovery."
    return {"choice": choice, "reason": reason}


class RefinementAgent(AgentBase):
    """All-200 sub-cluster refinement agent."""

    agent_id = "refine"
    node_id = "refine"

    REFINE_DATASET_ID = "builtin-all-200"  # development default

    async def _ensure_all200_ddia_code(self, cells: List[int]) -> Optional[str]:
        scoped = self.session.context.get("designed_code_by_dataset") or {}
        entry = scoped.get(self.REFINE_DATASET_ID) if isinstance(scoped, dict) else None
        code = entry.get("code") if isinstance(entry, dict) else None
        if code:
            return str(code)

        await self.thinking(
            "All-200 refinement needs its own CDDI function. "
            f"Designing CDDI on selected All-200 cells (n={len(cells)}) before sub-clustering."
        )
        if not is_reflective_search_mode(self.session.mode):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: mmd_designer.design_mmd(
                    dataset_id=self.REFINE_DATASET_ID,
                    version=int(self.session.context.get("last_designed_version", 1)) + 1,
                    max_attempts=3,
                    sandbox_cells=min(max(len(cells), 4), 20),
                    cells_filter=cells,
                    write_back=False,
                ),
            )
            if not result.passed:
                return None
            by_dataset = dict(self.session.context.get("designed_code_by_dataset") or {})
            by_dataset[self.REFINE_DATASET_ID] = {
                "code": result.code,
                "function_name": result.function_name,
                "selected_cells": list(cells),
                "t": time.time(),
            }
            self.session.context["designed_code_by_dataset"] = by_dataset
            self.session.context["last_designed_code"] = result.code
            self.session.context["last_designed_dataset_id"] = self.REFINE_DATASET_ID
            self.session.context["last_designed_version"] = int(self.session.context.get("last_designed_version", 1)) + 1
            await store.update(self.session)
            return result.code

        original_dataset = self.session.active_dataset_id
        self.session.active_dataset_id = self.REFINE_DATASET_ID
        await store.update(self.session)
        try:
            designer = DesignerAgent(self.session)
            result = await designer.run(
                user_text="design CDDI for All-200 refinement",
                cells_filter=cells,
            )
        finally:
            self.session.active_dataset_id = original_dataset
            self.session.set_node(self.node_id, "running")
            self.session.active_agent = self.agent_id
            await store.update(self.session)

        if not result.get("passed"):
            return None
        scoped = self.session.context.get("designed_code_by_dataset") or {}
        entry = scoped.get(self.REFINE_DATASET_ID) if isinstance(scoped, dict) else None
        code = entry.get("code") if isinstance(entry, dict) else None
        return str(code) if code else None

    async def run(
        self,
        user_text: str = "",
        *,
        cluster_ids: Optional[List[int]] = None,
        **_: Any,
    ) -> Dict[str, Any]:  # type: ignore[override]
        """Run refinement on one OR several parent clusters.

        Resolution order for which clusters to refine:
            1. Explicit ``cluster_ids`` kwarg (passed by the
               ``refine_clusters`` REST action).
            2. Cluster ids parsed from ``user_text`` (e.g. "refine clusters
               0, 3 and 5").
            3. ``session.selected_clusters`` (sticky multi-selection from
               the Decisions UI).
            4. ``[session.selected_cluster]`` (legacy single selection).
        """
        await self.started()

        # ---- 1. pick parent cluster(s) ----
        fallback: List[int] = []
        if cluster_ids:
            fallback = [int(c) for c in cluster_ids]
        elif self.session.selected_clusters:
            fallback = [int(c) for c in self.session.selected_clusters]
        elif self.session.selected_cluster is not None:
            fallback = [int(self.session.selected_cluster)]

        parents = _parse_clusters(user_text, fallback)
        # Drop dupes but preserve sort order for stable IDs in filenames.
        parents = sorted({int(p) for p in parents})

        if not parents:
            msg = (
                "Tell me which cluster(s) you want to refine — e.g. "
                "‘I'm interested in Cluster 1’, or tick several cluster "
                "cards then click *Refine selected* to run a multi-cluster "
                "All-200 sub-analysis."
            )
            await self.say(msg)
            await self.finished(result={"passed": False, "error": "no cluster selected"})
            return {"passed": False, "error": "no cluster selected"}

        # Persist selections — both legacy single and the new list so chat
        # context / older code paths still see the right thing.
        self.session.selected_clusters = parents
        self.session.selected_cluster = parents[0]
        await store.update(self.session)

        label = _cluster_label(parents)

        # ---- 2. locate prior Cluster_label.xlsx ----
        cluster_xlsx_str = self.session.context.get("last_cluster_xlsx")
        if not cluster_xlsx_str or not Path(cluster_xlsx_str).exists():
            await self.failed(
                "No prior cluster artefact found. Please run the Cluster agent first."
            )
            return {"passed": False, "error": "Cluster_label.xlsx missing"}
        cluster_xlsx = Path(cluster_xlsx_str)

        try:
            cells_by_cluster = _read_cells_for_clusters(cluster_xlsx, parents)
        except Exception as exc:
            await self.failed(f"Failed to read {cluster_xlsx.name}: {exc}")
            return {"passed": False, "error": str(exc)}

        # Flatten: the All-200 sub-analysis runs on the union of cells
        # across every selected parent cluster, then re-clusters that
        # combined pool.
        cells: List[int] = sorted({c for sub in cells_by_cluster.values() for c in sub})

        if len(cells) < 4:
            breakdown = ", ".join(
                f"Cluster {cid}={len(v)}" for cid, v in cells_by_cluster.items()
            )
            msg = (
                f"{label} contains only {len(cells)} cell(s) in total ({breakdown}); "
                "I need at least 4 cells to run a meaningful sub-cluster. "
                "Pick a larger cluster or tick more cluster cards."
            )
            await self.say(msg)
            await self.finished(result={"passed": False, "error": "too few cells", "cells": cells})
            return {"passed": False, "error": "too few cells", "cells": cells}

        # ---- 3. resolve / request All-200 dataset ----
        refine_ds_id = self.session.context.get("refine_dataset_id") or self.REFINE_DATASET_ID
        all200_ready = bool(self.session.context.get("all200_ready"))
        try:
            refine_ds = data_resolver.resolve(refine_ds_id)
        except FileNotFoundError:
            refine_ds = None

        if not all200_ready or refine_ds is None or refine_ds.cell_count == 0:
            available_id = refine_ds.dataset_id if refine_ds and refine_ds.cell_count > 0 else None
            breakdown = ", ".join(
                f"Cluster {cid}: {len(v)} cells" for cid, v in cells_by_cluster.items()
            )
            prompt_text = (
                f"To refine {label}, please run the All-200s acquisition "
                f"protocol on the union of cells (n={len(cells)}): {cells}.\n"
                f"  ({breakdown})\n"
                "Once uploaded, click ‘Confirm All-200s Ready’ (or send action.confirm_all200) "
                "and I'll restart the refinement."
            )
            if available_id:
                prompt_text += (
                    f"\n\nDevelopment shortcut: a built-in reference dataset "
                    f"`{available_id}` ({refine_ds.cell_count} cells) is available. "
                    "Click ‘Confirm All-200s Ready’ to use it."
                )
            await self.emit(
                "user_prompt_required",
                prompt="confirm_all200",
                # Single-cluster backwards compat fields:
                cluster=parents[0],
                # New multi-cluster fields:
                clusters=parents,
                cells_by_cluster={str(k): v for k, v in cells_by_cluster.items()},
                cells=cells,
                suggested_dataset_id=available_id,
                message=prompt_text,
            )
            # Persist BOTH single (legacy) and multi pending state so the
            # confirm_all200 handler in pipeline.py can pick whichever is
            # appropriate.
            self.session.context["pending_refine_cluster"] = parents[0]
            self.session.context["pending_refine_clusters"] = parents
            self.session.context["pending_refine_cells"] = cells
            await self.say(prompt_text)
            await self.finished(result={"passed": False, "awaiting": "confirm_all200"})
            return {
                "passed": False,
                "awaiting": "confirm_all200",
                "cells": cells,
                "clusters": parents,
            }

        # ---- 4. ensure All-200-specific CDDI function ----
        gpt_code = await self._ensure_all200_ddia_code(cells)
        if not gpt_code:
            await self.failed(
                "No All-200 CDDI function available. Run the Designer for All-200 first."
            )
            return {"passed": False, "error": "no all200 CDDI function"}

        try:
            mmd_runner.inject_mmd_code(gpt_code)
            mmd_fn = mmd_runner.load_injected_function()
        except Exception as exc:
            await self.failed(f"Failed to inject/load CDDI function: {exc}")
            return {"passed": False, "error": str(exc)}

        # ---- 5. compute CDDI on the All-200 curves for the selected cells ----
        # Filename tag for outputs:
        #   single cluster    -> "cluster3"
        #   multi-cluster     -> "clusters_0-3-5"
        # so artefacts remain unique per selection group.
        if len(parents) == 1:
            tag = f"cluster{parents[0]}"
        else:
            tag = "clusters_" + "-".join(str(p) for p in parents)
        await self.thinking(
            f"Refining {label} ({len(cells)} cells) using "
            f"All-200 dataset `{refine_ds.dataset_id}` ({refine_ds.cell_count} cells). "
            "Running sub-CDDI then re-clustering on OCV/IR/CDDI only."
        )

        sub_csv = settings.session_workspace(self.session.session_id) / f"sub_cddi_{tag}.csv"

        loop = asyncio.get_running_loop()
        prog_q: asyncio.Queue[Optional[Any]] = asyncio.Queue()

        async def consume() -> None:
            while True:
                p = await prog_q.get()
                if p is None:
                    break
                await self.emit("chart_data", chart="sub_mmd_progress", data={
                    "scope": "sub",
                    # Backwards compat: keep `cluster` as the primary parent
                    # so older frontend code still gets a sensible number.
                    "cluster": parents[0],
                    "clusters": parents,
                    "phase": p.phase,
                    "index": p.index,
                    "total": p.total,
                    "completed": p.index,
                    "cell_num": p.cell_num,
                    "mmd_loss": p.mmd_loss,
                    "elapsed_s": p.elapsed_s,
                    "error": p.error,
                })

        consumer = asyncio.create_task(consume())

        compute_call = await self.tool_call(
            "mmd_runner.run_full_pipeline (subset)",
            {
                "data_folder": str(refine_ds.folder),
                "cells": sorted(cells),
                "out_csv": str(sub_csv),
            },
            status="running",
        )

        start_ts = time.time()
        try:
            mmd_result = await loop.run_in_executor(
                None,
                lambda: mmd_runner.run_full_pipeline(
                    data_folder=Path(refine_ds.folder),
                    model_filename=refine_ds.model_file,
                    char_filename=refine_ds.char_file,
                    out_csv=sub_csv,
                    mmd_fn=mmd_fn,
                    on_progress=lambda p: loop.call_soon_threadsafe(prog_q.put_nowait, p),
                    cells_filter=cells,
                ),
            )
        finally:
            await prog_q.put(None)
            await consumer

        compute_elapsed = time.time() - start_ts
        results = mmd_result.get("results", [])
        await self.tool_done(
            compute_call,
            "mmd_runner.run_full_pipeline (subset)",
            duration_ms=compute_elapsed * 1000,
            result={
                "computed_cells": len(results),
                "out_csv": mmd_result.get("saved_to"),
                "skipped": len(mmd_result.get("skipped", [])),
            },
        )
        if not results or len(results) < 4:
            await self.failed(
                f"Sub-CDDI produced only {len(results)} valid result(s); need ≥4 to sub-cluster."
            )
            return {"passed": False, "error": "insufficient sub-CDDI results"}

        # ---- 6. build selected-cell OCV/IR/CDDI input and sub-cluster ----
        statis_xlsx = Path(refine_ds.indicator_path) if refine_ds.indicator_path else None
        if not statis_xlsx or not statis_xlsx.exists():
            await self.failed("Refine dataset has no OCV/IR statistics workbook.")
            return {"passed": False, "error": "statis file missing"}

        if len(parents) == 1:
            sub_label_xlsx = settings.session_workspace(self.session.session_id) / f"Sub_Cluster{parents[0]}_label.xlsx"
        else:
            sub_label_xlsx = settings.session_workspace(self.session.session_id) / (
                "Sub_Clusters_" + "-".join(str(p) for p in parents) + "_label.xlsx"
            )
        input_dir = settings.session_workspace(self.session.session_id) / "cluster_inputs"
        all200_input_xlsx = input_dir / f"all200_cluster_input_{tag}.xlsx"
        try:
            all200_input_df = cluster_input_builder.build_cluster_input(
                statis_xlsx=statis_xlsx,
                ddia_csv=sub_csv,
                out_xlsx=all200_input_xlsx,
                cell_nums=cells,
            )
        except Exception as exc:  # noqa: BLE001
            await self.failed(f"Failed to build All-200 clustering input: {exc}")
            return {"passed": False, "error": str(exc)}
        input_art = Artifact(
            name=all200_input_xlsx.name,
            kind="xlsx",
            path=str(all200_input_xlsx),
            size=all200_input_xlsx.stat().st_size,
            preview={
                "parent_clusters": parents,
                "n_samples": int(len(all200_input_df)),
                "columns": ["Cell_num", "OCV", "IR", "CDDI"],
                "description": "Selected-cell All-200 clustering input; Capacity is not included.",
            },
        )
        d_input = self.session.add_artifact(input_art)
        await store.update(self.session)
        await self.artifact(
            name=d_input["name"],
            kind=d_input["kind"],
            path=d_input["path"],
            **(d_input.get("preview") or {}),
        )
        iter_q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

        def on_iter(it: int, best: Individual) -> None:
            if it % ITER_PROGRESS_STRIDE != 0:
                return
            tick = {
                "scope": "sub",
                "cluster": parents[0],
                "clusters": parents,
                "iter": it,
                "fitness": float(best.fitness),
                "dbi": float(best.dbi),
                "cap_std": float(best.cap_std),
                "k": int(best.matrix[0, 0]),
            }
            loop.call_soon_threadsafe(iter_q.put_nowait, tick)

        async def consume_iter() -> None:
            while True:
                tick = await iter_q.get()
                if tick is None:
                    break
                await self.emit("chart_data", chart="sub_cluster_iter", data=tick)

        iter_consumer = asyncio.create_task(consume_iter())

        cluster_call = await self.tool_call(
            "clustering_runner.cluster_subset",
            {
                "mmd_csv": None,
                "indicator_xlsx": str(all200_input_xlsx),
                "cells": sorted(cells),
                "feature_source": "all_200_ocv_ir_ddia",
            },
            status="running",
        )

        start_ts2 = time.time()
        try:
            sub = await loop.run_in_executor(
                None,
                lambda: clustering_runner.cluster_subset(
                    mmd_csv=None,
                    indicator_xlsx=all200_input_xlsx,
                    cell_nums=cells,
                    out_xlsx=sub_label_xlsx,
                    pop_size=120,
                    max_iter=60,
                    min_k=2,
                    max_k=4,
                    use_capacity=False,
                    seed=42,
                    on_iter=on_iter,
                    feature_weights=self.session.context.get("last_cluster_feature_weights"),
                    posthoc_xlsx=statis_xlsx,
                ),
            )
        finally:
            await iter_q.put(None)
            await iter_consumer

        sub_elapsed = time.time() - start_ts2
        await self.tool_done(
            cluster_call,
            "clustering_runner.cluster_subset",
            duration_ms=sub_elapsed * 1000,
            result={
                "k": sub["k"],
                "dbi": sub["dbi"],
                "fitness": sub["fitness"],
                "n_samples": sub["n_samples"],
                "cluster_xlsx": sub.get("cluster_xlsx"),
            },
        )

        # ---- 7. emit sub-cluster chart_data with scope='sub' ----
        summary = sub["summary"]
        sub_decisions = []
        for c in summary["per_cluster"]:
            sub_decisions.append({
                "cluster": c["cluster"],
                "size": c["size"],
                "capacity": c["capacity"],
                "mmd_loss": c["mmd_loss"],
                "decision": _scope_decision(c["cluster"], c["capacity"]["mean"]),
            })

        # All sub_* chart events carry both `parent_cluster` (single,
        # legacy) and `parent_clusters` (list, new). The frontend Refine
        # tab can use either; older charts only need the singular field.
        primary_parent = parents[0]
        await self.emit("chart_data", chart="sub_cluster_scatter", data={
            "scope": "sub",
            "parent_cluster": primary_parent,
            "parent_clusters": parents,
            "points": summary["scatter"],
            "axes": summary["axes"],
            "k": summary["k"],
        })
        await self.emit("chart_data", chart="sub_capacity_box", data={
            "scope": "sub",
            "parent_cluster": primary_parent,
            "parent_clusters": parents,
            "clusters": summary["boxplot"],
        })
        await self.emit("chart_data", chart="sub_fingerprint", data={
            "scope": "sub",
            "parent_cluster": primary_parent,
            "parent_clusters": parents,
            **summary["fingerprint"],
        })
        await self.emit("chart_data", chart="sub_cluster_summary", data={
            "scope": "sub",
            "parent_cluster": primary_parent,
            "parent_clusters": parents,
            "k": sub["k"],
            "n_samples": sub["n_samples"],
            "fitness": sub["fitness"],
            "dbi": sub["dbi"],
            "elapsed_s": sub["elapsed_s"],
            "per_cluster": sub_decisions,
            "cells": sub["cell_nums"],
        })

        # artefact
        if sub.get("cluster_xlsx") and Path(sub["cluster_xlsx"]).exists():
            cx = Path(sub["cluster_xlsx"])
            art = Artifact(
                name=cx.name,
                kind="xlsx",
                path=str(cx),
                size=cx.stat().st_size,
                preview={
                    "parent_cluster": primary_parent,
                    "parent_clusters": parents,
                    "k": sub["k"],
                    "n_samples": sub["n_samples"],
                    "sizes": [c["size"] for c in summary["per_cluster"]],
                },
            )
            d = self.session.add_artifact(art)
            await store.update(self.session)
            await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))

        # Stash per-parent and aggregate context. We key the per-cluster
        # path by the SAME tag we used for the filename so callers can
        # locate any past refinement run.
        self.session.context[f"sub_cluster_xlsx_{tag}"] = sub.get("cluster_xlsx")
        for parent_id in parents:
            # Preserve/update one key per parent so the global sorting table
            # can include every refinement done so far, not just this run.
            self.session.context[f"sub_cluster_xlsx_{parent_id}"] = sub.get("cluster_xlsx")
        self.session.context["last_sub_cluster"] = {
            "parent_cluster": primary_parent,
            "parent_clusters": parents,
            "k": sub["k"],
            "sizes": [c["size"] for c in summary["per_cluster"]],
            "cells": sub["cell_nums"],
        }
        await store.update(self.session)

        # ---- 7b. merge sub-cluster labels back into the parent clustering ---
        # Build the (parent -> sub_xlsx) map for THIS refinement. When the
        # user refined multiple parents in one go, every refined parent gets
        # the same `sub.cluster_xlsx` (since cluster_subset re-clusters the
        # UNION pool); we still expose them so the merge function can read
        # cells back per-parent.
        sub_xlsx = sub.get("cluster_xlsx")
        merged_payload: Optional[Dict[str, Any]] = None
        if sub_xlsx and Path(sub_xlsx).exists():
            sub_paths: Dict[int, Path] = {int(p): Path(sub_xlsx) for p in parents}
            merged_out = settings.session_workspace(self.session.session_id) / (
                f"Merged_Cluster_label_{tag}.xlsx"
            )
            try:
                merged_payload = cluster_merge.merge_subclusters_back(
                    parent_xlsx=cluster_xlsx,
                    sub_xlsxs_by_parent=sub_paths,
                    out_xlsx=merged_out,
                )
            except Exception as exc:  # noqa: BLE001
                await self.log(f"merge_subclusters_back failed: {exc}", level="warning")
                merged_payload = None

        global_payload: Optional[Dict[str, Any]] = None
        if sub_xlsx and Path(sub_xlsx).exists():
            global_out = settings.session_workspace(self.session.session_id) / "Global_Sorting_Result.xlsx"
            try:
                global_payload = cluster_merge.merge_all_available_subclusters(
                    parent_xlsx=cluster_xlsx,
                    session_ctx=self.session.context,
                    out_xlsx=global_out,
                )
            except Exception as exc:  # noqa: BLE001
                await self.log(f"merge_all_available_subclusters failed: {exc}", level="warning")
                global_payload = None

        if merged_payload is not None:
            merged_disp = merged_payload.get("dispersion") or {}
            await self.emit("chart_data", chart="merged_cluster_dispersion", data={
                "parent_clusters": parents,
                "k_before": merged_payload.get("k_before"),
                "k_after": merged_payload.get("k_after"),
                "mapping": merged_payload.get("mapping"),
                "dispersion": merged_disp,
            })
            if merged_payload.get("xlsx_path") and Path(merged_payload["xlsx_path"]).exists():
                mp = Path(merged_payload["xlsx_path"])
                art = Artifact(
                    name=mp.name,
                    kind="xlsx",
                    path=str(mp),
                    size=mp.stat().st_size,
                    preview={
                        "parent_clusters": parents,
                        "k_before": merged_payload.get("k_before"),
                        "k_after": merged_payload.get("k_after"),
                        "description": "Original labels with refined parents replaced by their All-200 sub-cluster splits.",
                    },
                )
                d = self.session.add_artifact(art)
                await store.update(self.session)
                await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))
            if (
                merged_payload.get("dispersion_xlsx_path")
                and Path(merged_payload["dispersion_xlsx_path"]).exists()
            ):
                dp = Path(merged_payload["dispersion_xlsx_path"])
                art = Artifact(
                    name=dp.name,
                    kind="xlsx",
                    path=str(dp),
                    size=dp.stat().st_size,
                    preview={
                        "parent_clusters": parents,
                        "k_after": merged_payload.get("k_after"),
                        "description": "Dispersion statistics recomputed after merging All-200 sub-clusters.",
                    },
                )
                d = self.session.add_artifact(art)
                await store.update(self.session)
                await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))
            self.session.context["last_merged_xlsx"] = merged_payload.get("xlsx_path")
            self.session.context["last_merged_dispersion_xlsx"] = merged_payload.get("dispersion_xlsx_path")
            self.session.context["last_merged_dispersion"] = merged_disp
            await store.update(self.session)

        if global_payload is not None:
            global_disp = global_payload.get("dispersion") or {}
            await self.emit("chart_data", chart="global_sorting_dispersion", data={
                "parent_clusters": parents,
                "k_before": global_payload.get("k_before"),
                "k_after": global_payload.get("k_after"),
                "mapping": global_payload.get("mapping"),
                "dispersion": global_disp,
            })
            if global_payload.get("xlsx_path") and Path(global_payload["xlsx_path"]).exists():
                gp = Path(global_payload["xlsx_path"])
                art = Artifact(
                    name=gp.name,
                    kind="xlsx",
                    path=str(gp),
                    size=gp.stat().st_size,
                    preview={
                        "parent_clusters": parents,
                        "k_before": global_payload.get("k_before"),
                        "k_after": global_payload.get("k_after"),
                        "description": (
                            "Global sorting table: unrefined cells keep First-200 labels; "
                            "refined cells use All-200 sub-cluster labels."
                        ),
                    },
                )
                d = self.session.add_artifact(art)
                await store.update(self.session)
                await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))
            if (
                global_payload.get("dispersion_xlsx_path")
                and Path(global_payload["dispersion_xlsx_path"]).exists()
            ):
                gdp = Path(global_payload["dispersion_xlsx_path"])
                art = Artifact(
                    name=gdp.name,
                    kind="xlsx",
                    path=str(gdp),
                    size=gdp.stat().st_size,
                    preview={
                        "k_after": global_payload.get("k_after"),
                        "description": "Final dispersion statistics for Global_Sorting_Result.xlsx.",
                    },
                )
                d = self.session.add_artifact(art)
                await store.update(self.session)
                await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))
            self.session.context["last_global_sorting_xlsx"] = global_payload.get("xlsx_path")
            self.session.context["last_global_sorting_dispersion_xlsx"] = global_payload.get("dispersion_xlsx_path")
            self.session.context["last_global_sorting_dispersion"] = global_disp
            await store.update(self.session)

        sizes = [c["size"] for c in summary["per_cluster"]]
        result_payload = {
            "passed": True,
            "parent_cluster": primary_parent,
            "parent_clusters": parents,
            "k": sub["k"],
            "sizes": sizes,
            "dbi": sub["dbi"],
            "fitness": sub["fitness"],
            "elapsed_s": sub["elapsed_s"],
            "cells": sub["cell_nums"],
            "global_sorting_xlsx": (
                global_payload.get("xlsx_path") if global_payload is not None else None
            ),
        }
        await self.finished(result=result_payload)

        msg_parts = [
            f"Refined {label}: {sub['n_samples']} cells re-split into "
            f"{sub['k']} sub-cluster(s) (sizes={sizes}, DBI={sub['dbi']:.3f}). "
            f"Switch the main stage to 'Refine' to inspect the sub-overlay."
        ]
        if merged_payload is not None:
            mapping = merged_payload.get("mapping") or []
            disp = merged_payload.get("dispersion") or {}
            cap_aggr = disp.get("capacity_aggregate") or {}
            map_lines = [
                f"  • parent Cluster {m['parent_cluster']} → {m['display']} "
                f"(new label {m['new_label']}, n={m['n_cells']})"
                for m in mapping
            ]
            merge_section = [
                f"**Merged labelling**: k went from "
                f"{merged_payload['k_before']} → {merged_payload['k_after']} groups.",
            ]
            if map_lines:
                merge_section.append("\n".join(map_lines))
            cap_lines: list[str] = []
            for r in disp.get("per_cluster") or []:
                cap = r.get("Capacity") or {}
                if cap.get("mean") is not None and cap.get("std") is not None:
                    cv_pct = (cap.get("cv") or 0.0) * 100.0
                    cap_lines.append(
                        f"  • Cluster {r['cluster']} (n={r['size']}): "
                        f"μ={cap['mean']:.3f}, σ={cap['std']:.3f}, CV={cv_pct:.2f}%"
                    )
            if cap_lines:
                merge_section.append("Capacity dispersion AFTER merge:\n" + "\n".join(cap_lines))
            extras: list[str] = []
            if cap_aggr.get("f_ratio") is not None:
                extras.append(f"between/within F={cap_aggr['f_ratio']:.2f}")
            if cap_aggr.get("mean_of_std") is not None:
                extras.append(f"mean within-cluster σ={cap_aggr['mean_of_std']:.3f}")
            if extras:
                merge_section.append("Aggregate: " + ", ".join(extras) + ".")
            # Curve-feature dispersion table (top-N noisiest curve features
            # + per-cluster σ). Mirrors what ClusterAgent posts after a
            # main cluster run so the user sees the same level of detail
            # for the merged grouping. Empty dispersion / Capacity-only
            # datasets return None and the block is silently dropped.
            curve_md = clustering_runner.format_curve_dispersion_markdown(disp)
            if curve_md:
                merge_section.append(curve_md)
            if merged_payload.get("xlsx_path"):
                fname = Path(merged_payload["xlsx_path"]).name
                merge_section.append(
                    f"📥 Download `{fname}` from the Artifacts panel "
                    "(sheets: merged_labels, mapping). The recomputed dispersion table is "
                    "also available as a separate `_dispersion.xlsx` artifact."
                )
            msg_parts.append("\n\n".join(merge_section))
        if global_payload is not None and global_payload.get("xlsx_path"):
            gname = Path(global_payload["xlsx_path"]).name
            msg_parts.append(
                f"**Global sorting result** is ready: `{gname}`. "
                "未精细化电芯保留 First-200 主分组；已精细化电芯使用 All-200 子分组，"
                "并在 `Label_source` 中标明 `first_200_main` / `all_200_refined`。"
            )
        await self.say("\n\n".join(msg_parts))
        return {**result_payload, "cluster_xlsx": sub.get("cluster_xlsx")}
