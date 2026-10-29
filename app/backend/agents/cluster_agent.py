"""Cluster Agent.

Inputs:
    - session.context["last_mmd_csv"] (set by ComputeAgent), or
      falls back to ``mmd_results.csv`` inside the session workspace.
    - dataset.indicator_path points to Statis data.xlsx for OCV/IR only.
      A session-local cluster_inputs/*.xlsx file is built with
      Cell_num/OCV/IR/CDDI before clustering.

Outputs (events):
    - thinking lines + a single tool_call wrapping the clustering job
    - chart_data(cluster_iter) live per-iteration fitness/DBI (every N iters)
    - chart_data(cluster_scatter)     final OCV vs IR scatter (color = cluster)
    - chart_data(capacity_box)        per-cluster capacity arrays for boxplot
    - chart_data(fingerprint)         per-cluster normalised feature means (radar)
    - chart_data(cluster_summary)     per-cluster size, mean capacity, mean MMD
    - artifact(Cluster_label.xlsx)    full table with new Cluster_label column
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.backend.agents.base import AgentBase
from app.backend.orchestrator.session_store import Artifact, store
from app.backend.services import cluster_input_builder, clustering_runner, data_resolver
from app.backend.services.clustering_runner import Individual
from app.backend.settings import settings


ITER_PROGRESS_STRIDE = 5


class _ClusterCancelled(RuntimeError):
    """Raised inside the GA on_iter callback when the user cancels via chat."""
    pass


def _recycle_decision(mean_cap: Optional[float], cap_std: Optional[float]) -> Dict[str, Any]:
    """Heuristic recommendation per cluster — refined later by domain users."""
    if mean_cap is None:
        return {"choice": "needs_review", "reason": "capacity stats unavailable"}
    if mean_cap >= 1.9:
        return {
            "choice": "direct_reuse",
            "reason": f"Mean capacity {mean_cap:.3f} is high; cells appear healthy.",
        }
    if mean_cap >= 1.7:
        return {
            "choice": "second_life",
            "reason": f"Mean capacity {mean_cap:.3f} suggests reduced but usable health "
                      "(stationary storage candidates).",
        }
    return {
        "choice": "material_recycle",
        "reason": f"Mean capacity {mean_cap:.3f} indicates significant degradation; recommend "
                  "material recovery.",
    }


class ClusterAgent(AgentBase):
    agent_id = "cluster"
    node_id = "cluster"

    async def _log_llm(self, tag: str, line: str) -> None:
        """Stream a single runtime-log line into the shared llm_log channel.

        Same format as Designer / Compute so the user has ONE live terminal
        showing every agent's heartbeat. Tags: info | ok | warn | error.
        """
        await self.emit(
            "chart_data",
            chart="llm_log",
            data={"line": line[:500], "tag": tag, "agent": self.agent_id},
        )

    async def run(
        self,
        user_text: str = "",
        *,
        target_k: Optional[int] = None,
        cluster_params: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:  # type: ignore[override]
        await self.started()
        await self._log_llm("info", "Cluster agent started.")
        # `target_k` is intentionally NOT sticky. Earlier versions read
        # session.context["target_k"] here, so clicking Auto after a fixed K
        # run silently reused the previous K. Now Auto really means adaptive.
        if target_k is not None:
            try:
                target_k = int(target_k)
            except (TypeError, ValueError):
                target_k = None
        if target_k is not None:
            target_k = max(2, min(target_k, 12))  # clamp to reasonable range

        # Conversational overrides: min_k / max_k / pop_size / max_iter /
        # feedback_metric. RouterAgent parses these from chat and we apply
        # them on top of the defaults below. `target_k` (explicit single k)
        # still wins over min_k/max_k range.
        cparams = dict(cluster_params or {})
        cfg_pop = int(cparams.get("pop_size") or 200)
        cfg_iter = int(cparams.get("max_iter") or 80)
        cfg_metric = str(cparams.get("feedback_metric") or "combined")
        cfg_min_k = cparams.get("min_k")
        cfg_max_k = cparams.get("max_k")
        cfg_weights = clustering_runner.normalize_feature_weights(
            cparams.get("feature_weights")
        )

        try:
            ds = data_resolver.resolve(self.session.active_dataset_id)
        except FileNotFoundError as exc:
            await self.failed(str(exc))
            return {"passed": False, "error": str(exc)}

        if ds.indicator_path is None or not Path(ds.indicator_path).exists():
            await self.failed(
                "Active dataset has no OCV/IR statistics workbook. Cannot run clustering."
            )
            return {"passed": False, "error": "indicator file missing"}

        csv_by_dataset = dict(self.session.context.get("ddia_csv_by_dataset") or {})
        csv_by_dataset.update(self.session.context.get("cddi_csv_by_dataset") or {})
        csv_str = (
            csv_by_dataset.get(ds.dataset_id)
            if isinstance(csv_by_dataset, dict)
            else None
        ) or self.session.context.get("last_cddi_csv") or self.session.context.get("last_mmd_csv")
        csv = Path(csv_str) if csv_str else settings.session_workspace(self.session.session_id) / "cddi_results.csv"
        if not csv_str and not csv.exists():
            csv = csv.with_name("mmd_results.csv")
        if not csv.exists():
            await self.failed(
                "No CDDI results available. Please run the CDDI Compute agent first."
            )
            return {"passed": False, "error": "CDDI results missing"}

        # Resolve final k bounds: explicit single-k > chat range > Auto.
        if target_k is not None:
            run_min_k, run_max_k = target_k, target_k
            k_mode = f"fixed k={target_k}"
        elif cfg_min_k is not None and cfg_max_k is not None:
            run_min_k = max(2, min(int(cfg_min_k), 12))
            run_max_k = max(run_min_k, min(int(cfg_max_k), 12))
            k_mode = f"k range {run_min_k}..{run_max_k}"
        else:
            run_min_k, run_max_k = 2, None
            k_mode = "adaptive (GA-KMeans auto-selects k)"

        input_dir = settings.session_workspace(self.session.session_id) / "cluster_inputs"
        cluster_input_xlsx = input_dir / "first200_cluster_input.xlsx"
        try:
            input_df = cluster_input_builder.build_cluster_input(
                statis_xlsx=Path(ds.indicator_path),
                ddia_csv=csv,
                out_xlsx=cluster_input_xlsx,
            )
        except Exception as exc:  # noqa: BLE001
            await self.failed(f"Failed to build clustering input: {exc}")
            return {"passed": False, "error": str(exc)}
        input_art = Artifact(
            name=cluster_input_xlsx.name,
            kind="xlsx",
            path=str(cluster_input_xlsx),
            size=cluster_input_xlsx.stat().st_size,
            preview={
                "n_samples": int(len(input_df)),
                "columns": ["Cell_num", "OCV", "IR", "CDDI"],
                "description": "Minimal clustering input; Capacity is not included.",
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

        await self.thinking(
            f"Clustering session `{self.session.session_id}`: "
            f"input={cluster_input_xlsx.name}, mode={k_mode}. "
            f"pop={cfg_pop}, max_iter={cfg_iter}, feedback={cfg_metric}. "
            f"weights OCV:IR:CDDI={cfg_weights['ocv']}:"
            f"{cfg_weights['ir']}:{cfg_weights['ddia']}. "
            "Running GA-KMeans on OCV/IR/CDDI only; Capacity is not used."
        )
        await self._log_llm(
            "info",
            f"Starting GA-KMeans: pop={cfg_pop}, max_iter={cfg_iter}, {k_mode}, "
            f"feedback={cfg_metric}, weights OCV/IR/CDDI="
            f"{cfg_weights['ocv']}/{cfg_weights['ir']}/{cfg_weights['ddia']}.",
        )

        out_xlsx = settings.session_workspace(self.session.session_id) / "first200_cluster_input_used.xlsx"

        cluster_kwargs: Dict[str, Any] = {
            "mmd_csv": None,
            "indicator_xlsx": str(cluster_input_xlsx),
            "out_xlsx": str(out_xlsx),
            "pop_size": cfg_pop,
            "max_iter": cfg_iter,
            "feedback_metric": cfg_metric,
            "feature_weights": cfg_weights,
            "min_k": run_min_k,
            "max_k": run_max_k,
            "use_capacity": False,
        }

        call_id = await self.tool_call(
            "clustering_runner.cluster_dataset",
            cluster_kwargs,
            status="running",
        )

        loop = asyncio.get_running_loop()
        iter_q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        history_box: Dict[str, Any] = {"history": []}

        def on_iter(it: int, best: Individual) -> None:
            # Cooperative cancel: pipeline flips `cancel_cluster` when the
            # user types "stop" / "abort" mid-GA. Raising propagates out of
            # the `run_in_executor`-wrapped clustering_runner call, which we
            # catch below and treat as a clean cancellation. We don't reset
            # the flag here — the post-mortem block does that so we can
            # distinguish "running" vs "just finished cancelled".
            if self.session.context.get("cancel_cluster"):
                raise _ClusterCancelled()
            if it % ITER_PROGRESS_STRIDE != 0:
                return
            tick = {
                "iter": it,
                "fitness": float(best.fitness),
                "dbi": float(best.dbi),
                "cap_std": float(best.cap_std),
                "k": int(best.matrix[0, 0]),
            }
            history_box["history"].append(tick)
            loop.call_soon_threadsafe(iter_q.put_nowait, tick)
            # Also push a llm_log line so the user sees text-form heartbeat
            # in the shared log terminal (useful when the cluster chart isn't
            # visible yet).
            loop.call_soon_threadsafe(
                self._log_q_put,  # type: ignore[arg-type]
                ("info", f"GA iter {it}: k={int(best.matrix[0, 0])} fitness={best.fitness:.3f} DBI={best.dbi:.3f}"),
            )

        # Tiny helper to bridge thread → async log queue (used by on_iter).
        log_q: asyncio.Queue[Optional[tuple]] = asyncio.Queue()

        def _log_q_put(item) -> None:
            log_q.put_nowait(item)
        self._log_q_put = _log_q_put  # type: ignore[attr-defined]

        async def consume_logs() -> None:
            while True:
                item = await log_q.get()
                if item is None:
                    break
                tag, line = item
                await self._log_llm(tag, line)
        log_consumer = asyncio.create_task(consume_logs())

        async def consume() -> None:
            while True:
                tick = await iter_q.get()
                if tick is None:
                    break
                await self.emit("chart_data", chart="cluster_iter", data=tick)

        consumer = asyncio.create_task(consume())
        start_ts = time.time()
        cancelled = False
        try:
            cluster_result = await loop.run_in_executor(
                None,
                lambda: clustering_runner.cluster_dataset(
                    mmd_csv=None,
                    indicator_xlsx=cluster_input_xlsx,
                    out_xlsx=out_xlsx,
                    pop_size=cfg_pop,
                    max_iter=cfg_iter,
                    min_k=run_min_k,
                    max_k=run_max_k,
                    use_capacity=False,
                    seed=42,
                    on_iter=on_iter,
                    feedback_metric=cfg_metric,
                    feature_weights=cfg_weights,
                    posthoc_xlsx=Path(ds.indicator_path),
                ),
            )
        except _ClusterCancelled:
            cancelled = True
            cluster_result = None  # filled below from history_box if needed
        finally:
            await iter_q.put(None)
            await log_q.put(None)
            await consumer
            await log_consumer

        if cancelled:
            self.session.context["cancel_cluster"] = False
            elapsed = time.time() - start_ts
            best_so_far = history_box["history"][-1] if history_box["history"] else None
            await self._log_llm(
                "warn",
                f"Clustering cancelled at iter ~{best_so_far['iter'] if best_so_far else '?'}: "
                "no labels were written.",
            )
            await self.tool_done(
                call_id,
                "clustering_runner.cluster_dataset",
                duration_ms=elapsed * 1000,
                error="cancelled by user",
            )
            await self.emit(
                "agent_finished",
                error="cancelled by user",
                status="cancelled",
                reason="user_cancel",
                best_so_far=best_so_far,
            )
            self.session.set_node(self.node_id, "idle")
            await store.update(self.session)
            await self.say(
                "Clustering cancelled. "
                + (
                    f"Best iter so far: k={best_so_far['k']}, DBI={best_so_far['dbi']:.3f}. "
                    if best_so_far else ""
                )
                + "Tell me what to do next."
            )
            return {"passed": False, "error": "cancelled", "cancelled": True}

        elapsed = time.time() - start_ts
        await self.tool_done(
            call_id,
            "clustering_runner.cluster_dataset",
            duration_ms=elapsed * 1000,
            result={
                "k": cluster_result["k"],
                "dbi": cluster_result["dbi"],
                "fitness": cluster_result["fitness"],
                "n_samples": cluster_result["n_samples"],
                "cluster_xlsx": cluster_result.get("cluster_xlsx"),
            },
        )

        summary = cluster_result["summary"]
        decisions = []
        for clust in summary["per_cluster"]:
            decisions.append({
                "cluster": clust["cluster"],
                "size": clust["size"],
                "capacity": clust["capacity"],
                "mmd_loss": clust["mmd_loss"],
                "decision": _recycle_decision(
                    clust["capacity"]["mean"], clust["capacity"]["std"]
                ),
            })

        await self.emit("chart_data", chart="cluster_scatter", data={
            "points": summary["scatter"],
            "axes": summary["axes"],
            "k": summary["k"],
        })
        await self.emit("chart_data", chart="capacity_box", data={"clusters": summary["boxplot"]})
        await self.emit("chart_data", chart="fingerprint", data=summary["fingerprint"])
        await self.emit("chart_data", chart="cluster_summary", data={
            "k": cluster_result["k"],
            "n_samples": cluster_result["n_samples"],
            "fitness": cluster_result["fitness"],
            "dbi": cluster_result["dbi"],
            "elapsed_s": cluster_result["elapsed_s"],
            "per_cluster": decisions,
        })

        cluster_xlsx = cluster_result.get("cluster_xlsx")
        if cluster_xlsx and Path(cluster_xlsx).exists():
            artifact = Artifact(
                name="Cluster_label.xlsx",
                kind="xlsx",
                path=cluster_xlsx,
                size=Path(cluster_xlsx).stat().st_size,
                preview={
                    "k": cluster_result["k"],
                    "n_samples": cluster_result["n_samples"],
                    "sizes": [c["size"] for c in summary["per_cluster"]],
                },
            )
            d = self.session.add_artifact(artifact)
            await store.update(self.session)
            await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))

        # ---- Dispersion (per-cluster × per-feature variance) ----------------
        # Surface as: chat-stream block, chart_data event, AND downloadable XLSX.
        dispersion = summary.get("dispersion") or {}
        if dispersion:
            await self.emit("chart_data", chart="cluster_dispersion", data=dispersion)
            try:
                dispersion_xlsx = (
                    settings.session_workspace(self.session.session_id)
                    / "cluster_dispersion.xlsx"
                )
                clustering_runner.write_dispersion_xlsx(dispersion, dispersion_xlsx)
                disp_artifact = Artifact(
                    name="cluster_dispersion.xlsx",
                    kind="xlsx",
                    path=str(dispersion_xlsx),
                    size=dispersion_xlsx.stat().st_size,
                    preview={
                        "k": cluster_result["k"],
                        "features": len(dispersion.get("features") or []),
                        "description": "Per-cluster × per-feature mean/std/CV; capacity F-ratio sheet.",
                    },
                )
                d = self.session.add_artifact(disp_artifact)
                await store.update(self.session)
                await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))
            except Exception as exc:  # noqa: BLE001
                await self._log_llm("warn", f"Failed to write dispersion XLSX: {exc}")

        if cluster_result.get("merged_xlsx") and Path(cluster_result["merged_xlsx"]).exists():
            mp = Path(cluster_result["merged_xlsx"])
            artifact = Artifact(
                name=mp.name,
                kind="xlsx",
                path=str(mp),
                size=mp.stat().st_size,
                preview={"description": "OCV/IR/CDDI clustering input actually used by GA-KMeans"},
            )
            d = self.session.add_artifact(artifact)
            await store.update(self.session)
            await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))

        self.session.context["last_cluster_xlsx"] = cluster_xlsx
        self.session.context["last_cluster_k"] = cluster_result["k"]
        self.session.context["last_cluster_per_cluster"] = decisions
        self.session.context["last_cluster_feature_weights"] = cluster_result.get("feature_weights")
        self.session.context["last_cluster_features"] = cluster_result.get("feature_cols")
        await store.update(self.session)

        sizes = [c["size"] for c in summary["per_cluster"]]
        await self._log_llm(
            "ok",
            f"Clustering done: k={cluster_result['k']}, sizes={sizes}, "
            f"DBI={cluster_result['dbi']:.3f}, weights OCV/IR/CDDI="
            f"{cfg_weights['ocv']}/{cfg_weights['ir']}/{cfg_weights['ddia']}, "
            f"elapsed={elapsed:.1f}s.",
        )
        await self.finished(
            result={
                "passed": True,
                "k": cluster_result["k"],
                "sizes": sizes,
                "dbi": cluster_result["dbi"],
                "fitness": cluster_result["fitness"],
                "elapsed_s": cluster_result["elapsed_s"],
                "cluster_xlsx": cluster_xlsx,
            }
        )

        msg_parts = [
            f"Clustered {cluster_result['n_samples']} cells into {cluster_result['k']} "
            f"groups (sizes={sizes}, DBI={cluster_result['dbi']:.3f}, "
            f"fitness={cluster_result['fitness']:.3f}) in {cluster_result['elapsed_s']:.1f}s. "
            f"Feature weights OCV/IR/CDDI = {cfg_weights['ocv']}/"
            f"{cfg_weights['ir']}/{cfg_weights['ddia']}."
        ]
        if dispersion:
            cap_aggr = dispersion.get("capacity_aggregate") or {}
            cap_rows: list[str] = []
            for r in dispersion.get("per_cluster") or []:
                cap = r.get("Capacity") or {}
                if cap.get("std") is not None and cap.get("mean") is not None:
                    cv_pct = (cap.get("cv") or 0.0) * 100.0
                    cap_rows.append(
                        f"  • Cluster {r['cluster']} (n={r['size']}): "
                        f"capacity μ={cap['mean']:.3f}, σ={cap['std']:.3f}, CV={cv_pct:.2f}%"
                    )
            if cap_rows:
                msg_parts.append("**Per-cluster capacity dispersion:**\n" + "\n".join(cap_rows))
            extras: list[str] = []
            if cap_aggr.get("tightest_cluster") is not None:
                extras.append(f"tightest = Cluster {cap_aggr['tightest_cluster']}")
            if cap_aggr.get("loosest_cluster") is not None:
                extras.append(f"loosest = Cluster {cap_aggr['loosest_cluster']}")
            if cap_aggr.get("f_ratio") is not None:
                extras.append(f"between/within variance F={cap_aggr['f_ratio']:.2f}")
            if cap_aggr.get("mean_of_std") is not None:
                extras.append(f"mean within-cluster σ={cap_aggr['mean_of_std']:.3f}")
            if extras:
                msg_parts.append("Aggregate: " + ", ".join(extras) + ".")
            # Curve-feature dispersion block (top-N noisiest features +
            # a per-cluster σ table). Renders as a real markdown table
            # in ChatPanel once react-markdown is enabled.
            curve_md = clustering_runner.format_curve_dispersion_markdown(dispersion)
            if curve_md:
                msg_parts.append(curve_md)
            msg_parts.append(
                "📥 Full per-feature mean/std/CV table available as "
                "`cluster_dispersion.xlsx` in the Artifacts panel."
            )
        msg_parts.append(
            "Review the cluster fingerprint and capacity box plots, then point at the "
            "cluster you want to refine (e.g. 'I'm interested in Cluster 1') to trigger the "
            "All-200s refinement workflow."
        )
        await self.say("\n\n".join(msg_parts))
        return {"passed": True, "k": cluster_result["k"], "cluster_xlsx": cluster_xlsx}
