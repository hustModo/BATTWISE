"""Offline CDDI naming and saved-artifact compatibility regressions."""
from __future__ import annotations

import ast
import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents import chat_agent, router_agent
from app.backend.api import rest
from app.backend.naming import METRIC_NAME, METHOD_NAME, SYSTEM_NAME
from app.backend.orchestrator import pipeline
from app.backend.services import cluster_input_builder, cluster_merge, clustering_runner, llm_router, mmd_runner, planner


def test_names_and_routing() -> None:
    assert (SYSTEM_NAME, METRIC_NAME, METHOD_NAME) == ("BATTWISE", "CDDI", "BATTWISE-CDDI")
    assert "DDIA" not in chat_agent._CHAT_SYSTEM
    for relative in ("pages/Landing.tsx", "components/ChatPanel.tsx",
                     "components/charts/MMDDistribution.tsx", "components/charts/MMDCodeViewer.tsx"):
        text = (ROOT / "app/frontend/src" / relative).read_text(encoding="utf-8")
        assert "DDIA" not in text and "CDDI" in text, relative
    assert "BatRecycler-MAS" not in (ROOT / "app/frontend/index.html").read_text(encoding="utf-8")
    schema_names = {tool["name"] for tool in llm_router.AGENT_TOOLS}
    assert {"design_cddi", "compute_cddi"} <= schema_names
    assert not {"design_ddia", "compute_mmd"} & schema_names
    for tool, agent in (("design_cddi", "designer"), ("design_ddia", "designer"),
                        ("compute_cddi", "compute"), ("compute_mmd", "compute")):
        decision = llm_router._coerce_decision({"tool": tool, "confidence": 1.0})
        assert decision.next_agent == agent, decision
        assert llm_router.decision_to_pipeline_dict(decision)["next_agent"] == agent
    for text in ("Design a new CDDI function", "设计 CDDI 函数", "Design a new DDIA function"):
        assert router_agent._heuristic(text)["intent"] == "design_mmd", text
    for text in ("Compute CDDI across all cells", "计算 CDDI", "Compute MMD"):
        assert router_agent._heuristic(text)["intent"] == "compute_mmd", text
    assert router_agent._extract_cluster_weights("OCV=1 IR=2 CDDI=3") == {"ocv": 1.0, "ir": 2.0, "ddia": 3.0}
    assert router_agent.RouterAgent._extract_mode_hint("Run BATTWISE Reflective Search") == "reflective_search"
    for key in ("cddi_weight", "ddia_weight"):
        assert planner.validate_step_params("cluster", {key: 3}) == {"cddi_weight": 3.0}
        kwargs = pipeline._step_kwargs({"agent": "cluster", "params": {key: 3}})
        assert kwargs["cluster_params"]["feature_weights"]["ddia"] == 3
    assert clustering_runner.normalize_feature_weights({"cddi": 2})["ddia"] == 2


def test_new_and_saved_tables() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        cells = [2, 4, 5, 6]
        values = [0.1, 0.2, 0.3, 0.4]
        stats = folder / "statistics.xlsx"
        pd.DataFrame({"Cell_num": cells, "OCV": [3.3, 3.4, 3.5, 3.6],
                      "IR": [8., 9., 10., 11.], "Capacity": [2., 1.9, 1.8, 1.7]}).to_excel(stats, index=False)
        csv = folder / "cddi_results.csv"
        results = [{"Cell_num": cell, "MMD_Loss": value} for cell, value in zip(cells, values)]
        mmd_runner.save_mmd_csv(results, csv)
        saved = pd.read_csv(csv)
        assert list(saved.columns) == ["Cell_num", "CDDI"]
        assert saved["CDDI"].tolist() == values
        for header in ("CDDI", "DDIA", "MMD_Loss", "DDIA_Loss", "DDI"):
            pd.DataFrame({"Cell_num": cells, header: values}).to_csv(csv, index=False)
            out = folder / "cluster_input.xlsx"
            frame = cluster_input_builder.build_cluster_input(statis_xlsx=stats, ddia_csv=csv, out_xlsx=out)
            assert list(frame.columns) == ["Cell_num", "OCV", "IR", "CDDI"]
            assert frame["CDDI"].tolist() == values
            loaded = clustering_runner.load_cluster_input(out)
            assert loaded["MMD_Loss"].tolist() == values
        both = clustering_runner._normalise_indicator_columns(pd.DataFrame({"CDDI": values, "MMD_Loss": [9.] * 4, "DDIA": [8.] * 4}))
        assert list(both.columns) == ["MMD_Loss"]
        assert both["MMD_Loss"].tolist() == values
        for header in ("CDDI", "DDIA", "MMD_Loss"):
            parent = pd.DataFrame({"Cell_num": cells, "OCV": [3.3, 3.4, 3.5, 3.6],
                                   "IR": [8., 9., 10., 11.], header: values,
                                   "Cluster_label": [0, 0, 1, 1], "DDIA_source": ["first_200s"] * 4})
            path = folder / "parent.xlsx"
            parent.to_excel(path, index=False)
            merged = cluster_merge.merge_subclusters_back(parent_xlsx=path, sub_xlsxs_by_parent={}, out_xlsx=folder / "merged.xlsx")
            output = merged["merged_df"]
            assert "CDDI_source" in output and "DDIA_source" not in output
            assert "CDDI" in output and "MMD_Loss" not in output and "DDIA" not in output
            assert output["CDDI"].tolist() == values
        dispersion = merged["dispersion"]
        label = clustering_runner.format_curve_dispersion_markdown(dispersion)
        assert label and "CDDI" in label and "MMD_Loss" not in label
        report = pd.read_excel(folder / "merged_dispersion.xlsx", sheet_name="by_feature")
        assert "CDDI" in report["feature"].values


def test_periodic_reflection_callback() -> None:
    # Compile only the callback so this regression never imports search clients
    # or starts a search. Execute its real periodic-update branch with stubs.
    source = ROOT / "Evo_Textgrad_battery_eoh_deepseek/reflective_metric_search.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReflectiveMetricSearch")
    callback = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_reflection_callback")
    namespace = {"logging": logging}
    module = ast.Module(body=[callback], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    weights = Mock(return_value={"e1": 0.5, "e2": 0.5})
    memory = Mock()
    stub = SimpleNamespace(reflection_enabled=True, reflection_frequency=3,
                           adaptive_operator_selection=True,
                           generate_performance_reflection=lambda *args: (None, None),
                           update_gain_weighted_memory=memory,
                           adaptive_operator_selection_weights=weights)
    namespace["_reflection_callback"](stub, {}, [], "e1", 3)
    weights.assert_called_once_with()
    memory.assert_called_once_with()


def test_function_download_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gpt.py"
        path.write_text("# saved function\n", encoding="utf-8")
        for name in ("cddi_function.py", "gpt.py"):
            session = SimpleNamespace(artifacts=[{"name": name, "path": str(path)}])
            with patch.object(rest.store, "get", new=AsyncMock(return_value=session)):
                response = asyncio.run(rest.get_artifact("test-download", name))
            assert response.filename == "cddi_function.py"


def main() -> int:
    for test in (test_names_and_routing, test_new_and_saved_tables, test_periodic_reflection_callback, test_function_download_name):
        test()
        print(test.__name__ + ": PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
