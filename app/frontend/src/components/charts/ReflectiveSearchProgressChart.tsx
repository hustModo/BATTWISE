import ReactECharts from "echarts-for-react";
import { motion } from "framer-motion";
import clsx from "clsx";
import useSession from "@/store/sessionStore";

export function ReflectiveSearchProgressChart() {
  const search = useSession((s: any) => s.reflectiveSearch);
  const cancelReflectiveSearch = useSession(
    (s: any) => s.cancelReflectiveSearch,
  );

  const points = search.history.filter((p: any) => p.objective !== null);
  const xs = points.map((p: any) => p.generation as number);
  const ys = points.map((p: any) => p.objective as number);

  const option = {
    backgroundColor: "transparent",
    grid: { left: 56, right: 16, top: 28, bottom: 36 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(15, 23, 42, 0.92)",
      borderWidth: 0,
      textStyle: { color: "#e2e8f0" },
    },
    xAxis: {
      type: "value",
      name: "generation",
      min: 0,
      max: search.total_generations ?? "dataMax",
      axisLabel: { color: "#94a3b8" },
      axisLine: { lineStyle: { color: "#334155" } },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "value",
      name: "objective",
      axisLabel: { color: "#94a3b8" },
      axisLine: { lineStyle: { color: "#334155" } },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        name: "best objective",
        type: "line",
        data: xs.map((g: number, i: number) => [g, ys[i]]),
        smooth: 0.15,
        showSymbol: true,
        symbol: "circle",
        symbolSize: 8,
        itemStyle: { color: "#34d399" },
        lineStyle: { color: "#34d399", width: 2 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(52, 211, 153, 0.35)" },
              { offset: 1, color: "rgba(52, 211, 153, 0.0)" },
            ],
          },
        },
      },
    ],
  };

  const progress =
    search.total_generations && search.total_generations > 0
      ? Math.min(100, Math.round((search.current_generation / search.total_generations) * 100))
      : null;

  return (
    <div className="card flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span
            className={clsx(
              "inline-flex h-2 w-2 rounded-full",
              search.running ? "bg-emerald-400 animate-pulse" : "bg-slate-500",
            )}
          />
          <h3 className="text-sm font-semibold text-slate-800">
            CDDI program discovery
          </h3>
          <span className="text-xs text-slate-400">
            gen {search.current_generation}
            {search.total_generations !== null ? ` / ${search.total_generations}` : ""}
            {search.best_objective !== null
              ? `  ·  best ${search.best_objective.toFixed(4)}`
              : ""}
          </span>
        </div>
        {search.running && (
          <motion.button
            whileTap={{ scale: 0.96 }}
            onClick={() => void cancelReflectiveSearch()}
            className="px-3 py-1 text-xs rounded-md bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
          >
            Cancel
          </motion.button>
        )}
      </div>

      {progress !== null && (
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full bg-emerald-400/80 transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      <div className="flex-1 min-h-[240px]">
        {points.length === 0 ? (
          <div className="h-full flex items-center justify-center text-sm text-slate-500">
            {search.running
              ? "Waiting for first generation to finish…"
              : "No reflective-search run recorded yet."}
          </div>
        ) : (
          <ReactECharts
            option={option}
            style={{ height: "100%", width: "100%" }}
            opts={{ renderer: "canvas" }}
            notMerge
            lazyUpdate
          />
        )}
      </div>

      {search.last_iter_eval && (
        <div className="text-xs text-slate-500">
          last eval: iter {search.last_iter_eval.iter} · resp #
          {search.last_iter_eval.response_id}
          {search.last_iter_eval.objective !== null && Number.isFinite(search.last_iter_eval.objective)
            ? `  ·  obj=${search.last_iter_eval.objective.toFixed(4)}`
            : ""}
        </div>
      )}
    </div>
  );
}
