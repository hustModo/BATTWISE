import ReactECharts from "echarts-for-react";
import { useMemo } from "react";
import useSession from "@/store/sessionStore";

// 12-color palette so k up to 12 each get a distinct hue (matches the
// reference figure: C1..C6 ≈ teal / mauve / green / red / blue / amber).
const PALETTE = [
  "#0EA5A4", "#8B5CF6", "#22C55E", "#EF4444", "#3B82F6", "#F59E0B",
  "#EC4899", "#14B8A6", "#A855F7", "#84CC16", "#F97316", "#06B6D4",
];

// Friendly axis labels (matches the paper figure the user shared).
const AXIS_UNITS: Record<string, string> = {
  ir: "IR (mΩ)",
  capacity: "Capacity (Ah)",
  ocv: "OCV (V)",
  mmd_loss: "CDDI Loss",
  cddi: "CDDI",
  ddia: "CDDI",
  ddi: "CDDI",
};

const axisLabel = (raw?: string) => {
  if (!raw) return "Component";
  return AXIS_UNITS[raw.toLowerCase()] ?? raw;
};

interface Props { scope?: "main" | "sub" }

export default function ClusterScatter({ scope = "main" }: Props) {
  const scatter = useSession((s) =>
    scope === "sub" ? s.subCharts.cluster_scatter : s.charts.cluster_scatter,
  );
  const iter = useSession((s) =>
    scope === "sub" ? s.subCharts.cluster_iter : s.charts.cluster_iter,
  );
  const select = useSession((s) => s.selectCluster);
  const selected = useSession((s) => s.session?.selected_cluster ?? null);

  const option = useMemo(() => {
    if (!scatter || !scatter.points) return null;
    // Backend (clustering_runner.summarize_clusters) emits:
    //   points: [{ cell_num, cluster, ocv, ir, capacity, mmd_loss }, ...]
    //   axes:   { x: "OCV", y: "IR" }
    // We normalize that into ECharts' [x, y, cell, cluster] tuples here.
    const xKey = (scatter.axes?.x ?? "OCV").toLowerCase();
    const yKey = (scatter.axes?.y ?? "IR").toLowerCase();
    const points: any[] = scatter.points;
    const groups = new Map<number, any[]>();
    points.forEach((p) => {
      const x = p[xKey] ?? p.x ?? p.ocv;
      const y = p[yKey] ?? p.y ?? p.ir;
      const cell = p.cell_num ?? p.cell;
      const cluster = p.cluster;
      if (x === null || y === null || x === undefined || y === undefined) return;
      const arr = groups.get(cluster) ?? [];
      arr.push([Number(x), Number(y), cell, cluster]);
      groups.set(cluster, arr);
    });
    if (groups.size === 0) return null;
    const series = Array.from(groups.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([cluster, data]) => ({
        name: `Cluster ${cluster}`,
        type: "scatter",
        data,
        symbolSize: (d: any) => (d[3] === selected ? 13 : 8),
        itemStyle: { color: PALETTE[cluster % PALETTE.length], opacity: selected === null || selected === cluster ? 0.95 : 0.25 },
        emphasis: { focus: "self" },
      }));
    const xLabel = axisLabel(scatter.axes?.x ?? scatter.x_label ?? "Component 1");
    const yLabel = axisLabel(scatter.axes?.y ?? scatter.y_label ?? "Component 2");
    return {
      grid: { left: 48, right: 18, top: 44, bottom: 42, containLabel: true },
      legend: { top: 4, textStyle: { fontSize: 11 } },
      tooltip: {
        trigger: "item",
        formatter: (p: any) =>
          `Cell #${p.data[2]}<br/>Cluster: ${p.data[3]}<br/>${xLabel}: ${Number(p.data[0]).toFixed(3)}<br/>${yLabel}: ${Number(p.data[1]).toFixed(3)}`,
      },
      xAxis: {
        type: "value",
        name: xLabel,
        nameLocation: "middle",
        nameGap: 26,
        nameTextStyle: { fontSize: 11 },
        scale: true,
      },
      yAxis: {
        type: "value",
        name: yLabel,
        nameLocation: "middle",
        nameGap: 36,
        nameTextStyle: { fontSize: 11 },
        scale: true,
      },
      series,
    };
  }, [scatter, selected]);

  return (
    <div className="card p-4 flex flex-col gap-2 h-full min-h-0 overflow-hidden">
      <div className="flex items-center justify-between gap-2 flex-wrap shrink-0">
        <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
          {scope === "sub" ? "Sub-cluster Scatter" : "Cluster Scatter"}
        </div>
        {scatter?.k !== undefined && (
          <div className="text-[11px] text-slate-500">
            K = {scatter.k} · DBI = {scatter.dbi?.toFixed?.(3) ?? "—"}
          </div>
        )}
        {iter && iter.length > 0 && (
          <div className="text-[11px] text-battery-electrolyte">
            iter {iter[iter.length - 1].iter}/{iter[iter.length - 1].max_iter ?? "?"}
          </div>
        )}
      </div>
      <div className="flex-1 min-h-0">
        {option ? (
          <ReactECharts
            option={option}
            style={{ height: "100%", width: "100%" }}
            onEvents={{
              click: (p: any) => {
                const cluster = Array.isArray(p?.data) ? p.data[3] : p?.data?.cluster;
                if (typeof cluster === "number") select(cluster);
              },
            }}
          />
        ) : (
          <div className="h-full grid place-items-center text-slate-400 text-sm">
            Scatter will appear after clustering finishes. Click a cluster to focus.
          </div>
        )}
      </div>
    </div>
  );
}
