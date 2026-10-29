import ReactECharts from "echarts-for-react";
import { useMemo } from "react";
import useSession from "@/store/sessionStore";

// Same palette as ClusterScatter so the same cluster shares a hue across charts.
const PALETTE = [
  "#0EA5A4", "#8B5CF6", "#22C55E", "#EF4444", "#3B82F6", "#F59E0B",
  "#EC4899", "#14B8A6", "#A855F7", "#84CC16", "#F97316", "#06B6D4",
];

interface Props { scope?: "main" | "sub" }

export default function CapacityBoxplot({ scope = "main" }: Props) {
  const box = useSession((s) =>
    scope === "sub" ? s.subCharts.capacity_box : s.charts.capacity_box,
  );

  const option = useMemo(() => {
    if (!box || !box.clusters || box.clusters.length === 0) return null;
    // Backend emits raw arrays per cluster: { cluster, values: number[] }.
    // We compute [min, Q1, median, Q3, max] here in the browser so the
    // boxplot stays a pure-data contract on the wire.
    const quantile = (sorted: number[], q: number): number => {
      if (sorted.length === 0) return NaN;
      if (sorted.length === 1) return sorted[0];
      const pos = (sorted.length - 1) * q;
      const lo = Math.floor(pos);
      const hi = Math.ceil(pos);
      if (lo === hi) return sorted[lo];
      return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    };

    const labels: string[] = [];
    const clusterIds: number[] = [];
    const data: number[][] = [];
    const counts: number[] = [];
    let outliers: Array<[number, number]> = [];
    box.clusters.forEach((c: any) => {
      const raw = (c.values ?? []).filter((v: any) => Number.isFinite(v)) as number[];
      if (raw.length === 0) return;
      const sorted = [...raw].sort((a, b) => a - b);
      const q1 = quantile(sorted, 0.25);
      const med = quantile(sorted, 0.5);
      const q3 = quantile(sorted, 0.75);
      const iqr = q3 - q1;
      const lowFence = q1 - 1.5 * iqr;
      const highFence = q3 + 1.5 * iqr;
      const whiskerLo = Math.max(sorted[0], lowFence);
      const whiskerHi = Math.min(sorted[sorted.length - 1], highFence);
      labels.push(`C${c.cluster}`);
      clusterIds.push(Number(c.cluster));
      data.push([whiskerLo, q1, med, q3, whiskerHi]);
      counts.push(raw.length);
      const boxIndex = data.length - 1;
      sorted
        .filter((v) => v < lowFence || v > highFence)
        .forEach((v) => outliers.push([boxIndex, v]));
    });
    if (data.length === 0) return null;
    // Per-cluster colors so this view ties visually back to the scatter.
    const styledBoxData = data.map((row, i) => ({
      value: row,
      itemStyle: {
        borderColor: PALETTE[clusterIds[i] % PALETTE.length],
        color: PALETTE[clusterIds[i] % PALETTE.length] + "33", // ~20% alpha fill
        borderWidth: 1.4,
      },
    }));
    return {
      grid: { left: 50, right: 18, top: 28, bottom: 42, containLabel: true },
      tooltip: {
        trigger: "item",
        formatter: (p: any) => {
          if (p.seriesType === "boxplot") {
            const d = p.data?.value ?? p.data;
            const cluster = labels[p.dataIndex];
            const n = counts[p.dataIndex];
            return `${cluster} (n=${n})<br/>whisker-lo: ${d[1]?.toFixed?.(3)}<br/>Q1: ${d[2]?.toFixed?.(3)}<br/>median: ${d[3]?.toFixed?.(3)}<br/>Q3: ${d[4]?.toFixed?.(3)}<br/>whisker-hi: ${d[5]?.toFixed?.(3)}`;
          }
          return p.value;
        },
      },
      xAxis: {
        type: "category",
        data: labels,
        name: "Cluster ID",
        nameLocation: "middle",
        nameGap: 28,
        nameTextStyle: { fontSize: 11 },
        axisLabel: { fontSize: 11 },
      },
      yAxis: {
        type: "value",
        name: (box.metric ?? "Capacity") === "Capacity" ? "Capacity (Ah)" : box.metric,
        nameLocation: "middle",
        nameGap: 38,
        nameTextStyle: { fontSize: 11 },
        scale: true,
      },
      series: [
        {
          type: "boxplot",
          data: styledBoxData,
        },
        outliers.length > 0
          ? {
              type: "scatter",
              data: outliers,
              symbolSize: 5,
              itemStyle: { color: "#EF4444", opacity: 0.7 },
            }
          : undefined,
      ].filter(Boolean),
    };
  }, [box]);

  return (
    <div className="card p-4 flex flex-col gap-2 h-full min-h-0 overflow-hidden">
      <div className="text-xs uppercase tracking-widest text-slate-500 font-medium shrink-0">
        {scope === "sub" ? "Sub-cluster" : "Per-cluster"} {box?.metric ?? "Capacity"} Boxplot
      </div>
      <div className="flex-1 min-h-0">
        {option ? (
          <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />
        ) : (
          <div className="h-full grid place-items-center text-slate-400 text-sm">
            Boxplot appears after clustering.
          </div>
        )}
      </div>
    </div>
  );
}
