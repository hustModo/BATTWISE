import ReactECharts from "echarts-for-react";
import { useMemo } from "react";
import useSession from "@/store/sessionStore";

// Same cluster-id palette as ClusterScatter / CapacityBoxplot. Always index
// by the numeric cluster id, not by series order, so colors stay stable even
// if a subset only contains clusters 0/3/5.
const PALETTE = [
  "#0EA5A4", "#8B5CF6", "#22C55E", "#EF4444", "#3B82F6", "#F59E0B",
  "#EC4899", "#14B8A6", "#A855F7", "#84CC16", "#F97316", "#06B6D4",
];

interface Props { scope?: "main" | "sub" }

export default function ClusterFingerprint({ scope = "main" }: Props) {
  const fp = useSession((s) =>
    scope === "sub" ? s.subCharts.fingerprint : s.charts.fingerprint,
  );
  const selected = useSession((s) => s.session?.selected_cluster ?? null);

  const option = useMemo(() => {
    if (!fp || !fp.clusters || fp.clusters.length === 0) return null;
    // Backend emits:
    //   { features: ["Urest", "UCC,init", ...], clusters: [{cluster, values:[...]}] }
    // ``values`` are min-max normalized to [0, 1] per feature, so we use max=1.
    const names: string[] =
      (fp.features as string[]) ??
      (fp.indicators as Array<{ name: string }>)?.map((i) => i.name) ??
      [];
    if (names.length === 0) return null;
    const clusters = fp.clusters as Array<{ cluster: number; values: Array<number | null> }>;
    return {
      tooltip: { trigger: "item" },
      legend: { top: 4, textStyle: { fontSize: 11 } },
      radar: {
        indicator: names.map((n) => ({
          name: /^(mmd_loss|ddia|ddi|cddi)$/i.test(n) ? "CDDI" : n,
          max: 1.0,
        })),
        radius: "58%",
        center: ["50%", "54%"],
        splitArea: { areaStyle: { color: ["rgba(248,250,252,0.7)", "rgba(241,245,249,0.6)"] } },
        axisName: { fontSize: 10, color: "#475569" },
      },
      series: [
        {
          type: "radar",
          data: clusters.map((c) => ({
            name: `Cluster ${c.cluster}`,
            value: c.values.map((v) => (v == null || Number.isNaN(v) ? 0 : v)),
            itemStyle: { color: PALETTE[c.cluster % PALETTE.length] },
            lineStyle: { width: c.cluster === selected ? 3 : 1.5 },
            areaStyle: { opacity: c.cluster === selected ? 0.35 : selected === null ? 0.18 : 0.05 },
          })),
        },
      ],
    };
  }, [fp, selected]);

  const featureKind = (fp as { feature_kind?: string } | null | undefined)?.feature_kind;
  const curveOnly = featureKind === "curve_only";
  const clusterFeatures = featureKind === "cluster_features";

  return (
    <div className="card p-4 flex flex-col gap-2 h-full min-h-0 overflow-hidden">
      <div className="flex items-center justify-between gap-2 shrink-0">
        <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
          {scope === "sub" ? "Sub-cluster Feature Fingerprint" : "Cluster-Feature Fingerprint"}
        </div>
        {clusterFeatures && (
          <div
            className="text-[10px] text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-1.5 py-0.5"
            title="Radar uses the actual clustering features: OCV, IR, and CDDI. Capacity is excluded from clustering."
          >
            OCV / IR / CDDI
          </div>
        )}
        {curveOnly && (
          <div
            className="text-[10px] text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-1.5 py-0.5"
            title="Radar uses curve-derived features only (ICA, CV, gradient, capacity-voltage); electrochemical-only inputs (OCV/IR/EIS fits) are excluded because they're not measurable in production."
          >
            curve features only
          </div>
        )}
      </div>
      <div className="flex-1 min-h-0">
        {option ? (
          <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />
        ) : (
          <div className="h-full grid place-items-center text-slate-400 text-sm">
            Feature fingerprint appears after clustering.
          </div>
        )}
      </div>
    </div>
  );
}
