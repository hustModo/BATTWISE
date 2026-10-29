import ReactECharts from "echarts-for-react";
import useSession from "@/store/sessionStore";

export default function MMDDistribution() {
  const hist = useSession((s) => s.charts.mmd_hist);
  const progress = useSession((s) => s.charts.mmd_progress);
  const correlation = useSession((s) => s.charts.mmd_correlation);

  const histogram = hist?.histogram ?? hist;
  const bins = (histogram?.bin_edges ?? histogram?.bins) as number[] | undefined;
  const counts = histogram?.counts as number[] | undefined;
  const mean_sig =
    (hist?.mean_sig ?? hist?.correlation?.mean_sig ?? histogram?.mean_sig) as
      | number
      | undefined;

  const histOption = bins && counts
    ? {
        grid: { left: 48, right: 18, top: 28, bottom: 50, containLabel: true },
        tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
        xAxis: {
          type: "category",
          data: bins.slice(0, -1).map((b, i) => `${b.toFixed(2)}-${bins[i + 1].toFixed(2)}`),
          axisLabel: { fontSize: 9, rotate: 35, hideOverlap: true },
          name: "CDDI",
          nameLocation: "middle",
          nameGap: 28,
          nameTextStyle: { fontSize: 11 },
        },
        yAxis: { type: "value", name: "count", nameTextStyle: { fontSize: 11 } },
        series: [
          {
            type: "bar",
            data: counts,
            itemStyle: { color: "#0EA5A4", borderRadius: [3, 3, 0, 0] },
            barWidth: "85%",
          },
        ],
      }
    : null;

  const corrRows = correlation?.rows as Array<{ name: string; r: number; p: number; significant: boolean }> | undefined;

  return (
    <div className="card p-4 flex flex-col gap-3 h-full min-h-0 overflow-hidden">
      <div className="flex items-center justify-between gap-2 flex-wrap shrink-0">
        <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
          CDDI Distribution
        </div>
        {progress && (
          <div className="text-[11px] text-slate-500">
            {progress.completed}/{progress.total} cells
            {progress.eta_sec !== undefined && ` · ETA ${Math.round(progress.eta_sec)}s`}
          </div>
        )}
        {mean_sig !== undefined && (
          <div className="text-[11px] px-2 py-0.5 rounded bg-battery-electrolyte/10 text-battery-electrolyte">
            mean_sig = {mean_sig.toFixed(3)}
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0">
        {histOption ? (
          <ReactECharts option={histOption} style={{ height: "100%", width: "100%" }} />
        ) : (
          <div className="h-full grid place-items-center text-slate-400 text-sm">
            Histogram will appear after compute finishes.
          </div>
        )}
      </div>

      {corrRows && corrRows.length > 0 && (
        <div className="overflow-y-auto max-h-32 scrollbar-thin border-t pt-2 shrink-0">
          <div className="text-[11px] text-slate-500 mb-1">
            Pearson correlation (|r|, p &lt; 0.1 = significant)
          </div>
          <table className="w-full text-[11px]">
            <thead className="text-slate-400">
              <tr>
                <th className="text-left font-normal">Indicator</th>
                <th className="text-right font-normal">|r|</th>
                <th className="text-right font-normal">p</th>
                <th className="text-center font-normal">sig</th>
              </tr>
            </thead>
            <tbody>
              {corrRows.slice(0, 12).map((r) => (
                <tr key={r.name}>
                  <td className="text-slate-700 truncate max-w-[10rem]">{r.name}</td>
                  <td className="text-right tabular-nums">{Math.abs(r.r).toFixed(3)}</td>
                  <td className="text-right tabular-nums">{r.p.toFixed(3)}</td>
                  <td className="text-center">
                    {r.significant ? <span className="text-battery-charge">✓</span> : "·"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
