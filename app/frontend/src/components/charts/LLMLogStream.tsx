import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import useSession from "@/store/sessionStore";

type Tag = "info" | "ok" | "warn" | "error";

const TAG_STYLE: Record<Tag, { className: string; symbol: string }> = {
  info:  { className: "text-slate-200",   symbol: "·" },
  ok:    { className: "text-emerald-300", symbol: "✓" },
  warn:  { className: "text-amber-300",   symbol: "!" },
  error: { className: "text-red-300",     symbol: "✗" },
};

const AGENT_COLOR: Record<string, string> = {
  designer: "bg-violet-500/30 text-violet-200 border-violet-500/30",
  compute:  "bg-sky-500/30 text-sky-200 border-sky-500/30",
  cluster:  "bg-emerald-500/30 text-emerald-200 border-emerald-500/30",
  refine:   "bg-amber-500/30 text-amber-200 border-amber-500/30",
  router:   "bg-rose-500/30 text-rose-200 border-rose-500/30",
};

function fmtTime(t: number): string {
  const d = new Date(t * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

/**
 * Shared "live agent terminal" used in Lightweight mode (and visible in all
 * modes). Listens to `chart_data(llm_log)` events from Designer / Compute /
 * Cluster — each line carries an `agent`, a `tag` (info|ok|warn|error) and
 * the message itself. Renders them inline like a tail -f mini terminal that
 * the user can scroll, filter by agent, and pause autoscroll on.
 *
 * Why this exists: the Execution Timeline mixes EVERY event kind (router
 * decisions, tool_call updates, chart_data, artifact, ...). It's great for
 * audit but bad for "is the system still alive?". This stream shows JUST the
 * heartbeat of the running agent (LLM call durations, sandbox progress,
 * per-cell MMD ticks, GA-KMeans iteration ticks).
 */
export function LLMLogStream() {
  const logs = useSession((s) => s.llmLogs ?? []);

  const [autoscroll, setAutoscroll] = useState(true);
  const [agentFilter, setAgentFilter] = useState<"all" | "designer" | "compute" | "cluster">("all");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const filtered = useMemo(() => {
    if (agentFilter === "all") return logs;
    return logs.filter((l) => l.agent === agentFilter);
  }, [logs, agentFilter]);

  useEffect(() => {
    if (!autoscroll || !scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [filtered.length, autoscroll]);

  // Quick counts to expose at the header — handy when filter is off.
  const counts = useMemo(() => {
    const c = { designer: 0, compute: 0, cluster: 0, refine: 0, errors: 0 };
    for (const l of logs) {
      if (l.agent in c) (c as any)[l.agent] += 1;
      if (l.tag === "error") c.errors += 1;
    }
    return c;
  }, [logs]);

  return (
    <div className="card p-3 flex flex-col gap-2 h-full min-h-0 overflow-hidden bg-slate-900 border-slate-800">
      <div className="flex items-center justify-between gap-2 flex-wrap shrink-0">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-slate-100">Agent runtime log</h3>
          <span className="text-xs text-slate-400">{logs.length}/400 lines</span>
          {counts.designer > 0 && (
            <span className="text-[11px] text-violet-300">D:{counts.designer}</span>
          )}
          {counts.compute > 0 && (
            <span className="text-[11px] text-sky-300">C:{counts.compute}</span>
          )}
          {counts.cluster > 0 && (
            <span className="text-[11px] text-emerald-300">K:{counts.cluster}</span>
          )}
          {counts.errors > 0 && (
            <span className="text-[11px] text-red-300">⚠ {counts.errors} error(s)</span>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <select
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value as any)}
            className="bg-slate-800 text-slate-200 border border-slate-700 rounded px-1.5 py-0.5"
          >
            <option value="all">all agents</option>
            <option value="designer">designer</option>
            <option value="compute">compute</option>
            <option value="cluster">cluster</option>
          </select>
          <label className="inline-flex items-center gap-1 text-slate-400 select-none cursor-pointer">
            <input
              type="checkbox"
              checked={autoscroll}
              onChange={(e) => setAutoscroll(e.target.checked)}
              className="accent-emerald-500"
            />
            autoscroll
          </label>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto scrollbar-thin rounded-md bg-slate-950/70 border border-slate-800 px-2 py-1.5 font-mono text-[11px] leading-[1.5]"
      >
        {filtered.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500 text-center px-4">
            Waiting for agent activity…
            <br />
            (Designer / Compute / Cluster will stream here when they run.)
          </div>
        ) : (
          filtered.map((l, i) => {
            const s = TAG_STYLE[l.tag] ?? TAG_STYLE.info;
            const badge = AGENT_COLOR[l.agent] ?? "bg-slate-600/40 text-slate-300 border-slate-600/30";
            return (
              <div key={i} className={clsx("whitespace-pre-wrap flex gap-1.5 min-w-0", s.className)}>
                <span className="text-slate-500 shrink-0">{fmtTime(l.t)}</span>
                <span
                  className={clsx(
                    "shrink-0 inline-block min-w-[60px] text-center rounded border px-1 text-[10px] font-medium uppercase tracking-wider",
                    badge,
                  )}
                >
                  {l.agent}
                </span>
                <span className="opacity-70 shrink-0">{s.symbol}</span>
                <span className="break-words min-w-0">{l.line}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
