import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import useSession from "@/store/sessionStore";

type Tag = "info" | "ok" | "inf" | "best" | "warn" | "error";

const TAG_STYLE: Record<Tag, { className: string; symbol: string }> = {
  info:  { className: "text-slate-300",  symbol: "·" },
  ok:    { className: "text-emerald-300", symbol: "✓" },
  inf:   { className: "text-amber-300",  symbol: "∞" },
  best:  { className: "text-cyan-300 font-semibold", symbol: "★" },
  warn:  { className: "text-yellow-300", symbol: "!" },
  error: { className: "text-red-300",    symbol: "✗" },
};

function shortLine(s: string): string {
  // strip the leading `[YYYY-MM-DD HH:MM:SS,mmm][module][LEVEL] - ` prefix
  // produced by Python's logging.basicConfig — we'll re-add a compact HH:MM:SS.
  const m = s.match(/^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})[,\.]\d+\]\[[^\]]+\]\[[^\]]+\]\s*-?\s*(.*)$/);
  if (m) {
    const hhmmss = m[1].slice(-8);
    return `${hhmmss} · ${m[2]}`;
  }
  return s;
}

export function ReflectiveSearchLogStream() {
  const search = useSession((s: any) => s.reflectiveSearch);
  const logs = (search.logs ?? []) as Array<{ t: number; line: string; tag: Tag }>;
  const candidates = (search.candidates ?? {}) as Record<
    number,
    { generation: number; ok: number; inf: number; total: number }
  >;

  const [autoscroll, setAutoscroll] = useState(true);
  const [filter, setFilter] = useState<"all" | "best" | "no_http">("no_http");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const filtered = useMemo(() => {
    if (filter === "best") return logs.filter((l) => l.tag === "best" || l.tag === "inf");
    if (filter === "no_http") return logs; // already pre-filtered server-side
    return logs;
  }, [logs, filter]);

  useEffect(() => {
    if (!autoscroll || !scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [filtered.length, autoscroll]);

  const summary = useMemo(() => {
    const gens = Object.values(candidates).sort((a, b) => a.generation - b.generation);
    const latest = gens[gens.length - 1];
    const totals = gens.reduce(
      (acc, g) => ({ ok: acc.ok + g.ok, inf: acc.inf + g.inf, total: acc.total + g.total }),
      { ok: 0, inf: 0, total: 0 },
    );
    return { latest, totals, count: gens.length };
  }, [candidates]);

  return (
    <div className="card flex flex-col gap-2 h-full min-h-[280px]">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-slate-100">Reflective-search live log</h3>
          <span className="text-xs text-slate-400">
            {logs.length}/500 lines
          </span>
          {summary.latest && (
            <span className="text-xs text-slate-400">
              · gen {summary.latest.generation}: {summary.latest.ok} ok ·{" "}
              <span className="text-amber-300">{summary.latest.inf} inf</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="bg-slate-800 text-slate-200 border border-slate-700 rounded px-1.5 py-0.5"
          >
            <option value="no_http">algorithm only</option>
            <option value="best">best / inf only</option>
            <option value="all">everything</option>
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
        className="flex-1 overflow-y-auto rounded-md bg-slate-950/70 border border-slate-800 px-2 py-1.5 font-mono text-[11px] leading-[1.45]"
      >
        {filtered.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500">
            Waiting for reflective-search output…
          </div>
        ) : (
          filtered.map((l, i) => {
            const s = TAG_STYLE[l.tag] ?? TAG_STYLE.info;
            return (
              <div key={i} className={clsx("whitespace-pre-wrap", s.className)}>
                <span className="mr-1 opacity-70">{s.symbol}</span>
                {shortLine(l.line)}
              </div>
            );
          })
        )}
      </div>

      {summary.count > 0 && (
        <div className="text-xs text-slate-500">
          totals: {summary.totals.ok} ok · {summary.totals.inf} inf over{" "}
          {summary.count} generation(s) — `inf` means LLM-written code threw an
          exception (e.g. wrong array shape). Evolution will weed them out.
        </div>
      )}
    </div>
  );
}
