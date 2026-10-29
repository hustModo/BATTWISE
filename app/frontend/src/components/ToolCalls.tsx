import useSession from "@/store/sessionStore";
import clsx from "clsx";
import { formatDuration, formatTime } from "@/utils/format";

const STATUS_COLOR: Record<string, string> = {
  running: "bg-battery-electrolyte/15 text-battery-electrolyte",
  ok: "bg-battery-charge/15 text-battery-charge",
  error: "bg-battery-danger/15 text-battery-danger",
};

export default function ToolCalls() {
  const calls = useSession((s) => s.toolCalls);

  return (
    // h-full: the Workbench gives this component a fixed-height (170 px)
    // slot. Without h-full the card sizes to its content and would overflow
    // (overlapping ArtifactsPanel above and the chat panel layout).
    <div className="card overflow-hidden flex flex-col h-full">
      <div className="px-4 py-3 border-b flex items-center justify-between">
        <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
          Tool Calls
        </div>
        <div className="text-[11px] text-slate-400">{calls.length}</div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-thin">
        {calls.length === 0 && (
          <div className="text-center text-slate-400 text-sm mt-6">
            Tool calls and their results will appear here.
          </div>
        )}
        {calls.slice().reverse().map((c, i) => (
          <div
            key={`${c.call_id ?? c.t}-${i}`}
            className="rounded-lg border border-slate-200 px-3 py-2 hover:bg-slate-50 transition-colors"
          >
            <div className="flex items-center justify-between">
              <div className="font-medium text-sm text-slate-800">{c.tool}</div>
              <div
                className={clsx(
                  "px-2 py-0.5 rounded text-[10px] uppercase tracking-wide font-semibold",
                  STATUS_COLOR[c.status] ?? "bg-slate-100 text-slate-600",
                )}
              >
                {c.status}
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex justify-between mt-0.5">
              <span>
                {c.agent ?? "—"} · {formatTime(c.t)}
              </span>
              <span>{formatDuration(c.duration_ms)}</span>
            </div>
            {c.args && Object.keys(c.args).length > 0 && (
              <details className="mt-1 text-[11px] text-slate-500">
                <summary className="cursor-pointer select-none">args</summary>
                <pre className="bg-slate-50 p-2 mt-1 rounded text-[10px] overflow-x-auto">
                  {JSON.stringify(c.args, null, 2)}
                </pre>
              </details>
            )}
            {c.result !== undefined && c.result !== null && (
              <details className="mt-1 text-[11px] text-slate-500">
                <summary className="cursor-pointer select-none">result</summary>
                <pre className="bg-slate-50 p-2 mt-1 rounded text-[10px] overflow-x-auto max-h-40">
                  {typeof c.result === "string"
                    ? c.result
                    : JSON.stringify(c.result, null, 2)}
                </pre>
              </details>
            )}
            {c.error && (
              <div className="mt-1 text-[11px] text-battery-danger">{c.error}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
