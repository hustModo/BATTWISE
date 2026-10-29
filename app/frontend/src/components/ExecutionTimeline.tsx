import useSession from "@/store/sessionStore";
import clsx from "clsx";
import { formatTime } from "@/utils/format";

const KIND_COLOR: Record<string, string> = {
  agent_started: "bg-battery-electrolyte",
  agent_finished: "bg-battery-charge",
  thinking: "bg-slate-400",
  log: "bg-slate-300",
  router_decision: "bg-agent-router",
  tool_call: "bg-agent-designer",
  code_update: "bg-agent-designer",
  chart_data: "bg-agent-compute",
  artifact: "bg-battery-amber",
  user_prompt_required: "bg-battery-danger",
};

export default function ExecutionTimeline() {
  const events = useSession((s) => s.timeline);

  return (
    // h-full so we always fill the Workbench's flex-1 slot. The inner
    // scrollbar (.scrollbar-thin, 9px) handles overflow when there are
    // more events than fit on screen.
    <div className="card overflow-hidden flex flex-col h-full">
      <div className="px-4 py-3 border-b flex items-center justify-between">
        <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
          Execution Timeline
        </div>
        <div className="text-[11px] text-slate-400">{events.length} event(s)</div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 scrollbar-thin">
        {events.length === 0 && (
          <div className="text-center text-slate-400 text-sm mt-6">No events yet.</div>
        )}
        <div className="relative pl-3">
          {events.slice().reverse().map((e) => (
            <div key={e.id} className="relative pl-4 py-1.5">
              <span
                className={clsx(
                  "absolute left-0 top-2.5 w-2 h-2 rounded-full",
                  KIND_COLOR[e.kind] ?? "bg-slate-400",
                )}
              />
              <div className="text-[11px] text-slate-400">
                {formatTime(e.t)} · {e.kind}
                {e.agent ? ` · ${e.agent}` : ""}
              </div>
              <div className="text-[13px] text-slate-700 break-words">{e.text}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
