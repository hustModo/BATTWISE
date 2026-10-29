import { useState } from "react";
import clsx from "clsx";
import useSession from "@/store/sessionStore";

/**
 * Small "AI is tracking" card pinned above the chat panel.
 *
 * Displays the running summary built by services/conversation_memory on
 * the backend. The card collapses by default (single-line tagline) and
 * expands on click to show the full memo. Hidden when no summary exists.
 *
 * The point is for the user to *see* what the assistant remembers across
 * turns — otherwise the LLM router's "memory" is invisible and feels like
 * magic that might be wrong.
 */
export default function ConversationSummary() {
  const summary = useSession((s) => s.conversationSummary);
  const [expanded, setExpanded] = useState(false);

  if (!summary) return null;

  const want = summary.what_user_wants?.trim() || "(no goal recorded yet)";
  const state = summary.current_state?.trim();
  const done = summary.what_was_done?.filter(Boolean) ?? [];
  const prefs = Object.entries(summary.user_preferences || {}).filter(
    ([, v]) => v !== null && v !== undefined && String(v).length > 0,
  );
  const open = summary.open_questions?.filter(Boolean) ?? [];

  return (
    <div
      className={clsx(
        "card overflow-hidden shrink-0 border-l-2 border-l-battery-accent/70",
        "text-[12px] leading-snug",
      )}
    >
      <button
        type="button"
        className="w-full px-3 py-2 flex items-start gap-2 hover:bg-slate-50/60 transition-colors"
        onClick={() => setExpanded((v) => !v)}
        title={expanded ? "Collapse" : "Expand"}
      >
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-battery-accent mt-1 shrink-0" />
        <div className="flex-1 min-w-0 text-left">
          <div className="text-[10px] uppercase tracking-widest text-slate-400 font-medium">
            AI is tracking
          </div>
          <div className="text-slate-700 truncate font-medium">{want}</div>
        </div>
        <span className="text-slate-400 text-xs shrink-0">{expanded ? "▾" : "▸"}</span>
      </button>

      {expanded && (
        <div className="px-3 pb-3 pt-1 space-y-2 border-t border-slate-100">
          {state && (
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-400">
                Current state
              </div>
              <div className="text-slate-600">{state}</div>
            </div>
          )}
          {done.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-400">
                Done so far
              </div>
              <ul className="list-disc pl-4 text-slate-600 space-y-0.5">
                {done.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>
          )}
          {prefs.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-400">
                Preferences
              </div>
              <div className="flex flex-wrap gap-1">
                {prefs.map(([k, v]) => (
                  <span
                    key={k}
                    className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 text-[11px] font-mono"
                  >
                    {k}={String(v)}
                  </span>
                ))}
              </div>
            </div>
          )}
          {open.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-400">
                Open questions
              </div>
              <ul className="list-disc pl-4 text-slate-600 space-y-0.5">
                {open.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
