import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import useSession from "@/store/sessionStore";
import { formatTime } from "@/utils/format";
import MessageContent from "./MessageContent";

const SUGGESTIONS = [
  "Design a new CDDI function for First-200s data.",
  "Compute CDDI across all cells now.",
  "Run clustering on the latest CDDI results.",
  "I'm interested in Cluster 1 — refine it.",
  "你好，今天这个系统状态怎么样？",
  "你能用简单的话解释一下现在做到哪一步了吗？",
  "CDDI 是用来干什么的？",
];

export default function ChatPanel() {
  const messages = useSession((s) => s.messages);
  const send = useSession((s) => s.sendMessage);
  const awaitingResponse = useSession((s) => s.awaitingResponse);
  const activeAgent = useSession((s) => s.activeAgent);
  const pipeline = useSession((s) => s.pipeline);
  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 1e6, behavior: "smooth" });
  }, [messages.length, awaitingResponse]);

  const onSend = async () => {
    const v = text.trim();
    if (!v) return;
    setText("");
    await send(v);
  };

  return (
    <div className="card flex flex-col h-full min-h-0 overflow-hidden">
      <div className="px-4 py-3 border-b flex items-center justify-between shrink-0">
        <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
          Conversation
        </div>
        <div className="text-[11px] text-slate-400">{messages.length} message(s)</div>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin">
        {messages.length === 0 && !awaitingResponse && (
          <div className="text-center text-slate-400 text-sm mt-6">
            You can chat naturally, ask questions, or instruct the next agent.
          </div>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            className={clsx(
              "flex flex-col min-w-0",
              m.role === "user" ? "items-end" : "items-start",
            )}
          >
            <div
              className={clsx(
                "max-w-[92%] px-3.5 py-2 rounded-2xl text-sm shadow-sm break-words",
                m.role === "user"
                  ? "bg-battery-electrolyte text-white rounded-br-sm whitespace-pre-wrap"
                  : "bg-slate-100 text-slate-800 rounded-bl-sm",
              )}
            >
              {m.role === "user" ? (
                m.text
              ) : (
                <>
                  <MessageContent text={m.text} />
                  {m.streaming && (
                    <span
                      className="inline-block w-1.5 h-3 ml-0.5 align-middle bg-battery-electrolyte/70 animate-pulse rounded-sm"
                      aria-hidden
                    />
                  )}
                </>
              )}
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              {m.agent ? `${m.agent} · ` : ""}
              {formatTime(m.t)}
              {m.streaming && m.role === "assistant" && (
                <span className="ml-1 text-battery-electrolyte/80">· streaming…</span>
              )}
            </div>
          </div>
        ))}
        {awaitingResponse && (
          <div className="flex flex-col items-start">
            <div className="max-w-[92%] px-3.5 py-2 rounded-2xl rounded-bl-sm text-sm shadow-sm bg-slate-100 text-slate-700">
              <span className="inline-flex items-center gap-1.5">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-battery-electrolyte animate-pulse" />
                {activeAgent
                  ? `${activeAgent} 正在处理...`
                  : pipeline.router === "running"
                    ? "正在判断你的意图..."
                    : "正在思考..."}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 mt-1">system</div>
          </div>
        )}
      </div>
      {messages.length === 0 && !awaitingResponse && (
        <div className="px-4 pb-2 flex flex-wrap gap-1.5">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setText(s)}
              className="text-[11px] px-2 py-1 rounded-md border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}
      <div className="border-t p-3 flex gap-2 shrink-0">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              onSend();
            }
          }}
          rows={2}
          placeholder="Ask anything, or instruct the next agent (⌘/Ctrl + ↩ to send)"
          className="flex-1 px-3 py-2 rounded-lg border border-slate-200 text-sm resize-none focus:border-battery-electrolyte focus:ring-2 focus:ring-battery-electrolyte/20 outline-none"
        />
        <button
          onClick={onSend}
          className="px-4 py-2 rounded-lg bg-battery-electrolyte text-white text-sm font-medium hover:shadow-card transition-shadow"
        >
          Send
        </button>
      </div>
    </div>
  );
}
