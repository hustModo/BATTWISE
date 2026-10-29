import { useEffect } from "react";
import TopBar from "@/components/TopBar";
import AgentPipeline from "@/components/AgentPipeline";
import ChatPanel from "@/components/ChatPanel";
import ConversationSummary from "@/components/ConversationSummary";
import ExecutionTimeline from "@/components/ExecutionTimeline";
import PlanPanel from "@/components/PlanPanel";
import ToolCalls from "@/components/ToolCalls";
import MainStage from "@/components/MainStage";
import RefinementPanel from "@/components/RefinementPanel";
import ArtifactsPanel from "@/components/ArtifactsPanel";
import useSession from "@/store/sessionStore";

export default function Workbench() {
  const sessionId = useSession((s) => s.sessionId);
  const startSession = useSession((s) => s.startSession);
  const connected = useSession((s) => s.connected);

  useEffect(() => {
    if (!sessionId) startSession();
  }, [sessionId, startSession]);

  return (
    <div className="h-screen min-w-0 flex flex-col bg-battery-bg overflow-hidden">
      <TopBar />
      {/* Disconnect banner — visible when the WS dropped (backend
          crashed or restarted). The TopBar already has a small "offline"
          pill but users said they missed it; a full-width red strip is
          impossible to ignore. */}
      {sessionId && !connected && (
        <div className="bg-battery-danger/15 border-y border-battery-danger/30 text-battery-danger text-xs px-4 py-1.5 flex items-center justify-center gap-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-battery-danger animate-pulse" />
          <span>
            <b>Backend disconnected.</b> Reconnecting… If this persists, the
            backend may have crashed — run{" "}
            <code className="px-1.5 py-0.5 bg-white/60 rounded font-mono">
              deploy/kill_backend.sh
            </code>{" "}
            then{" "}
            <code className="px-1.5 py-0.5 bg-white/60 rounded font-mono">
              deploy/run_dev.sh backend
            </code>
            .
          </span>
        </div>
      )}
      <div className="px-5 pt-4">
        <AgentPipeline />
      </div>
      <div className="flex-1 min-h-0 grid grid-cols-1 xl:grid-cols-[minmax(280px,0.95fr)_minmax(520px,2fr)_minmax(300px,1fr)] gap-4 p-4 overflow-hidden">
        <aside className="min-w-0 min-h-0 flex flex-col gap-3 overflow-hidden">
          {/* Smart-routing context cards. Both auto-hide when there's no
              data — so a fresh session shows nothing until the user has
              actually asked for something. */}
          <PlanPanel />
          <ConversationSummary />
          <div className="flex-1 min-h-0">
            <ChatPanel />
          </div>
        </aside>
        <section className="min-w-0 min-h-0 overflow-hidden">
          <MainStage />
        </section>
        {/*
          Right rail layout: Timeline expands to fill remaining vertical
          space (and scrolls INTERNALLY when events exceed its height).
          Artifacts and ToolCalls have fixed compact heights so they don't
          eat Timeline's space. Each panel manages its own internal scroll
          — no outer-rail scroll (user feedback: "滑动做错了" — nested
          scroll was confusing).
        */}
        <aside className="min-w-0 min-h-0 flex flex-col gap-3 overflow-hidden">
          <div className="flex-1 min-h-0">
            <ExecutionTimeline />
          </div>
          <div className="h-[140px] shrink-0">
            <ArtifactsPanel />
          </div>
          <div className="h-[170px] shrink-0">
            <ToolCalls />
          </div>
        </aside>
      </div>
      <RefinementPanel />
    </div>
  );
}
