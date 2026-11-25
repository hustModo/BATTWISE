import { useEffect, useState } from "react";
import clsx from "clsx";
import MMDCodeViewer from "@/components/charts/MMDCodeViewer";
import MMDDistribution from "@/components/charts/MMDDistribution";
import ClusterScatter from "@/components/charts/ClusterScatter";
import CapacityBoxplot from "@/components/charts/CapacityBoxplot";
import ClusterFingerprint from "@/components/charts/ClusterFingerprint";
import RecyclingDecisionCards from "@/components/charts/RecyclingDecisionCards";
import { ReflectiveSearchProgressChart } from "@/components/charts/ReflectiveSearchProgressChart";
import { ReflectiveSearchLogStream } from "@/components/charts/ReflectiveSearchLogStream";
import { LLMLogStream } from "@/components/charts/LLMLogStream";
import ClusterKControls from "@/components/charts/ClusterKControls";
import useSession from "@/store/sessionStore";

type TabId = "design" | "compute" | "cluster" | "decisions" | "refine";

const TABS: { id: TabId; label: string; node: string }[] = [
  { id: "design", label: "CDDI program discovery", node: "designer" },
  { id: "compute", label: "Compute", node: "compute" },
  { id: "cluster", label: "Cluster", node: "cluster" },
  { id: "decisions", label: "Decisions", node: "cluster" },
  { id: "refine", label: "Refine", node: "refine" },
];

export default function MainStage() {
  const [tab, setTab] = useState<TabId>("design");
  const pipeline = useSession((s) => s.pipeline);
  const hasSubScatter = useSession((s) => !!s.subCharts.cluster_scatter);
  const sessionMode = useSession((s) => s.session?.mode);
  const reflectiveSearchActive = useSession(
    (s) => s.reflectiveSearch.running || s.reflectiveSearch.history.length > 0,
  );
  const showReflectiveSearch =
    sessionMode === "reflective_search" ||
    sessionMode === "full_eoh" ||
    reflectiveSearchActive;

  useEffect(() => {
    if (hasSubScatter) setTab("refine");
  }, [hasSubScatter]);

  return (
    <div className="h-full min-h-0 flex flex-col overflow-hidden">
      <div className="flex items-center gap-2 mb-3 shrink-0 overflow-x-auto scrollbar-thin pb-1">
        {TABS.map((t) => {
          const state = pipeline[t.node as keyof typeof pipeline];
          const dot =
            state === "running"
              ? "bg-battery-electrolyte animate-pulse"
              : state === "done"
                ? "bg-battery-charge"
                : state === "error"
                  ? "bg-battery-danger"
                  : "bg-slate-300";
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={clsx(
                "px-3.5 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors",
                tab === t.id
                  ? "bg-battery-electrolyte text-white shadow-card"
                  : "bg-white text-slate-700 hover:bg-slate-100 border border-slate-200",
              )}
            >
              <span className={clsx("w-1.5 h-1.5 rounded-full", dot)} />
              {t.label}
            </button>
          );
        })}
      </div>

      <div className="flex-1 min-h-0 overflow-hidden">
        {tab === "design" && (
          <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4 h-full min-h-0 overflow-hidden">
            <MMDCodeViewer />
            {showReflectiveSearch ? (
              // Two-row right column when EOH is running: chart on top,
              // live log stream below. Both share equal height.
              <div className="grid grid-rows-2 gap-4 min-h-0 overflow-hidden">
                <ReflectiveSearchProgressChart />
                <ReflectiveSearchLogStream />
              </div>
            ) : (
              // Lightweight mode: top half shows CDDI distribution (after
              // Compute runs), bottom half streams the live agent log so the
              // user can see each LLM/sandbox/quality step instead of staring
              // at a static "running" toolcall.
              <div className="grid grid-rows-[minmax(220px,1fr)_minmax(180px,0.85fr)] gap-4 min-h-0 overflow-hidden">
                <MMDDistribution />
                <LLMLogStream />
              </div>
            )}
          </div>
        )}
        {tab === "compute" && (
          // During Compute we want the live log VERY prominent — that's where
          // the user feels "stuck" the most. Distribution sits on top, log on
          // bottom; both half-height so the heartbeat is always on screen.
          <div className="grid grid-rows-[minmax(260px,1fr)_minmax(180px,0.75fr)] gap-4 h-full min-h-0 overflow-hidden">
            <MMDDistribution />
            <LLMLogStream />
          </div>
        )}
        {tab === "cluster" && (
          <div className="flex flex-col gap-3 h-full min-h-0 overflow-hidden">
            <ClusterKControls />
            <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4 flex-1 min-h-0 overflow-hidden">
              <ClusterScatter />
              <div className="grid grid-rows-2 gap-4 min-h-0 overflow-hidden">
                <CapacityBoxplot />
                <ClusterFingerprint />
              </div>
            </div>
          </div>
        )}
        {tab === "decisions" && (
          <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4 h-full min-h-0 overflow-hidden">
            <RecyclingDecisionCards />
            <ClusterFingerprint />
          </div>
        )}
        {tab === "refine" && (
          <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4 h-full min-h-0 overflow-hidden">
            <div className="grid grid-rows-2 gap-4 min-h-0 overflow-hidden">
              <ClusterScatter scope="sub" />
              <CapacityBoxplot scope="sub" />
            </div>
            <div className="grid grid-rows-2 gap-4 min-h-0 overflow-hidden">
              <ClusterFingerprint scope="sub" />
              <RecyclingDecisionCards scope="sub" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
