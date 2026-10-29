import { create } from "zustand";
import { api } from "@/api/rest";
import { SessionSocket } from "@/api/ws";
import type {
  AgentId,
  ArtifactItem,
  ChatMessage,
  ChartDataEvent,
  CodeUpdateEvent,
  ConversationSummary,
  ConversationSummaryEvent,
  PipelineState,
  PlanCreatedEvent,
  PlanUpdatedEvent,
  RouterDecisionEvent,
  SessionState,
  ToolCallEvent,
  UserPromptRequiredEvent,
  WorkflowPlan,
  WSEvent,
} from "@/types";

const EMPTY_PIPELINE: PipelineState = {
  router: "idle",
  designer: "idle",
  compute: "idle",
  cluster: "idle",
  refine: "idle",
};

const normalizeSessionMode = (mode: SessionState["mode"]): SessionState["mode"] =>
  mode === "full_eoh" ? "reflective_search" : mode;

const normalizeSessionState = (state: SessionState): SessionState => ({
  ...state,
  mode: normalizeSessionMode(state.mode),
});

interface TimelineEntry {
  id: string;
  t: number;
  kind: string;
  agent: AgentId | null;
  text: string;
  payload?: any;
}

interface ToolCallCard {
  call_id?: string;
  tool: string;
  agent: AgentId | null;
  status: "running" | "ok" | "error";
  duration_ms?: number;
  args?: any;
  result?: any;
  error?: string | null;
  t: number;
}

interface ChartBag {
  mmd_progress?: any;
  mmd_hist?: any;
  mmd_correlation?: any;
  cluster_iter: Array<any>;
  cluster_scatter?: any;
  capacity_box?: any;
  fingerprint?: any;
  cluster_summary?: any;
  eoh_progress?: any;
}

interface SubChartBag {
  mmd_progress?: any;
  cluster_iter: Array<any>;
  cluster_scatter?: any;
  capacity_box?: any;
  fingerprint?: any;
  cluster_summary?: any;
}

interface ReflectiveSearchProgress {
  generation: number;
  objective: number | null;
  total?: number;
}

interface ReflectiveSearchLogLine {
  t: number;
  line: string;
  tag: "info" | "ok" | "inf" | "best" | "warn" | "error";
}

interface ReflectiveSearchCandidateStat {
  generation: number;
  ok: number;
  inf: number;
  total: number;
}

// Shared "runtime log" stream populated by ALL agents (Designer / Compute /
// Cluster). Backend emits `chart_data(llm_log, { line, tag, agent })`. The
// LLMLogStream component renders the most-recent lines as a coloured mini
// terminal so the user can always tell which agent is doing what.
export interface LLMLogLine {
  t: number;
  line: string;
  tag: "info" | "ok" | "warn" | "error";
  agent: string;
}

interface ReflectiveSearchState {
  running: boolean;
  current_generation: number;
  total_generations: number | null;
  best_objective: number | null;
  history: ReflectiveSearchProgress[]; // one point per generation
  output_dir?: string;
  last_iter_eval?: { iter: number; response_id: number; objective: number | null };
  logs: ReflectiveSearchLogLine[]; // ring-buffered raw log lines (max 500)
  candidates: Record<number, ReflectiveSearchCandidateStat>; // per-generation health
}

interface RefinePrompt {
  prompt: string;
  cluster?: number;
  cells?: number[];
  suggested_dataset_id?: string;
  message: string;
  t: number;
}

interface CodeState {
  path?: string;
  code: string;
  nan_check?: CodeUpdateEvent["nan_check"];
  final?: boolean;
}

interface StoreState {
  connected: boolean;
  sessionId: string | null;
  session: SessionState | null;
  llm: { has_key: boolean; model?: string; api_key_masked?: string; source?: string } | null;
  socket: SessionSocket | null;

  pipeline: PipelineState;
  activeAgent: AgentId | null;
  awaitingResponse: boolean;
  messages: ChatMessage[];
  artifacts: ArtifactItem[];
  timeline: TimelineEntry[];
  toolCalls: ToolCallCard[];
  charts: ChartBag;
  subCharts: SubChartBag;
  reflectiveSearch: ReflectiveSearchState;
  // shared agent-runtime log (ring-buffered to 400 lines)
  llmLogs: LLMLogLine[];
  code: CodeState;
  routerDecision: RouterDecisionEvent | null;
  // Live running summary built by services/conversation_memory. Backend
  // publishes ``conversation_summary`` whenever the memo refreshes.
  conversationSummary: ConversationSummary | null;
  // Live workflow plan from services/planner. Null when no plan is active.
  plan: WorkflowPlan | null;
  // Slot-filling clarification text from the LLM router. The router asks
  // the user a follow-up question instead of dispatching an agent; we
  // surface it as a pinned banner above the chat box.
  routerClarification: string | null;
  refinePrompt: RefinePrompt | null;
  refineSubmitting: boolean;
  // Count of consecutive compute-timeout auto-retries; reset after a
  // successful compute or when the user issues a brand-new command. Capped
  // by COMPUTE_AUTO_RETRY_MAX in applyEvent to avoid infinite loops if the
  // LLM keeps emitting bad code.
  computeRetryCount: number;
  // When true, the next time Designer finishes successfully we will
  // automatically dispatch a Compute command (used by the auto-retry path
  // so the user doesn't have to manually type both messages).
  awaitingAutoCompute: boolean;

  bootstrap: () => Promise<void>;
  startSession: () => Promise<void>;
  sendMessage: (text: string) => Promise<void>;
  selectCluster: (cluster: number) => Promise<void>;
  toggleMode: (mode: "lightweight" | "reflective_search") => Promise<void>;
  refineCluster: (cluster: number) => Promise<void>;
  // Multi-cluster ops for the Decisions tab tickbox UI. `toggleCluster`
  // adds/removes a single id from the working selection (without
  // dispatching anything yet). `refineClusters` triggers the actual
  // RefinementAgent run on the full selection.
  toggleCluster: (cluster: number) => Promise<void>;
  refineClusters: (clusters: number[]) => Promise<void>;
  clearSelectedClusters: () => Promise<void>;
  confirmAll200: (datasetId?: string) => Promise<void>;
  dismissRefinePrompt: () => void;
  // Workflow-plan controls: confirm = run current step, skip = mark
  // current step skipped + advance, cancel = abort the plan.
  planAction: (action: "confirm" | "skip" | "cancel") => Promise<void>;
  // Edit a single pending step's params in-place from PlanPanel. Backend
  // clamps + validates against the agent's schema; returns the cleaned
  // params (which may differ from what we sent).
  planUpdateStep: (
    stepId: number,
    params: Record<string, unknown>,
  ) => Promise<{ ok: boolean; error?: string; params?: Record<string, unknown> }>;
  dismissClarification: () => void;
  cancelReflectiveSearch: () => Promise<void>;
  setDataset: (datasetId: string) => Promise<void>;
  refreshSession: () => Promise<void>;
  refreshLLM: () => Promise<void>;
  setLLM: (llm: { has_key: boolean; model?: string; api_key_masked?: string; source?: string } | null) => void;
  applyEvent: (ev: WSEvent) => void;
}

const useSession = create<StoreState>((set, get) => ({
  connected: false,
  sessionId: null,
  session: null,
  llm: null,
  socket: null,

  pipeline: { ...EMPTY_PIPELINE },
  activeAgent: null,
  awaitingResponse: false,
  messages: [],
  artifacts: [],
  timeline: [],
  toolCalls: [],
  charts: { cluster_iter: [] },
  subCharts: { cluster_iter: [] },
  reflectiveSearch: {
    running: false,
    current_generation: 0,
    total_generations: null,
    best_objective: null,
    history: [],
    logs: [],
    candidates: {},
  },
  llmLogs: [],
  code: { code: "" },
  routerDecision: null,
  conversationSummary: null,
  plan: null,
  routerClarification: null,
  refinePrompt: null,
  refineSubmitting: false,
  computeRetryCount: 0,
  awaitingAutoCompute: false,

  async bootstrap() {
    try {
      const h = await api.health();
      const llm = await api.getLLMSettings();
      set({ llm });
    } catch (e) {
      console.warn("bootstrap failed", e);
    }
  },

  async startSession() {
    const { socket } = get();
    socket?.close();
    const { session_id, state: rawState } = await api.createSession();
    const state = normalizeSessionState(rawState);
    set({
      sessionId: session_id,
      session: state,
      messages: state.messages,
      artifacts: state.artifacts,
      pipeline: state.pipeline_state || { ...EMPTY_PIPELINE },
      timeline: [],
      toolCalls: [],
      charts: { cluster_iter: [] },
      subCharts: { cluster_iter: [] },
      reflectiveSearch: {
        running: false,
        current_generation: 0,
        total_generations: null,
        best_objective: null,
        history: [],
        logs: [],
        candidates: {},
      },
      llmLogs: [],
      code: { code: "" },
      routerDecision: null,
      conversationSummary: null,
      plan: null,
      routerClarification: null,
      refinePrompt: null,
      refineSubmitting: false,
      awaitingResponse: false,
      connected: false,
    });
    const s = new SessionSocket(session_id);
    s.subscribe(get().applyEvent);
    s.open();
    set({ socket: s });
  },

  async refreshSession() {
    const sid = get().sessionId;
    if (!sid) return;
    const state = normalizeSessionState(await api.getSessionState(sid));
    set({
      session: state,
      messages: state.messages,
      artifacts: state.artifacts,
      pipeline: state.pipeline_state,
    });
  },

  async sendMessage(text: string) {
    const sid = get().sessionId;
    if (!sid) return;
    // optimistic append
    set((s) => ({
      awaitingResponse: true,
      messages: [
        ...s.messages,
        { id: `tmp-${Date.now()}`, role: "user", agent: null, text, t: Date.now() / 1000 },
      ],
    }));
    await api.postMessage(sid, text);
  },

  async selectCluster(cluster: number) {
    const sid = get().sessionId;
    if (!sid) return;
    await api.postAction(sid, { type: "select_cluster", cluster });
    set((s) => ({
      session: s.session
        ? {
            ...s.session,
            selected_cluster: cluster,
            // Single-cluster click also resets the multi-select state so
            // the UI doesn't keep stale tickboxes from a previous group.
            selected_clusters: [cluster],
          }
        : null,
    }));
  },

  async toggleCluster(cluster: number) {
    const sid = get().sessionId;
    if (!sid) return;
    const current = get().session?.selected_clusters ?? [];
    const next = current.includes(cluster)
      ? current.filter((c) => c !== cluster)
      : [...current, cluster].sort((a, b) => a - b);
    // Optimistic update — backend persists for chat-agent context.
    set((s) => ({
      session: s.session
        ? {
            ...s.session,
            selected_clusters: next,
            selected_cluster: next[0] ?? null,
          }
        : null,
    }));
    try {
      await api.postAction(sid, { type: "select_clusters", clusters: next });
    } catch (e) {
      console.warn("select_clusters action failed", e);
    }
  },

  async clearSelectedClusters() {
    const sid = get().sessionId;
    set((s) => ({
      session: s.session
        ? { ...s.session, selected_clusters: [], selected_cluster: null }
        : null,
    }));
    if (sid) {
      try {
        await api.postAction(sid, { type: "select_clusters", clusters: [] });
      } catch (e) {
        console.warn("clear select_clusters failed", e);
      }
    }
  },

  async refineClusters(clusters: number[]) {
    const sid = get().sessionId;
    if (!sid || clusters.length === 0) return;
    set({ subCharts: { cluster_iter: [] }, refinePrompt: null, refineSubmitting: true });
    try {
      await api.postAction(sid, {
        type: "refine_clusters",
        clusters: clusters,
      });
    } finally {
      set({ refineSubmitting: false });
    }
  },

  async toggleMode(mode) {
    const sid = get().sessionId;
    if (!sid) return;
    await api.postAction(sid, { type: "toggle_mode", mode });
    set((s) => ({ session: s.session ? { ...s.session, mode } : null }));
  },

  async refineCluster(cluster: number) {
    const sid = get().sessionId;
    if (!sid) return;
    set({ subCharts: { cluster_iter: [] }, refineSubmitting: true });
    try {
      await api.postAction(sid, { type: "refine_cluster", cluster });
    } finally {
      set({ refineSubmitting: false });
    }
  },

  async confirmAll200(datasetId?: string) {
    const sid = get().sessionId;
    if (!sid) return;
    set({ subCharts: { cluster_iter: [] }, refinePrompt: null, refineSubmitting: true });
    const body: Record<string, any> = { type: "confirm_all200" };
    if (datasetId) body.dataset_id = datasetId;
    try {
      await api.postAction(sid, body);
    } finally {
      set({ refineSubmitting: false });
    }
  },

  dismissRefinePrompt() {
    set({ refinePrompt: null });
  },

  async planAction(action) {
    const sid = get().sessionId;
    if (!sid) return;
    try {
      await api.postAction(sid, { type: "plan_action", action });
    } catch (e) {
      console.warn("plan_action failed", e);
    }
  },

  async planUpdateStep(stepId, params) {
    const sid = get().sessionId;
    if (!sid) return { ok: false, error: "no session" };
    try {
      const res = (await api.postAction(sid, {
        type: "plan_update_step",
        step_id: stepId,
        params,
      })) as { ok: boolean; error?: string; params?: Record<string, unknown> };
      // No need to mirror into state here — backend immediately
      // broadcasts plan_updated which our reducer picks up.
      return res;
    } catch (e: any) {
      console.warn("plan_update_step failed", e);
      return { ok: false, error: String(e?.message ?? e) };
    }
  },

  dismissClarification() {
    set({ routerClarification: null });
  },

  async cancelReflectiveSearch() {
    const sid = get().sessionId;
    if (!sid) return;
    await api.postAction(sid, { type: "cancel_reflective_search" });
  },

  async setDataset(datasetId: string) {
    const sid = get().sessionId;
    if (!sid) return;
    await api.postAction(sid, { type: "set_dataset", dataset_id: datasetId });
    await get().refreshSession();
  },

  async refreshLLM() {
    try {
      const llm = await api.getLLMSettings();
      set({ llm });
    } catch (e) {
      console.warn("refreshLLM failed", e);
    }
  },

  setLLM(llm) {
    set({ llm });
  },

  applyEvent(ev: WSEvent) {
    const t = (ev as any).t || Date.now() / 1000;
    const agent = (ev as any).agent ?? null;

    if (ev.kind === "hello") {
      // hello arrives BOTH on first connect AND on every reconnect (the
      // WS client auto-retries with exponential backoff). On reconnect
      // the backend may have:
      //   - been restarted (new in-memory store, session loaded from disk)
      //   - lost the previous session entirely (rare; manual cleanup)
      // Either way, the timeline / toolCalls in the frontend store are
      // "ephemeral" events from BEFORE the disconnect. Leaving them in
      // place after a reconnect produces a deeply confusing UI where the
      // pipeline shows "idle" but the timeline pretends an agent was
      // mid-run — exactly the bug the user reported. So we clear them.
      // Messages / artifacts come straight from the persisted session
      // and reflect the real saved state.
      const state = normalizeSessionState(ev.state);
      set({
        session: state,
        messages: state.messages,
        artifacts: state.artifacts,
        pipeline: state.pipeline_state || { ...EMPTY_PIPELINE },
        // ephemeral things — reset so the right-rail matches reality.
        timeline: [],
        toolCalls: [],
        llmLogs: [],
        activeAgent: null,
        awaitingResponse: false,
        // also reset the auto-retry book-keeping; whatever happened in
        // the old session is no longer ours to retry.
        computeRetryCount: 0,
        awaitingAutoCompute: false,
        connected: true,
      });
      return;
    }
    if (ev.kind === "ping") {
      set({ connected: true });
      return;
    }
    // Synthesised by SessionSocket when the underlying WS closes — used
    // to flip the connection indicator off so the TopBar can warn the
    // user that the backend went away.
    if ((ev as any).kind === "disconnected") {
      set({ connected: false });
      return;
    }

    // timeline entry helper
    const addTimeline = (text: string, payload?: any) =>
      set((s) => ({
        timeline: [
          ...s.timeline.slice(-300),
          { id: (ev as any).event_id || `${ev.kind}-${t}`, t, kind: ev.kind, agent, text, payload },
        ],
      }));

    switch (ev.kind) {
      case "agent_started": {
        set((s) => ({
          activeAgent: agent,
          awaitingResponse: true,
          pipeline: ev.node_id
            ? { ...s.pipeline, [ev.node_id]: "running" }
            : s.pipeline,
          reflectiveSearch:
            agent === "designer" &&
            (s.session?.mode === "reflective_search" || s.session?.mode === "full_eoh")
              ? {
                  running: true,
                  current_generation: 0,
                  total_generations: null,
                  best_objective: null,
                  history: [],
                  logs: [],
                  candidates: {},
                }
              : s.reflectiveSearch,
        }));
        addTimeline(`${agent ?? "?"} started`);
        break;
      }
      case "agent_finished": {
        const finishedAgent = agent;
        const finishedStatus = (ev as any).status;
        const failReason = (ev as any).reason as string | undefined;
        set((s) => ({
          activeAgent: null,
          awaitingResponse: false,
          pipeline:
            finishedAgent && finishedAgent !== "router" && finishedAgent !== "chat"
              ? {
                  ...s.pipeline,
                  [finishedAgent as keyof PipelineState]:
                    finishedStatus === "error" ? "error" : "done",
                }
              : s.pipeline,
          reflectiveSearch:
            finishedAgent === "designer"
              ? { ...s.reflectiveSearch, running: false }
              : s.reflectiveSearch,
          // Reset the auto-retry counter once Compute succeeds.
          computeRetryCount:
            finishedAgent === "compute" && finishedStatus !== "error"
              ? 0
              : s.computeRetryCount,
        }));
        addTimeline(`${finishedAgent ?? "?"} finished`, (ev as any).result);

        // Auto-retry on compute timeout — REVISED.
        //
        // Previously we ran a fully-automatic chain
        //   compute_timeout → auto-Design → auto-Compute → ...
        // up to 2 times. The unintended effect was that when the LLM
        // kept producing pathological MMD functions, the chain just
        // re-spawned 3 × 90 s = 4.5 min of compute subprocess churn
        // before giving up — to the user it looked like "the backend
        // hung". One log loop accidentally OOM'd the backend.
        //
        // New policy:
        //   * One Design retry only (no automatic Compute follow-up).
        //   * After the new Design lands, the user inspects the code
        //     and decides whether to dispatch Compute themselves. The
        //     ⏱ chat message already tells them exactly what to type.
        //   * If `BATRECYCLER_NO_AUTORETRY` is set on the window object
        //     (devtools / tests), even the Design retry is skipped.
        const COMPUTE_AUTO_RETRY_MAX = 1;
        const autoRetryDisabled =
          typeof window !== "undefined" &&
          (window as any).BATRECYCLER_NO_AUTORETRY === true;
        if (
          finishedAgent === "compute" &&
          finishedStatus === "error" &&
          failReason === "compute_timeout" &&
          !autoRetryDisabled
        ) {
          const { computeRetryCount, sendMessage } = get();
          if (computeRetryCount < COMPUTE_AUTO_RETRY_MAX) {
            set({ computeRetryCount: computeRetryCount + 1 });
            const attemptNum = computeRetryCount + 1;
            console.info(
              `[auto-retry] compute timed out — re-designing CDDI ` +
                `(attempt ${attemptNum}/${COMPUTE_AUTO_RETRY_MAX}). ` +
                "Compute will NOT auto-run; user dispatches it manually.",
            );
            setTimeout(() => {
              sendMessage(
                `Design a new CDDI function for First 200s data. ` +
                  `(auto-retry ${attemptNum}/${COMPUTE_AUTO_RETRY_MAX} after compute timeout — ` +
                  `please review the new code and re-run Compute manually if it looks good)`,
              ).catch((e) => console.warn("auto-retry sendMessage failed", e));
            }, 1200);
          } else {
            console.warn(
              `[auto-retry] giving up after ${COMPUTE_AUTO_RETRY_MAX} attempt(s).`,
            );
            set({ computeRetryCount: 0 });
          }
        }
        // NOTE: we deliberately no longer auto-dispatch Compute after a
        // Designer success. The user (or a future "retry compute" UI
        // button) decides.
        break;
      }
      case "thinking": {
        addTimeline((ev as any).text || "thinking…");
        break;
      }
      case "log": {
        addTimeline(`[log:${(ev as any).level}] ${(ev as any).text}`);
        break;
      }
      case "router_decision": {
        const r = ev as RouterDecisionEvent;
        set({
          routerDecision: r,
          // If the smart router asked a slot-filling question, surface
          // it as a pinned banner. Clear when the next decision is NOT
          // a clarification.
          routerClarification:
            r.needs_clarification && r.clarification ? r.clarification : null,
        });
        addTimeline(
          `router → ${r.intent} → ${r.next_agent}` +
            (r.confidence !== undefined ? ` (conf=${r.confidence.toFixed(2)})` : "") +
            (r.tool ? ` [${r.tool}]` : ""),
          ev,
        );
        break;
      }
      case "conversation_summary": {
        const cs = ev as ConversationSummaryEvent;
        if (cs.summary) {
          set({ conversationSummary: cs.summary });
          addTimeline("conversation_summary refreshed", cs.summary);
        }
        break;
      }
      case "plan_created":
      case "plan_updated": {
        const p = ev as PlanCreatedEvent | PlanUpdatedEvent;
        if (p.plan) {
          // Mirror onto store. Null out when the plan reaches a terminal
          // state so the panel collapses cleanly.
          const terminal =
            p.plan.status === "done" || p.plan.status === "cancelled";
          set({
            plan: p.plan,
            // Clear after a short delay so the user sees the final state
            // briefly before the panel disappears. Done inline with a
            // setTimeout to keep this reducer pure-ish.
          });
          if (terminal) {
            setTimeout(() => {
              const cur = useSession.getState().plan;
              if (cur && cur === p.plan) useSession.setState({ plan: null });
            }, 6000);
          }
          addTimeline(`plan ${ev.kind === "plan_created" ? "created" : "updated"} (${p.plan.status})`, p.plan);
        }
        break;
      }
      case "tool_call": {
        const t_ev = ev as ToolCallEvent;
        set((s) => {
          let cards = s.toolCalls.slice();
          if (t_ev.update && t_ev.call_id) {
            cards = cards.map((c) =>
              c.call_id === t_ev.call_id
                ? {
                    ...c,
                    status: t_ev.status,
                    duration_ms: t_ev.duration_ms ?? c.duration_ms,
                    result: t_ev.result ?? c.result,
                    error: t_ev.error ?? c.error,
                  }
                : c,
            );
          } else {
            cards = [
              ...cards.slice(-50),
              {
                call_id: t_ev.call_id,
                tool: t_ev.tool,
                agent,
                status: t_ev.status,
                duration_ms: t_ev.duration_ms,
                args: t_ev.args,
                result: t_ev.result,
                error: t_ev.error,
                t,
              },
            ];
          }
          return { toolCalls: cards };
        });
        addTimeline(`tool ${t_ev.tool} [${t_ev.status}]`, t_ev);
        break;
      }
      case "message": {
        const msg = (ev as any).message as ChatMessage | undefined;
        if (msg) {
          set((s) => {
            // If a streaming placeholder with the same id exists,
            // promote it to the persisted message (drop `streaming`).
            const idx = s.messages.findIndex((m) => m.id === msg.id);
            if (idx >= 0) {
              const next = s.messages.slice();
              next[idx] = { ...msg, streaming: false };
              return {
                messages: next,
                awaitingResponse: msg.role === "assistant" ? false : s.awaitingResponse,
              };
            }
            return {
              messages: [...s.messages.filter((m) => m.id !== `tmp-${msg.id}`), msg],
              awaitingResponse: msg.role === "assistant" ? false : s.awaitingResponse,
            };
          });
          addTimeline(`message: ${msg.agent ?? msg.role}`, msg);
        }
        break;
      }
      case "chat_message_started": {
        const e = ev as any;
        const id: string | undefined = e.id;
        if (!id) break;
        const agent = (e.agent ?? "chat") as AgentId;
        set((s) => {
          if (s.messages.some((m) => m.id === id)) {
            // Already present (e.g. resumed after reconnect); just stop
            // showing the "thinking" stub.
            return { awaitingResponse: false };
          }
          const stub: ChatMessage = {
            id,
            role: (e.role ?? "assistant") as "assistant" | "user",
            agent,
            text: "",
            t: Date.now() / 1000,
            streaming: true,
          };
          return { messages: [...s.messages, stub], awaitingResponse: false };
        });
        break;
      }
      case "chat_delta": {
        const e = ev as any;
        const id: string | undefined = e.id;
        if (!id) break;
        const fullText: string | undefined =
          typeof e.text === "string" ? e.text : undefined;
        const delta: string | undefined =
          typeof e.delta === "string" ? e.delta : undefined;
        if (fullText === undefined && delta === undefined) break;
        set((s) => {
          const idx = s.messages.findIndex((m) => m.id === id);
          if (idx < 0) return {};
          const next = s.messages.slice();
          const existing = next[idx];
          const newText =
            fullText !== undefined ? fullText : existing.text + (delta ?? "");
          next[idx] = { ...existing, text: newText, streaming: true };
          return { messages: next };
        });
        break;
      }
      case "chat_message_ended": {
        const e = ev as any;
        const id: string | undefined = e.id;
        if (!id) break;
        const aborted = !!e.aborted;
        set((s) => {
          const idx = s.messages.findIndex((m) => m.id === id);
          if (idx < 0) return {};
          if (aborted && !s.messages[idx].text) {
            // No content arrived — drop the empty bubble so the UI
            // doesn't leave a phantom assistant message behind.
            return { messages: s.messages.filter((_, i) => i !== idx) };
          }
          const next = s.messages.slice();
          next[idx] = { ...next[idx], streaming: false };
          return { messages: next };
        });
        break;
      }
      case "code_update": {
        const c = ev as CodeUpdateEvent;
        set({ code: { path: c.path, code: c.code, nan_check: c.nan_check, final: c.final } });
        addTimeline(
          `code_update (attempt ${c.nan_check?.attempts ?? "?"}, passed=${c.nan_check?.passed})`,
        );
        break;
      }
      case "chart_data": {
        const c = ev as ChartDataEvent;
        if (
          c.chart === "reflective_search_progress" ||
          c.chart === "eoh_progress"
        ) {
          set((s) => {
            const d = c.data || {};
            const phase = d.phase;
            const next: ReflectiveSearchState = {
              ...s.reflectiveSearch,
              running: true,
            };
            if (phase === "generation") {
              next.current_generation = d.generation ?? next.current_generation;
              next.total_generations = d.total_generations ?? next.total_generations;
            } else if (phase === "best_objective") {
              const obj = d.objective;
              if (obj !== null && obj !== undefined) {
                next.best_objective =
                  next.best_objective === null
                    ? obj
                    : Math.max(next.best_objective ?? -Infinity, obj);
              }
              next.history = [
                ...next.history,
                {
                  generation: d.generation ?? next.current_generation,
                  objective: obj ?? null,
                },
              ];
            } else if (phase === "iter_eval") {
              next.last_iter_eval = {
                iter: d.iter,
                response_id: d.response_id,
                objective: d.objective ?? null,
              };
            }
            return { reflectiveSearch: next };
          });
          break;
        }
        if (c.chart === "reflective_search_log" || c.chart === "eoh_log") {
          // ring-buffer raw log lines (max 500) so the user sees a live tail.
          set((s) => {
            const d = c.data || {};
            const incoming: ReflectiveSearchLogLine = {
              t: ev.t ?? Date.now() / 1000,
              line: String(d.line ?? "").slice(0, 500),
              tag: (d.tag as ReflectiveSearchLogLine["tag"]) ?? "info",
            };
            const logs = [...s.reflectiveSearch.logs, incoming];
            if (logs.length > 500) logs.splice(0, logs.length - 500);
            return {
              reflectiveSearch: {
                ...s.reflectiveSearch,
                running: true,
                logs,
              },
            };
          });
          break;
        }
        // Shared agent runtime log (Designer / Compute / Cluster all push here).
        // Ring-buffered to 400 lines so a long Compute run can't OOM the store.
        if (c.chart === "llm_log") {
          set((s) => {
            const d = c.data || {};
            const incoming: LLMLogLine = {
              t: ev.t ?? Date.now() / 1000,
              line: String(d.line ?? "").slice(0, 500),
              tag: (d.tag as LLMLogLine["tag"]) ?? "info",
              agent: String(d.agent ?? (ev as any).agent ?? "?"),
            };
            const llmLogs = [...s.llmLogs, incoming];
            if (llmLogs.length > 400) llmLogs.splice(0, llmLogs.length - 400);
            return { llmLogs };
          });
          break;
        }
        if (
          c.chart === "reflective_search_candidates" ||
          c.chart === "eoh_candidates"
        ) {
          set((s) => {
            const d = c.data || {};
            const g = Number(d.generation ?? 0);
            const candidates = {
              ...s.reflectiveSearch.candidates,
              [g]: {
                generation: g,
                ok: Number(d.ok ?? 0),
                inf: Number(d.inf ?? 0),
                total: Number(d.total ?? 0),
              },
            };
            return {
              reflectiveSearch: { ...s.reflectiveSearch, candidates },
            };
          });
          break;
        }
        const isSub = c.chart.startsWith("sub_");
        const key = isSub ? c.chart.replace(/^sub_/, "") : c.chart;
        set((s) => {
          if (isSub) {
            const sub = { ...s.subCharts };
            if (key === "cluster_iter") {
              sub.cluster_iter = [...(sub.cluster_iter || []).slice(-200), c.data];
            } else {
              (sub as any)[key] = c.data;
            }
            return { subCharts: sub };
          }
          const charts = { ...s.charts };
          if (key === "cluster_iter") {
            charts.cluster_iter = [...(charts.cluster_iter || []).slice(-200), c.data];
          } else {
            (charts as any)[key] = c.data;
          }
          return { charts };
        });
        if (key !== "mmd_progress" && key !== "cluster_iter") {
          addTimeline(`chart_data: ${c.chart}`);
        }
        break;
      }
      case "artifact": {
        const a = ev as any;
        const item: ArtifactItem = {
          name: a.name,
          kind: a.mime || a.kind || "file",
          path: a.path,
          size: a.preview?.size || 0,
          created_at: t,
          preview: a.preview,
        };
        set((s) => {
          const exists = s.artifacts.find((x) => x.name === item.name);
          const next = exists
            ? s.artifacts.map((x) => (x.name === item.name ? item : x))
            : [...s.artifacts, item];
          return { artifacts: next };
        });
        addTimeline(`artifact: ${item.name}`, item);
        break;
      }
      case "user_prompt_required": {
        const p = ev as UserPromptRequiredEvent;
        set({
          refinePrompt: {
            prompt: p.prompt,
            cluster: p.cluster,
            cells: p.cells,
            suggested_dataset_id: p.suggested_dataset_id,
            message: p.message,
            t,
          },
        });
        addTimeline(`user_prompt: ${p.message}`, ev);
        break;
      }
      default:
        // unknown kind, still log so we can debug
        addTimeline(`[${(ev as any).kind}]`, ev);
    }
  },
}));

export default useSession;
