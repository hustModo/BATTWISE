// Shared TypeScript types mirroring backend event contract (docs/PLAN.md §五).

export type AgentId =
  | "router"
  | "designer"
  | "compute"
  | "cluster"
  | "refine"
  | "chat";

export type NodeId = "router" | "designer" | "compute" | "cluster" | "refine";

export type PipelineState = Record<NodeId, "idle" | "running" | "done" | "error">;

export type SessionMode = "lightweight" | "reflective_search" | "full_eoh";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  agent: AgentId | null;
  text: string;
  t: number;
  /**
   * True while the assistant message is still being streamed token-by-token
   * from the LLM. The UI shows a blinking cursor while this is set and
   * clears it on the final `message` event (or on `chat_message_ended`).
   */
  streaming?: boolean;
}

export interface ArtifactItem {
  name: string;
  kind: string;
  path: string;
  size: number;
  created_at: number;
  preview?: any;
}

export interface SessionState {
  session_id: string;
  created_at: number;
  updated_at: number;
  title: string;
  /**
   * ``full_eoh`` is accepted only when an older persisted session is loaded.
   * New UI actions and backend responses use ``reflective_search``.
   */
  mode: SessionMode;
  active_dataset_id: string;
  active_agent: AgentId | null;
  pipeline_state: PipelineState;
  selected_cluster: number | null;
  // Multi-cluster selection used by the Decisions tab tickboxes. The
  // backend keeps this in sync with `selected_cluster` (legacy single
  // selection always lives at index 0). Optional because older session
  // JSON files on disk may not carry the field; treat `undefined` as
  // empty array at read time.
  selected_clusters?: number[];
  messages: ChatMessage[];
  artifacts: ArtifactItem[];
  context: Record<string, any>;
}

export type WSEvent =
  | { kind: "hello"; session_id: string; state: SessionState }
  | { kind: "ping"; t: number }
  | (BaseEvent & {
      kind:
        | "agent_started"
        | "agent_finished"
        | "thinking"
        | "tool_call"
        | "router_decision"
        | "log"
        | "code_update"
        | "chart_data"
        | "artifact"
        | "message"
        | "chat_message_started"
        | "chat_delta"
        | "chat_message_ended"
        | "user_prompt_required"
        | "conversation_summary"
        | "plan_created"
        | "plan_updated";
      [key: string]: any;
    });

interface BaseEvent {
  t?: number;
  event_id?: string;
  agent?: AgentId;
}

export interface RouterDecisionEvent extends BaseEvent {
  kind: "router_decision";
  intent: string;
  next_agent: AgentId;
  reason?: string;
  // New fields produced by services/llm_router. All optional so older
  // backend versions / regex fallback don't fail the type guard.
  tool?: string;
  confidence?: number;
  needs_clarification?: boolean;
  clarification?: string | null;
  source?:
    | "regex-cancel"
    | "greeting-fast-path"
    | "llm-router"
    | "heuristic-fallback"
    | "fallback-chat"
    | "design_confirmation";
}

// ---------------------------------------------------------------------------
// Conversation summary — the running memo built by services/conversation_memory.
// Backend pushes a `conversation_summary` event whenever it refreshes the memo
// (throttled to ~8s). The frontend renders it as a small "AI 正在跟进:" card
// so the user can see what the assistant actually remembers across turns.
// ---------------------------------------------------------------------------
export interface ConversationSummary {
  what_user_wants: string;
  what_was_done: string[];
  current_state: string;
  user_preferences: Record<string, any>;
  open_questions: string[];
}

export interface ConversationSummaryEvent extends BaseEvent {
  kind: "conversation_summary";
  summary: ConversationSummary;
}

// ---------------------------------------------------------------------------
// Workflow plans — multi-step "Plan-and-execute" flow driven by
// services/planner.py. The backend stores the plan on session.context and
// broadcasts updates as either plan_created (first time) or plan_updated
// (after each confirm / skip / cancel / step completion).
// ---------------------------------------------------------------------------
export type PlanStatus =
  | "awaiting_confirmation"
  | "in_progress"
  | "done"
  | "cancelled";

export type PlanStepStatus =
  | "pending"
  | "running"
  | "done"
  | "skipped"
  | "cancelled"
  | "error";

export interface PlanStep {
  id: number;
  agent: "designer" | "compute" | "cluster" | "refine";
  description: string;
  params: Record<string, any>;
  status: PlanStepStatus;
  result_summary?: string | null;
}

export interface WorkflowPlan {
  goal: string;
  steps: PlanStep[];
  current_idx: number;
  status: PlanStatus;
}

export interface PlanCreatedEvent extends BaseEvent {
  kind: "plan_created";
  plan: WorkflowPlan;
}

export interface PlanUpdatedEvent extends BaseEvent {
  kind: "plan_updated";
  plan: WorkflowPlan;
}

export interface CodeUpdateEvent extends BaseEvent {
  kind: "code_update";
  path: string;
  code: string;
  nan_check: { passed: boolean; attempts: number; last_error?: string | null };
  final?: boolean;
}

export interface ChartDataEvent extends BaseEvent {
  kind: "chart_data";
  chart:
    | "mmd_progress"
    | "mmd_hist"
    | "mmd_correlation"
    | "cluster_iter"
    | "cluster_scatter"
    | "capacity_box"
    | "fingerprint"
    | "cluster_summary"
    | "eoh_progress"
    | "eoh_log"
    | "eoh_candidates"
    | "reflective_search_progress"
    | "reflective_search_log"
    | "reflective_search_candidates"
    | "llm_log"
    | "sub_mmd_progress"
    | "sub_cluster_iter"
    | "sub_cluster_scatter"
    | "sub_capacity_box"
    | "sub_fingerprint"
    | "sub_cluster_summary"
    | "cluster_dispersion"
    | "merged_cluster_dispersion"
    | "global_sorting_dispersion";
  data: any;
}

export interface UserPromptRequiredEvent extends BaseEvent {
  kind: "user_prompt_required";
  prompt: string; // e.g. "confirm_all200"
  cluster?: number;
  cells?: number[];
  suggested_dataset_id?: string;
  message: string;
}

export interface ToolCallEvent extends BaseEvent {
  kind: "tool_call";
  tool: string;
  status: "running" | "ok" | "error";
  args?: Record<string, any>;
  result?: any;
  duration_ms?: number;
  error?: string | null;
  call_id?: string;
  update?: boolean;
}

export interface ArtifactEvent extends BaseEvent {
  kind: "artifact";
  name: string;
  mime: string;
  path: string;
  preview?: any;
}

export interface ThinkingEvent extends BaseEvent {
  kind: "thinking";
  text: string;
  [key: string]: any;
}

export interface AgentLifecycleEvent extends BaseEvent {
  kind: "agent_started" | "agent_finished";
  result?: any;
  status?: string;
  error?: string;
  // Failure cause tag (e.g. "compute_timeout") used by the frontend to
  // decide whether to auto-retry. Optional.
  reason?: string;
}

export interface LogEvent extends BaseEvent {
  kind: "log";
  level: "info" | "warn" | "error";
  text: string;
}
