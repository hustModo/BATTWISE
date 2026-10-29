import type { SessionState } from "@/types";

const BASE = "/api";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${res.statusText} — ${detail.slice(0, 200)}`);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return undefined as T;
}

export const api = {
  health: () => http<any>("/health"),

  createSession: () =>
    http<{ session_id: string; state: SessionState }>("/session", { method: "POST", body: "{}" }),

  getSessionState: (sid: string) => http<SessionState>(`/session/${sid}/state`),

  postMessage: (sid: string, text: string) =>
    http<{ accepted: boolean }>(`/session/${sid}/message`, {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  postAction: (sid: string, body: Record<string, any>) =>
    http<any>(`/session/${sid}/action`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listDatasets: () => http<{ datasets: any[] }>("/datasets"),

  createDatasetDraft: (body: { name: string; type?: string; dataset_id?: string }) =>
    http<{ dataset_id: string; name: string; type: string }>(`/datasets`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  uploadDatasetChunk: async (
    datasetId: string,
    args: {
      filename: string;
      slot: "cells" | "model" | "char" | "indicator";
      chunkIndex: number;
      chunkCount: number;
      blob: Blob;
    },
  ) => {
    const fd = new FormData();
    fd.set("filename", args.filename);
    fd.set("slot", args.slot);
    fd.set("chunk_index", String(args.chunkIndex));
    fd.set("chunk_count", String(args.chunkCount));
    fd.set("file", args.blob, args.filename);
    const res = await fetch(`${BASE}/datasets/${encodeURIComponent(datasetId)}/upload`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(`${res.status} ${res.statusText} — ${detail.slice(0, 200)}`);
    }
    return res.json();
  },

  finalizeDataset: (
    datasetId: string,
    body: { model_file?: string; char_file?: string; indicator_file?: string },
  ) =>
    http<any>(`/datasets/${encodeURIComponent(datasetId)}/finalize`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  deleteDataset: (datasetId: string) =>
    http<{ deleted: string }>(`/datasets/${encodeURIComponent(datasetId)}`, {
      method: "DELETE",
    }),

  getLLMSettings: () => http<any>("/settings/llm"),

  saveLLMSettings: (body: any) =>
    http<any>("/settings/llm", { method: "PUT", body: JSON.stringify(body) }),

  testLLMSettings: (body: any) =>
    http<any>("/settings/llm/test", { method: "POST", body: JSON.stringify(body) }),

  artifactDownloadUrl: (sid: string, name: string) =>
    `${BASE}/artifact/${sid}/${encodeURIComponent(name)}`,
};
