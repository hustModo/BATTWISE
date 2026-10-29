import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import { api } from "@/api/rest";

type SlotId = "cells" | "model" | "char" | "indicator";

interface DatasetSummary {
  dataset_id: string;
  name: string;
  type: string;
  cell_count: number;
  model_file: string;
  char_file: string;
  indicator_file?: string;
  builtin: boolean;
}

interface QueueItem {
  id: string; // unique id (filename:slot)
  filename: string;
  slot: SlotId;
  size: number;
  file: File;
  status: "queued" | "uploading" | "done" | "error";
  progress: number; // 0..100
  error?: string;
}

const SLOT_LABELS: Record<SlotId, string> = {
  cells: "Cell files",
  model: "Model workbook",
  char: "Characteristics workbook",
  indicator: "Indicator workbook",
};

const DATASET_TYPES = [
  { value: "first_200s", label: "First 200s (CV)" },
  { value: "all_200", label: "All 200 (CV / refinement)" },
  { value: "cc_200", label: "CC 200" },
  { value: "cv_100", label: "CV 100" },
  { value: "first_1000s", label: "First 1000s" },
  { value: "custom", label: "Custom" },
];

const CHUNK_BYTES = 1024 * 1024; // 1 MiB

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DatasetManagerDrawer({
  open,
  onClose,
  onDatasetCreated,
}: {
  open: boolean;
  onClose: () => void;
  onDatasetCreated?: (dataset_id: string) => void;
}) {
  const [existing, setExisting] = useState<DatasetSummary[]>([]);
  const [form, setForm] = useState({ name: "", type: "first_200s" });
  const [draftId, setDraftId] = useState<string | null>(null);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [activeSlot, setActiveSlot] = useState<SlotId>("cells");
  const [finalizing, setFinalizing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const reloadList = useCallback(async () => {
    try {
      const res = await api.listDatasets();
      setExisting(res.datasets || []);
    } catch (e: any) {
      setError(e?.message || "Failed to load datasets");
    }
  }, []);

  useEffect(() => {
    if (open) {
      void reloadList();
    } else {
      // reset on close
      setDraftId(null);
      setQueue([]);
      setForm({ name: "", type: "first_200s" });
      setError(null);
      setFinalizing(false);
    }
  }, [open, reloadList]);

  const completedBySlot = useMemo(() => {
    const map: Record<SlotId, QueueItem[]> = { cells: [], model: [], char: [], indicator: [] };
    for (const item of queue) {
      if (item.status === "done") map[item.slot].push(item);
    }
    return map;
  }, [queue]);

  const canFinalize = useMemo(() => {
    return !!draftId && completedBySlot.cells.length > 0 && !busy && !finalizing;
  }, [draftId, completedBySlot.cells.length, busy, finalizing]);

  async function ensureDraft(): Promise<string> {
    if (draftId) return draftId;
    if (!form.name.trim()) {
      throw new Error("Dataset name is required");
    }
    setBusy(true);
    try {
      const res = await api.createDatasetDraft({
        name: form.name.trim(),
        type: form.type,
      });
      setDraftId(res.dataset_id);
      return res.dataset_id;
    } finally {
      setBusy(false);
    }
  }

  async function uploadOne(ds: string, item: QueueItem): Promise<void> {
    const total = Math.max(1, Math.ceil(item.file.size / CHUNK_BYTES));
    for (let i = 0; i < total; i++) {
      const start = i * CHUNK_BYTES;
      const end = Math.min(item.file.size, start + CHUNK_BYTES);
      const chunk = item.file.slice(start, end);
      try {
        await api.uploadDatasetChunk(ds, {
          filename: item.filename,
          slot: item.slot,
          chunkIndex: i,
          chunkCount: total,
          blob: chunk,
        });
      } catch (e: any) {
        throw new Error(`${item.filename} chunk ${i + 1}/${total}: ${e?.message || e}`);
      }
      const progress = Math.round(((i + 1) / total) * 100);
      setQueue((q) =>
        q.map((x) => (x.id === item.id ? { ...x, progress, status: "uploading" } : x)),
      );
    }
    setQueue((q) =>
      q.map((x) => (x.id === item.id ? { ...x, progress: 100, status: "done" } : x)),
    );
  }

  async function addAndUpload(files: FileList | File[], slot: SlotId) {
    setError(null);
    try {
      const ds = await ensureDraft();
      const arr = Array.from(files);
      const newItems: QueueItem[] = arr.map((f) => ({
        id: `${slot}:${f.name}:${f.size}:${Math.random().toString(36).slice(2, 7)}`,
        filename: f.name,
        slot,
        size: f.size,
        file: f,
        status: "queued",
        progress: 0,
      }));
      setQueue((q) => [...q, ...newItems]);
      setBusy(true);
      for (const item of newItems) {
        try {
          await uploadOne(ds, item);
        } catch (e: any) {
          setQueue((q) =>
            q.map((x) =>
              x.id === item.id ? { ...x, status: "error", error: e?.message || String(e) } : x,
            ),
          );
        }
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (!files || !files.length) return;
    void addAndUpload(files, activeSlot);
    e.target.value = "";
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    if (!e.dataTransfer.files?.length) return;
    void addAndUpload(e.dataTransfer.files, activeSlot);
  }

  async function handleFinalize() {
    if (!draftId) return;
    setFinalizing(true);
    setError(null);
    try {
      const modelFile = completedBySlot.model[0]?.filename;
      const charFile = completedBySlot.char[0]?.filename;
      const indicatorFile = completedBySlot.indicator[0]?.filename;
      const info = await api.finalizeDataset(draftId, {
        model_file: modelFile || "",
        char_file: charFile || "",
        indicator_file: indicatorFile || undefined,
      });
      onDatasetCreated?.(info.dataset_id);
      await reloadList();
      setDraftId(null);
      setQueue([]);
      setForm({ name: "", type: form.type });
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setFinalizing(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm(`Delete dataset "${id}"? This removes the uploaded files.`)) return;
    try {
      await api.deleteDataset(id);
      await reloadList();
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  }

  const totalProgress = useMemo(() => {
    if (queue.length === 0) return 0;
    const sum = queue.reduce((acc, it) => acc + it.progress, 0);
    return Math.round(sum / queue.length);
  }, [queue]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            role="dialog"
            aria-modal="true"
            className="fixed top-0 right-0 z-50 h-screen w-full max-w-xl bg-white shadow-2xl border-l border-slate-200 flex flex-col"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 260, damping: 28 }}
          >
            <header className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-slate-800">Datasets</h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Upload cell xlsx files + model / indicator workbooks. Built-in datasets are
                  read-only.
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="close"
                className="text-slate-400 hover:text-slate-700 p-1"
              >
                ✕
              </button>
            </header>

            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
              {error && (
                <div className="text-xs rounded-md border border-red-200 bg-red-50/70 px-3 py-2 text-red-700 whitespace-pre-wrap break-words">
                  {error}
                </div>
              )}

              {/* Existing datasets */}
              <section>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Available datasets
                  </h3>
                  <button
                    type="button"
                    onClick={() => void reloadList()}
                    className="text-xs text-slate-400 hover:text-slate-700"
                  >
                    refresh
                  </button>
                </div>
                <div className="space-y-1.5">
                  {existing.length === 0 && (
                    <div className="text-xs text-slate-400">No datasets registered.</div>
                  )}
                  {existing.map((d) => (
                    <div
                      key={d.dataset_id}
                      className="flex items-center justify-between text-xs rounded-md border border-slate-200 px-3 py-2"
                    >
                      <div className="min-w-0">
                        <div className="font-medium text-slate-800 truncate">
                          {d.name}
                          {d.builtin && (
                            <span className="ml-2 text-[10px] uppercase tracking-wide text-slate-400">
                              built-in
                            </span>
                          )}
                        </div>
                        <div className="text-slate-500 font-mono truncate">
                          {d.dataset_id} · {d.type} · {d.cell_count} cells
                        </div>
                      </div>
                      {!d.builtin && (
                        <button
                          type="button"
                          onClick={() => void handleDelete(d.dataset_id)}
                          className="ml-3 text-slate-400 hover:text-red-500 text-xs"
                        >
                          delete
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </section>

              {/* New dataset */}
              <section className="rounded-lg border border-slate-200 p-4 space-y-4">
                <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wide">
                  Upload new dataset
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs font-medium text-slate-600 block mb-1">Name</label>
                    <input
                      type="text"
                      value={form.name}
                      disabled={!!draftId}
                      onChange={(e) => setForm({ ...form, name: e.target.value })}
                      placeholder="My battery batch"
                      className="w-full px-3 py-2 rounded-md border border-slate-200 text-sm focus:border-battery-electrolyte focus:ring-1 focus:ring-battery-electrolyte/30 outline-none disabled:bg-slate-50"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-600 block mb-1">Type</label>
                    <select
                      value={form.type}
                      disabled={!!draftId}
                      onChange={(e) => setForm({ ...form, type: e.target.value })}
                      className="w-full px-3 py-2 rounded-md border border-slate-200 text-sm focus:border-battery-electrolyte focus:ring-1 focus:ring-battery-electrolyte/30 outline-none disabled:bg-slate-50"
                    >
                      {DATASET_TYPES.map((t) => (
                        <option key={t.value} value={t.value}>
                          {t.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                {draftId && (
                  <div className="text-xs text-slate-500 font-mono">
                    Draft id: <span className="text-slate-700">{draftId}</span>
                  </div>
                )}

                {/* Slot tabs */}
                <div className="flex gap-1">
                  {(Object.keys(SLOT_LABELS) as SlotId[]).map((slot) => (
                    <button
                      key={slot}
                      type="button"
                      onClick={() => setActiveSlot(slot)}
                      className={clsx(
                        "px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5",
                        activeSlot === slot
                          ? "bg-battery-electrolyte text-white shadow-card"
                          : "bg-slate-100 text-slate-600 hover:bg-slate-200",
                      )}
                    >
                      {SLOT_LABELS[slot]}
                      {completedBySlot[slot].length > 0 && (
                        <span
                          className={clsx(
                            "text-[10px] px-1.5 rounded-full",
                            activeSlot === slot
                              ? "bg-white/25 text-white"
                              : "bg-emerald-100 text-emerald-700",
                          )}
                        >
                          {completedBySlot[slot].length}
                        </span>
                      )}
                    </button>
                  ))}
                </div>

                {/* Dropzone */}
                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={onDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={clsx(
                    "rounded-lg border-2 border-dashed px-4 py-6 text-center cursor-pointer text-sm transition-colors",
                    "border-slate-300 text-slate-500 hover:border-battery-electrolyte hover:text-battery-electrolyte hover:bg-battery-electrolyte/5",
                  )}
                >
                  <input
                    type="file"
                    multiple
                    ref={fileInputRef}
                    onChange={onFileChange}
                    className="hidden"
                    accept=".xls,.xlsx,.csv"
                  />
                  <div className="font-medium">
                    Drop files here, or click to browse — for slot{" "}
                    <span className="text-battery-electrolyte">{SLOT_LABELS[activeSlot]}</span>
                  </div>
                  <div className="text-xs mt-1 text-slate-400">
                    .xls / .xlsx / .csv up to 1 GiB each · chunked 1 MiB at a time
                  </div>
                </div>

                {/* Per-file progress */}
                {queue.length > 0 && (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-500">
                        {queue.filter((q) => q.status === "done").length} / {queue.length} files
                      </span>
                      <span className="text-slate-500">{totalProgress}%</span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full rounded-full bg-battery-electrolyte transition-all"
                        style={{ width: `${totalProgress}%` }}
                      />
                    </div>
                    <ul className="max-h-40 overflow-y-auto space-y-1 mt-2">
                      {queue.map((q) => (
                        <li
                          key={q.id}
                          className="flex items-center justify-between text-xs rounded-md px-2 py-1 bg-slate-50"
                        >
                          <div className="min-w-0 flex-1 mr-2">
                            <div className="truncate text-slate-700">
                              <span className="text-slate-400 mr-1">[{q.slot}]</span>
                              {q.filename}
                            </div>
                            {q.status === "error" && (
                              <div className="text-red-600 truncate">{q.error}</div>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-slate-400">{bytes(q.size)}</span>
                            <span
                              className={clsx(
                                "w-1.5 h-1.5 rounded-full",
                                q.status === "done"
                                  ? "bg-emerald-500"
                                  : q.status === "error"
                                    ? "bg-red-500"
                                    : q.status === "uploading"
                                      ? "bg-amber-500 animate-pulse"
                                      : "bg-slate-300",
                              )}
                            />
                            <span className="text-slate-500 w-10 text-right">
                              {q.progress}%
                            </span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            </div>

            <footer className="px-5 py-4 border-t border-slate-200 flex items-center justify-between gap-2">
              <div className="text-xs text-slate-500">
                {draftId
                  ? `Draft "${draftId}" · ${completedBySlot.cells.length} cell files`
                  : "Pick a name to start uploading"}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-3 py-1.5 rounded-md text-xs font-medium text-slate-600 hover:bg-slate-100"
                >
                  Close
                </button>
                <motion.button
                  whileTap={{ scale: 0.96 }}
                  type="button"
                  disabled={!canFinalize}
                  onClick={handleFinalize}
                  className="px-4 py-1.5 rounded-md text-xs font-semibold bg-battery-electrolyte text-white shadow-card disabled:opacity-50"
                >
                  {finalizing ? "Finalizing…" : "Finalize dataset"}
                </motion.button>
              </div>
            </footer>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
