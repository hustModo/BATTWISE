import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import { api } from "@/api/rest";
import useSession from "@/store/sessionStore";

interface LLMSettingsForm {
  model: string;
  base_url: string;
  api_key: string;
  temperature: number;
}

type TestResult =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "ok"; model: string; latency_ms?: number; sample?: string }
  | { kind: "error"; message: string };

const PRESET_PROVIDERS: Array<{ label: string; model: string; base_url: string }> = [
  {
    label: "DeepSeek (Aliyun DashScope)",
    model: "deepseek-v3",
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
  {
    label: "Qwen Max (DashScope)",
    model: "qwen-max-latest",
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
  {
    label: "OpenAI",
    model: "gpt-4o-mini",
    base_url: "https://api.openai.com/v1",
  },
  {
    label: "DeepSeek (direct)",
    model: "deepseek-chat",
    base_url: "https://api.deepseek.com/v1",
  },
];

const DEFAULT_FORM: LLMSettingsForm = {
  model: "",
  base_url: "",
  api_key: "",
  temperature: 1.0,
};

export default function LLMSettingsDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const setLLM = useSession((s) => s.setLLM);
  const [form, setForm] = useState<LLMSettingsForm>(DEFAULT_FORM);
  const [currentMask, setCurrentMask] = useState<string>("");
  const [currentSource, setCurrentSource] = useState<string>("none");
  const [hasKey, setHasKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [test, setTest] = useState<TestResult>({ kind: "idle" });
  const [touchedKey, setTouchedKey] = useState(false);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      try {
        const cur = await api.getLLMSettings();
        if (cancelled) return;
        setForm({
          model: cur.model || "",
          base_url: cur.base_url || "",
          api_key: "",
          temperature: typeof cur.temperature === "number" ? cur.temperature : 1.0,
        });
        setCurrentMask(cur.api_key_masked || "");
        setCurrentSource(cur.source || "none");
        setHasKey(!!cur.has_key);
        setTouchedKey(false);
        setTest({ kind: "idle" });
      } catch (e: any) {
        setTest({ kind: "error", message: e?.message || "failed to load settings" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open]);

  function applyPreset(p: (typeof PRESET_PROVIDERS)[number]) {
    setForm((f) => ({ ...f, model: p.model, base_url: p.base_url }));
  }

  async function handleTest() {
    setTest({ kind: "running" });
    try {
      const body: any = {
        model: form.model || undefined,
        base_url: form.base_url || undefined,
      };
      if (touchedKey && form.api_key) body.api_key = form.api_key;
      const res = await api.testLLMSettings(body);
      if (res?.ok) {
        setTest({
          kind: "ok",
          model: res.model,
          latency_ms: res.latency_ms,
          sample: res.sample,
        });
      } else {
        setTest({
          kind: "error",
          message: res?.error || "test failed (unknown reason)",
        });
      }
    } catch (e: any) {
      setTest({ kind: "error", message: e?.message || "request failed" });
    }
  }

  async function handleSave() {
    setSaving(true);
    setTest({ kind: "idle" });
    try {
      const body: any = {
        model: form.model,
        base_url: form.base_url,
        temperature: form.temperature,
      };
      if (touchedKey) body.api_key = form.api_key;
      const res = await api.saveLLMSettings(body);
      if (res?.saved) {
        const cur = await api.getLLMSettings();
        setCurrentMask(cur.api_key_masked || "");
        setCurrentSource(cur.source || "none");
        setHasKey(!!cur.has_key);
        setForm((f) => ({ ...f, api_key: "" }));
        setTouchedKey(false);
        setLLM({
          model: cur.model,
          has_key: cur.has_key,
          source: cur.source,
        });
        onClose();
      }
    } catch (e: any) {
      setTest({ kind: "error", message: e?.message || "save failed" });
    } finally {
      setSaving(false);
    }
  }

  async function handleClear() {
    if (!confirm("Clear stored LLM credentials?")) return;
    setSaving(true);
    try {
      await api.saveLLMSettings({
        model: form.model,
        base_url: form.base_url,
        api_key: "",
        temperature: form.temperature,
      });
      const cur = await api.getLLMSettings();
      setCurrentMask(cur.api_key_masked || "");
      setCurrentSource(cur.source || "none");
      setHasKey(!!cur.has_key);
      setLLM({
        model: cur.model || "",
        has_key: !!cur.has_key,
        source: cur.source || "none",
      });
    } finally {
      setSaving(false);
    }
  }

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
            className="fixed top-0 right-0 z-50 h-screen w-full max-w-md bg-white shadow-2xl border-l border-slate-200 flex flex-col"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 260, damping: 28 }}
          >
            <header className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-slate-800">
                  LLM Settings
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Used by Router, Designer, Chat. Stored encrypted on the host.
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

            <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5">
              {hasKey && (
                <div className="rounded-md border border-emerald-200 bg-emerald-50/60 px-3 py-2 text-xs text-emerald-800">
                  Current key: <span className="font-mono">{currentMask}</span>
                  <span className="text-emerald-700/70"> · source={currentSource}</span>
                </div>
              )}

              <section>
                <label className="text-xs font-medium text-slate-700 mb-1.5 block">
                  Preset
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {PRESET_PROVIDERS.map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => applyPreset(p)}
                      className={clsx(
                        "text-left px-3 py-2 rounded-md border text-xs",
                        form.model === p.model && form.base_url === p.base_url
                          ? "border-battery-electrolyte/70 bg-battery-electrolyte/5"
                          : "border-slate-200 hover:bg-slate-50",
                      )}
                    >
                      <div className="font-medium text-slate-800">{p.label}</div>
                      <div className="text-slate-500 font-mono mt-0.5 truncate">
                        {p.model}
                      </div>
                    </button>
                  ))}
                </div>
              </section>

              <section className="space-y-3">
                <div>
                  <label className="text-xs font-medium text-slate-700 mb-1 block">
                    Model
                  </label>
                  <input
                    type="text"
                    value={form.model}
                    onChange={(e) => setForm({ ...form, model: e.target.value })}
                    className="w-full px-3 py-2 rounded-md border border-slate-200 text-sm font-mono focus:border-battery-electrolyte focus:ring-1 focus:ring-battery-electrolyte/30 outline-none"
                    placeholder="deepseek-v3"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-slate-700 mb-1 block">
                    Base URL
                  </label>
                  <input
                    type="text"
                    value={form.base_url}
                    onChange={(e) =>
                      setForm({ ...form, base_url: e.target.value })
                    }
                    className="w-full px-3 py-2 rounded-md border border-slate-200 text-sm font-mono focus:border-battery-electrolyte focus:ring-1 focus:ring-battery-electrolyte/30 outline-none"
                    placeholder="https://api.openai.com/v1"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-slate-700 mb-1 block">
                    API Key
                    {hasKey && (
                      <span className="ml-2 text-slate-400">
                        (leave empty to keep current)
                      </span>
                    )}
                  </label>
                  <input
                    type="password"
                    value={form.api_key}
                    onChange={(e) => {
                      setTouchedKey(true);
                      setForm({ ...form, api_key: e.target.value });
                    }}
                    className="w-full px-3 py-2 rounded-md border border-slate-200 text-sm font-mono focus:border-battery-electrolyte focus:ring-1 focus:ring-battery-electrolyte/30 outline-none"
                    placeholder={hasKey ? currentMask : "sk-..."}
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-slate-700 mb-1 block">
                    Temperature{" "}
                    <span className="text-slate-400">({form.temperature.toFixed(2)})</span>
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.05}
                    value={form.temperature}
                    onChange={(e) =>
                      setForm({ ...form, temperature: parseFloat(e.target.value) })
                    }
                    className="w-full accent-battery-electrolyte"
                  />
                </div>
              </section>

              <section>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={test.kind === "running" || !form.model || !form.base_url}
                    onClick={handleTest}
                    className="px-3 py-1.5 rounded-md text-xs font-medium border border-battery-electrolyte/30 text-battery-electrolyte bg-battery-electrolyte/5 hover:bg-battery-electrolyte/10 disabled:opacity-50"
                  >
                    {test.kind === "running" ? "Testing…" : "Test Connection"}
                  </button>
                  {hasKey && (
                    <button
                      type="button"
                      onClick={handleClear}
                      className="px-3 py-1.5 rounded-md text-xs font-medium border border-slate-200 text-slate-600 hover:bg-slate-50"
                    >
                      Clear
                    </button>
                  )}
                </div>
                {test.kind === "ok" && (
                  <div className="mt-2 text-xs rounded-md border border-emerald-200 bg-emerald-50/60 px-3 py-2 text-emerald-800">
                    OK · {test.model}
                    {typeof test.latency_ms === "number" ? ` · ${test.latency_ms}ms` : ""}
                    {test.sample ? (
                      <div className="mt-1 text-emerald-700/70 font-mono truncate">
                        “{test.sample}”
                      </div>
                    ) : null}
                  </div>
                )}
                {test.kind === "error" && (
                  <div className="mt-2 text-xs rounded-md border border-red-200 bg-red-50/60 px-3 py-2 text-red-700 whitespace-pre-wrap break-words">
                    {test.message}
                  </div>
                )}
              </section>
            </div>

            <footer className="px-5 py-4 border-t border-slate-200 flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-3 py-1.5 rounded-md text-xs font-medium text-slate-600 hover:bg-slate-100"
              >
                Cancel
              </button>
              <motion.button
                whileTap={{ scale: 0.96 }}
                type="button"
                disabled={saving || !form.model || !form.base_url}
                onClick={handleSave}
                className="px-4 py-1.5 rounded-md text-xs font-semibold bg-battery-electrolyte text-white shadow-card disabled:opacity-50"
              >
                {saving ? "Saving…" : "Save"}
              </motion.button>
            </footer>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
