import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import useSession from "@/store/sessionStore";
import LLMSettingsDrawer from "@/components/LLMSettingsDrawer";
import DatasetManagerDrawer from "@/components/DatasetManagerDrawer";
import { api } from "@/api/rest";

export default function TopBar() {
  const session = useSession((s) => s.session);
  const llm = useSession((s) => s.llm);
  const connected = useSession((s) => s.connected);
  const toggleMode = useSession((s) => s.toggleMode);
  const refreshLLM = useSession((s) => s.refreshLLM);
  const setDataset = useSession((s) => s.setDataset);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [datasetsOpen, setDatasetsOpen] = useState(false);
  const [datasets, setDatasets] = useState<Array<{ dataset_id: string; name: string }>>([]);
  const [autoNudged, setAutoNudged] = useState(false);

  const mode = session?.mode ?? "lightweight";

  useEffect(() => {
    if (
      llm !== null &&
      !llm.has_key &&
      !autoNudged &&
      session?.session_id
    ) {
      setSettingsOpen(true);
      setAutoNudged(true);
    }
  }, [llm, autoNudged, session?.session_id]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const r = await api.listDatasets();
        if (!cancelled) setDatasets(r.datasets || []);
      } catch {
        /* ignore */
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [datasetsOpen, session?.active_dataset_id]);

  function handleSettingsClose() {
    setSettingsOpen(false);
    void refreshLLM();
  }

  return (
    <div className="h-14 px-5 flex items-center justify-between border-b bg-white/80 backdrop-blur sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-battery-electrolyte/15 grid place-items-center">
            <span className="text-battery-electrolyte text-base font-semibold">⌬</span>
          </div>
          <div className="font-semibold tracking-wide text-battery-ink">BATTWISE</div>
        </Link>
        {session?.session_id && (
          <div className="ml-3 text-xs text-slate-500 px-2 py-1 rounded-md bg-slate-100">
            Session #{session.session_id}
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 text-xs">
        <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-slate-100 text-slate-700">
          <span>Dataset:</span>
          <select
            className="bg-transparent outline-none font-medium cursor-pointer max-w-[12rem]"
            value={session?.active_dataset_id ?? ""}
            onChange={(e) => void setDataset(e.target.value)}
          >
            {!session?.active_dataset_id && <option value="">—</option>}
            {datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.name}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setDatasetsOpen(true)}
            className="ml-1 text-slate-400 hover:text-slate-700 text-sm leading-none"
            title="Manage datasets"
            aria-label="Manage datasets"
          >
            ＋
          </button>
        </div>
        <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-slate-100 text-slate-700">
          <span>Mode:</span>
          <select
            className="bg-transparent outline-none font-medium cursor-pointer"
            value={mode}
            onChange={(e) => toggleMode(e.target.value as any)}
          >
            <option value="lightweight">Lightweight</option>
            <option value="reflective_search">Reflective Search</option>
          </select>
        </div>
        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          className={`px-2 py-1 rounded-md cursor-pointer transition ${
            llm?.has_key
              ? "bg-battery-charge/10 text-battery-charge hover:bg-battery-charge/15"
              : "bg-battery-danger/10 text-battery-danger hover:bg-battery-danger/15"
          }`}
          title="Configure LLM credentials"
        >
          LLM: {llm?.has_key ? `${llm.model}` : "not configured"}
        </button>
        <div
          className={`flex items-center gap-1.5 px-2 py-1 rounded-md ${
            connected ? "text-battery-charge" : "text-slate-400"
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              connected ? "bg-battery-charge animate-pulse" : "bg-slate-300"
            }`}
          />
          {connected ? "live" : "offline"}
        </div>
        <motion.button
          whileHover={{ rotate: 30 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setSettingsOpen(true)}
          className="p-1.5 rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-700"
          aria-label="Settings"
          title="Settings"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </motion.button>
      </div>
      <LLMSettingsDrawer open={settingsOpen} onClose={handleSettingsClose} />
      <DatasetManagerDrawer
        open={datasetsOpen}
        onClose={() => setDatasetsOpen(false)}
        onDatasetCreated={(id) => {
          void setDataset(id);
        }}
      />
    </div>
  );
}
