import { AnimatePresence, motion } from "framer-motion";
import useSession from "@/store/sessionStore";

export default function RefinementPanel() {
  const prompt = useSession((s) => s.refinePrompt);
  const confirm = useSession((s) => s.confirmAll200);
  const dismiss = useSession((s) => s.dismissRefinePrompt);

  return (
    <AnimatePresence>
      {prompt && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-40 bg-slate-900/30 backdrop-blur-sm grid place-items-center p-4"
          onClick={dismiss}
        >
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.98 }}
            transition={{ duration: 0.18 }}
            className="card max-w-2xl w-full p-6 relative"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="text-xs uppercase tracking-widest text-battery-electrolyte font-semibold">
                  Refinement · All-200s Acquisition
                </div>
                <div className="mt-1 font-semibold text-lg">
                  {prompt.cluster !== undefined ? `Cluster ${prompt.cluster}` : "Refinement"}
                </div>
              </div>
              <button
                onClick={dismiss}
                className="text-slate-400 hover:text-slate-700 text-lg leading-none"
                aria-label="dismiss"
              >
                ×
              </button>
            </div>

            <div className="mt-3 text-sm text-slate-700 whitespace-pre-wrap">
              {prompt.message}
            </div>

            {prompt.cells && prompt.cells.length > 0 && (
              <div className="mt-4">
                <div className="text-[11px] uppercase tracking-wide text-slate-500 mb-1">
                  Target cells ({prompt.cells.length})
                </div>
                <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto scrollbar-thin">
                  {prompt.cells.map((c) => (
                    <span
                      key={c}
                      className="px-1.5 py-0.5 rounded text-[11px] bg-slate-100 text-slate-700 font-mono"
                    >
                      #{c}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-6 flex justify-end gap-2">
              <button
                onClick={dismiss}
                className="px-3 py-1.5 rounded-lg border border-slate-200 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => confirm(prompt.suggested_dataset_id)}
                className="px-3 py-1.5 rounded-lg bg-battery-electrolyte text-white text-sm font-medium hover:shadow-card transition-shadow"
              >
                {prompt.suggested_dataset_id
                  ? `Use ${prompt.suggested_dataset_id}`
                  : "Confirm All-200s Ready"}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
