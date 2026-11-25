import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import useSession from "@/store/sessionStore";

const STEPS = [
  {
    title: "1. Acquire",
    desc: "Run a First-200s charging protocol on each cell; upload the batch as a dataset.",
  },
  {
    title: "2. Design",
    desc: "Designer proposes programs. Seventeen charging curve descriptors guide scoring. Compute checks each selected program.",
  },
  {
    title: "3. Cluster",
    desc: "Adaptive GA-KMeans groups cells by OCV / IR / CDDI with multi-objective fitness.",
  },
  {
    title: "4. Refine",
    desc: "Select a group for All-200 refinement using 200 current–voltage pairs sampled uniformly from each complete curve.",
  },
];

export default function Landing() {
  const nav = useNavigate();
  const startSession = useSession((s) => s.startSession);
  const llm = useSession((s) => s.llm);

  const go = async () => {
    await startSession();
    nav("/workbench");
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="px-8 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-battery-electrolyte/15 grid place-items-center">
            <span className="text-battery-electrolyte text-lg font-semibold">⌬</span>
          </div>
          <div>
            <div className="font-semibold tracking-wide">BATTWISE</div>
            <div className="text-xs text-slate-500">
              Battery agentic triage and testing workflow with internal state evidence
            </div>
          </div>
        </div>
        <div className="text-xs text-slate-500">
          LLM:&nbsp;
          {llm?.has_key ? (
            <span className="text-battery-charge">
              {llm.model} ({llm.source}) {llm.api_key_masked}
            </span>
          ) : (
            <span className="text-battery-danger">not configured</span>
          )}
        </div>
      </header>

      <main className="flex-1 grid place-items-center px-6">
        <div className="max-w-5xl w-full">
          <motion.h1
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="text-4xl md:text-5xl font-bold text-battery-ink leading-tight"
          >
            Sort retired cells
            <br className="hidden md:block" />
            <span className="text-battery-electrolyte"> using their charging curves.</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="mt-5 text-slate-600 max-w-3xl"
          >
            Router and Planner organize the sorting tasks. Designer and Compute turn a cell curve
            and a reference response into one charging curve distribution discrepancy indicator
            (CDDI) per cell. Inspect the groups and select cells for refinement.
          </motion.p>

          <div className="mt-10 grid md:grid-cols-4 gap-4">
            {STEPS.map((s, i) => (
              <motion.div
                key={s.title}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, delay: 0.15 + i * 0.06 }}
                className="card p-5"
              >
                <div className="text-xs uppercase tracking-widest text-battery-electrolyte font-semibold">
                  {s.title}
                </div>
                <div className="mt-2 text-sm text-slate-700 leading-relaxed">{s.desc}</div>
              </motion.div>
            ))}
          </div>

          <div className="mt-12 flex flex-wrap gap-3">
            <button
              onClick={go}
              className="px-6 py-3 rounded-lg bg-battery-electrolyte text-white font-medium
                         shadow-card hover:shadow-cardHover transition-all"
            >
              Start a new session →
            </button>
            <button
              onClick={() => nav("/workbench")}
              className="px-6 py-3 rounded-lg border border-slate-300 text-slate-700
                         hover:bg-slate-50 transition-colors"
            >
              Open workbench
            </button>
          </div>
        </div>
      </main>

      <footer className="px-8 py-4 text-xs text-slate-400">
        BATTWISE · single-host research system · backend on :8765
      </footer>
    </div>
  );
}
