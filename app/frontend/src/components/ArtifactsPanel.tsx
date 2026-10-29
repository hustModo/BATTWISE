import { motion } from "framer-motion";
import useSession from "@/store/sessionStore";
import { api } from "@/api/rest";
import { formatBytes, formatTime, truncate } from "@/utils/format";

const KIND_ICON: Record<string, string> = {
  py: "🐍",
  python: "🐍",
  csv: "📊",
  xlsx: "📑",
  xls: "📑",
  json: "🧾",
  txt: "📝",
  file: "📦",
};

function kindColor(kind: string): string {
  switch (kind) {
    case "py":
    case "python":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    case "csv":
      return "bg-amber-50 text-amber-700 border-amber-200";
    case "xlsx":
    case "xls":
      return "bg-blue-50 text-blue-700 border-blue-200";
    case "json":
      return "bg-violet-50 text-violet-700 border-violet-200";
    default:
      return "bg-slate-50 text-slate-700 border-slate-200";
  }
}

export default function ArtifactsPanel() {
  const sid = useSession((s) => s.sessionId);
  const artifacts = useSession((s) => s.artifacts);

  return (
    // h-full so the card fills the parent box (Workbench gives it a fixed
    // height of 140 px); without it, the card would size to its content and
    // either overflow the slot (overlap with ToolCalls below) or shrink.
    <div className="card p-3 flex flex-col gap-2 overflow-hidden h-full">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
          Artifacts
        </div>
        <div className="text-[11px] text-slate-400">
          {artifacts.length} item{artifacts.length === 1 ? "" : "s"}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin space-y-1.5 min-h-0">
        {artifacts.length === 0 && (
          <div className="h-full grid place-items-center text-xs text-slate-400 py-2">
            Generated files (gpt.py, mmd_results.csv, Cluster_label.xlsx, …) will
            appear here.
          </div>
        )}
        {artifacts.map((a) => {
          const previewLines: string[] = [];
          if (a.preview && typeof a.preview === "object") {
            for (const [k, v] of Object.entries(a.preview).slice(0, 3)) {
              if (v === undefined || v === null || v === "") continue;
              if (typeof v === "object") continue;
              previewLines.push(`${k}: ${truncate(String(v), 40)}`);
            }
          }
          const icon = KIND_ICON[a.kind] || KIND_ICON.file;
          const color = kindColor(a.kind);
          const href = sid ? api.artifactDownloadUrl(sid, a.name) : "#";
          return (
            <motion.a
              key={a.name}
              href={href}
              download={a.name}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className={`block rounded-lg border ${color} px-2.5 py-2 hover:shadow-card transition-shadow group`}
            >
              <div className="flex items-start gap-2">
                <span className="text-base leading-5">{icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-mono text-xs truncate group-hover:underline">
                      {a.name}
                    </span>
                    <span className="text-[10px] text-slate-500 shrink-0">
                      {formatBytes(a.size)}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">
                    {a.kind.toUpperCase()}
                    {a.created_at ? ` · ${formatTime(a.created_at)}` : ""}
                  </div>
                  {previewLines.length > 0 && (
                    <div className="mt-1 space-y-0.5">
                      {previewLines.map((line, i) => (
                        <div
                          key={i}
                          className="text-[10px] text-slate-600 font-mono truncate"
                        >
                          {line}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <span className="text-slate-400 group-hover:text-current text-xs shrink-0">
                  ↓
                </span>
              </div>
            </motion.a>
          );
        })}
      </div>
    </div>
  );
}
