import Editor from "@monaco-editor/react";
import clsx from "clsx";
import { useCallback, useState } from "react";
import useSession from "@/store/sessionStore";
import { api } from "@/api/rest";

export default function MMDCodeViewer() {
  const code = useSession((s) => s.code);
  const sessionId = useSession((s) => s.sessionId);
  const artifacts = useSession((s) => s.artifacts);
  const [justCopied, setJustCopied] = useState(false);

  const status = code.nan_check?.passed
    ? code.final
      ? "final"
      : "passed"
    : code.nan_check?.attempts
      ? `repairing (${code.nan_check.attempts})`
      : "—";

  const statusClass = code.nan_check?.passed
    ? "bg-battery-charge/15 text-battery-charge"
    : code.nan_check?.attempts
      ? "bg-battery-amber/15 text-battery-amber"
      : "bg-slate-100 text-slate-500";

  const hasCode = Boolean(code.code && code.code.trim().length > 0);

  // Prefer the server-canonical artifact (registered after Designer finishes
  // a passing run — keeps preview / size / created_at metadata coherent).
  // Fall back to a Blob built from the in-memory streamed code so the
  // button still works mid-evolution before Designer registers the file.
  const downloadCode = useCallback(() => {
    if (!hasCode) return;
    const artifact = artifacts.find((a) => a.name === "cddi_function.py")
      ?? artifacts.find((a) => a.name === "gpt.py");
    if (artifact && sessionId) {
      const a = document.createElement("a");
      a.href = api.artifactDownloadUrl(sessionId, artifact.name);
      a.download = "cddi_function.py";
      document.body.appendChild(a);
      a.click();
      a.remove();
      return;
    }
    // Fallback: blob from the streamed code (works pre-artifact)
    const blob = new Blob([code.code ?? ""], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cddi_function.py";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [hasCode, artifacts, sessionId, code.code]);

  const copyCode = useCallback(async () => {
    if (!hasCode || !code.code) return;
    try {
      await navigator.clipboard.writeText(code.code);
      setJustCopied(true);
      setTimeout(() => setJustCopied(false), 1500);
    } catch {
      // ignore (clipboard permission may be denied)
    }
  }, [hasCode, code.code]);

  return (
    <div className="card flex flex-col h-full min-h-0 overflow-hidden">
      <div className="px-4 py-2.5 border-b flex items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
            CDDI Function (Designer)
          </div>
          {code.path && <div className="text-[11px] text-slate-400 truncate min-w-0">{code.path}</div>}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            type="button"
            onClick={copyCode}
            disabled={!hasCode}
            title="Copy current code to clipboard"
            className={clsx(
              "px-2 py-0.5 rounded text-[10px] uppercase font-semibold border transition-colors",
              hasCode
                ? "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                : "bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed",
            )}
          >
            {justCopied ? "Copied" : "Copy"}
          </button>
          <button
            type="button"
            onClick={downloadCode}
            disabled={!hasCode}
            title="Download CDDI function as cddi_function.py"
            className={clsx(
              "px-2 py-0.5 rounded text-[10px] uppercase font-semibold border transition-colors",
              hasCode
                ? "bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100"
                : "bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed",
            )}
          >
            ↓ Download .py
          </button>
          <div className={clsx("px-2 py-0.5 rounded text-[10px] uppercase font-semibold", statusClass)}>
            {status}
          </div>
        </div>
      </div>
      <div className="flex-1 min-h-0">
        {code.code ? (
          <Editor
            height="100%"
            defaultLanguage="python"
            value={code.code}
            theme="vs"
            options={{
              readOnly: true,
              fontSize: 12.5,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              wordWrap: "on",
              renderLineHighlight: "none",
              padding: { top: 8 },
            }}
          />
        ) : (
          <div className="h-full grid place-items-center text-slate-400 text-sm">
            Designer will stream code here once it generates a CDDI scoring function.
          </div>
        )}
      </div>
      {code.nan_check?.last_error && (
        <div className="px-3 py-1.5 border-t text-[11px] text-battery-danger bg-battery-danger/5 truncate">
          {code.nan_check.last_error}
        </div>
      )}
    </div>
  );
}
