import { Fragment, type ReactNode } from "react";

/**
 * Minimal markdown renderer for assistant chat messages.
 *
 * The backend posts richer messages now — capacity dispersion tables,
 * confirmation prompts with bullet lists, narrator-supplied paragraphs.
 * This component renders the subset of markdown we actually use:
 *
 *   - **bold**           (most-frequent emphasis style)
 *   - *italic*           (rare but useful)
 *   - `inline code`      (backticks around params, filenames, …)
 *   - bullet lists       ("- " / "* " / "• " at line start)
 *   - GitHub-style tables (consecutive lines starting with "|")
 *
 * Anything else is rendered as plain text with line breaks preserved,
 * exactly like the previous `whitespace-pre-wrap` behaviour.
 *
 * We avoid pulling in react-markdown to keep the bundle small (and to
 * stay buildable on a sandboxed network). The grammar handled here was
 * chosen to match what services/cluster_agent.py / pipeline.py /
 * narrator.py actually emit; if we start emitting raw HTML or nested
 * markdown, this renderer can be swapped without touching the call
 * sites.
 */

interface Props {
  text: string;
  className?: string;
}

const TABLE_LINE_RE = /^\s*\|.*\|\s*$/;
const TABLE_ALIGN_LINE_RE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$/;
const LIST_LINE_RE = /^\s*([-*•])\s+(.*)$/;
const HEADING_LINE_RE = /^(#{1,3})\s+(.+)$/;

interface TableBlock {
  type: "table";
  header: string[];
  rows: string[][];
}
interface ListBlock {
  type: "list";
  items: string[];
}
interface ParaBlock {
  type: "para";
  lines: string[];
}
interface HeadingBlock {
  type: "heading";
  level: 1 | 2 | 3;
  text: string;
}

type Block = TableBlock | ListBlock | ParaBlock | HeadingBlock;

function splitTableRow(line: string): string[] {
  // "| a | b | c |" → ["a", "b", "c"]. Trims trailing/leading empties from
  // the leading/closing pipe.
  const cells = line.split("|").map((c) => c.trim());
  if (cells.length && cells[0] === "") cells.shift();
  if (cells.length && cells[cells.length - 1] === "") cells.pop();
  return cells;
}

function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i++;
      continue;
    }

    // Headings (#, ##, ###)
    const headingMatch = line.match(HEADING_LINE_RE);
    if (headingMatch) {
      blocks.push({
        type: "heading",
        level: headingMatch[1].length as 1 | 2 | 3,
        text: headingMatch[2],
      });
      i++;
      continue;
    }

    // Table — needs a header line, an align line, then ≥ 1 body line.
    if (
      TABLE_LINE_RE.test(line) &&
      i + 1 < lines.length &&
      TABLE_ALIGN_LINE_RE.test(lines[i + 1])
    ) {
      const header = splitTableRow(line);
      const rows: string[][] = [];
      let j = i + 2;
      while (j < lines.length && TABLE_LINE_RE.test(lines[j])) {
        rows.push(splitTableRow(lines[j]));
        j++;
      }
      blocks.push({ type: "table", header, rows });
      i = j;
      continue;
    }

    // Bullet list.
    if (LIST_LINE_RE.test(line)) {
      const items: string[] = [];
      while (i < lines.length && LIST_LINE_RE.test(lines[i])) {
        items.push(lines[i].replace(LIST_LINE_RE, "$2"));
        i++;
      }
      blocks.push({ type: "list", items });
      continue;
    }

    // Paragraph — consume until blank line / table / list / heading.
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !LIST_LINE_RE.test(lines[i]) &&
      !HEADING_LINE_RE.test(lines[i]) &&
      !(
        TABLE_LINE_RE.test(lines[i]) &&
        i + 1 < lines.length &&
        TABLE_ALIGN_LINE_RE.test(lines[i + 1])
      )
    ) {
      para.push(lines[i]);
      i++;
    }
    if (para.length) blocks.push({ type: "para", lines: para });
  }
  return blocks;
}

/**
 * Render inline span: handles `**bold**`, `*italic*`, `` `code` ``.
 *
 * We do a single pass: scan for the next matching delimiter, emit the
 * preceding plain text, then the styled span. Conservatively requires
 * a closing delimiter — unmatched markers stay as literal text.
 */
function renderInline(text: string, keyPrefix: string): ReactNode {
  const out: ReactNode[] = [];
  let i = 0;
  let plain = "";
  let nodeIdx = 0;
  const flushPlain = () => {
    if (plain) {
      out.push(<Fragment key={`${keyPrefix}-t-${nodeIdx++}`}>{plain}</Fragment>);
      plain = "";
    }
  };
  while (i < text.length) {
    if (text[i] === "`") {
      const end = text.indexOf("`", i + 1);
      if (end > i) {
        flushPlain();
        out.push(
          <code
            key={`${keyPrefix}-c-${nodeIdx++}`}
            className="px-1 py-0.5 rounded bg-slate-200/60 text-[0.85em] font-mono"
          >
            {text.slice(i + 1, end)}
          </code>,
        );
        i = end + 1;
        continue;
      }
    } else if (text[i] === "*" && text[i + 1] === "*") {
      const end = text.indexOf("**", i + 2);
      if (end > i) {
        flushPlain();
        out.push(
          <strong key={`${keyPrefix}-b-${nodeIdx++}`} className="font-semibold">
            {text.slice(i + 2, end)}
          </strong>,
        );
        i = end + 2;
        continue;
      }
    } else if (text[i] === "*") {
      // single-asterisk italic. To avoid eating bullets ("* item")
      // we only treat it as italic when not preceded by whitespace
      // at the start of the line (the list-line regex already
      // caught that case before getting here).
      const end = text.indexOf("*", i + 1);
      if (end > i && text[end + 1] !== "*") {
        flushPlain();
        out.push(
          <em key={`${keyPrefix}-i-${nodeIdx++}`}>
            {text.slice(i + 1, end)}
          </em>,
        );
        i = end + 1;
        continue;
      }
    }
    plain += text[i];
    i++;
  }
  flushPlain();
  return out;
}

function renderParaLines(lines: string[], keyPrefix: string): ReactNode {
  return lines.map((ln, idx) => (
    <Fragment key={`${keyPrefix}-ln-${idx}`}>
      {idx > 0 && <br />}
      {renderInline(ln, `${keyPrefix}-i-${idx}`)}
    </Fragment>
  ));
}

export default function MessageContent({ text, className }: Props) {
  const blocks = parseBlocks(text);
  return (
    <div className={className}>
      {blocks.map((b, idx) => {
        if (b.type === "heading") {
          const Tag =
            b.level === 1 ? "h3" : b.level === 2 ? "h4" : "h5";
          const sizeCls =
            b.level === 1
              ? "text-base font-semibold mt-1 mb-1"
              : b.level === 2
                ? "text-sm font-semibold mt-1 mb-0.5"
                : "text-sm font-medium mt-0.5";
          return (
            <Tag key={idx} className={sizeCls}>
              {renderInline(b.text, `h-${idx}`)}
            </Tag>
          );
        }
        if (b.type === "table") {
          return (
            <div
              key={idx}
              className="my-1.5 overflow-x-auto scrollbar-thin -mx-1 px-1"
            >
              <table className="text-[12px] border-collapse">
                <thead>
                  <tr>
                    {b.header.map((h, j) => (
                      <th
                        key={j}
                        className="px-2 py-1 border border-slate-300/70 bg-slate-200/40 text-left font-semibold whitespace-nowrap"
                      >
                        {renderInline(h, `th-${idx}-${j}`)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {b.rows.map((row, ri) => (
                    <tr key={ri}>
                      {row.map((cell, ci) => (
                        <td
                          key={ci}
                          className="px-2 py-0.5 border border-slate-300/70 font-mono whitespace-nowrap"
                        >
                          {renderInline(cell, `td-${idx}-${ri}-${ci}`)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        if (b.type === "list") {
          return (
            <ul key={idx} className="list-disc pl-5 my-1 space-y-0.5">
              {b.items.map((item, j) => (
                <li key={j}>{renderInline(item, `li-${idx}-${j}`)}</li>
              ))}
            </ul>
          );
        }
        return (
          <p key={idx} className="my-1 leading-snug">
            {renderParaLines(b.lines, `p-${idx}`)}
          </p>
        );
      })}
    </div>
  );
}
