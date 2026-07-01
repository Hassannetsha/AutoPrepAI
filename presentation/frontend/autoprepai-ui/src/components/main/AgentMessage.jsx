import { useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, BarChart3, Info } from "lucide-react";

/**
 * Renders a structured pipeline-log bot message (the ✅ / • / 📊 / [System]
 * format your backend emits) with a real visual hierarchy:
 *  - the "Applied" line becomes a success header
 *  - consecutive "Skipping agent" lines collapse into one quiet toggle
 *  - the "Executing agent" line becomes a highlighted, pulsing active step
 *  - sub-steps (e.g. "Handling outliers") nest under it
 *  - the explanation becomes a labeled callout that truncates if long
 *  - the row/column summary becomes a stat pill
 *  - the trailing [System] note sits below a hairline, dimmed
 *
 * Falls back gracefully: any line that doesn't match a known pattern is
 * still rendered as plain text, so nothing is ever dropped.
 */

const SUGGESTION_RE = /^(?:\d+\.\s+)?(?<name>\w[\w\s()-]*?):\s+(?<desc>.*?)\s*\|\s*code:\s*(?<code>.*)$/;

function parseLines(text) {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

function classifyLine(line) {
  if (line.startsWith("✅")) return { type: "success", content: line.replace("✅", "").trim() };
  if (line.startsWith("📊")) return { type: "stat", content: line.replace("📊", "").trim() };
  if (line.startsWith("[System]")) return { type: "system", content: line.replace("[System]", "").trim() };
  if (/^•\s*Skipping agent:/i.test(line)) return { type: "skip", content: line.replace(/^•\s*/, "") };
  if (/^•\s*Executing agent:/i.test(line)) return { type: "exec", content: line.replace(/^•\s*/, "") };
  if (/^•\s*Explanation for/i.test(line)) return { type: "explanation", content: line.replace(/^•\s*/, "") };
  if (line.startsWith("•")) return { type: "sub", content: line.replace(/^•\s*/, "") };
  const s = line.match(SUGGESTION_RE);
  if (s) return { type: "suggestion", name: s.groups.name, desc: s.groups.desc, code: s.groups.code };
  return { type: "text", content: line };
}

// Collapse consecutive "skip" lines into a single group so the noisy
// "Skipping agent: X" x5 doesn't bury the one step that actually ran.
function groupLines(classified) {
  const groups = [];
  let skipBuffer = [];
  classified.forEach((item) => {
    if (item.type === "skip") {
      skipBuffer.push(item.content);
      return;
    }
    if (skipBuffer.length) {
      groups.push({ type: "skip-group", items: skipBuffer });
      skipBuffer = [];
    }
    groups.push(item);
  });
  if (skipBuffer.length) groups.push({ type: "skip-group", items: skipBuffer });
  return groups;
}

// Minimal **bold** support without pulling in a markdown parser.
function boldify(str) {
  return str.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

function SkipGroup({ items }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="agent-skip-group">
      <button className="agent-skip-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Skipped {items.length} agent{items.length > 1 ? "s" : ""}
      </button>
      {open && (
        <ul className="agent-skip-list">
          {items.map((item, i) => (
            <li key={i}>{item.replace(/^Skipping agent:\s*/i, "")}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SuggestionItem({ name, desc, code }) {
  return (
    <div className="agent-suggestion">
      <div className="agent-suggestion-header">
        <span className="agent-suggestion-name">{name}</span>
        <code className="agent-suggestion-code">{code}</code>
      </div>
      <p className="agent-suggestion-desc">{desc}</p>
    </div>
  );
}

function Explanation({ content }) {
  const [expanded, setExpanded] = useState(false);
  const match = content.match(/^Explanation for '([^']+)':\s*(.*)$/i);
  const label = match ? match[1] : "this step";
  const body = match ? match[2] : content;
  const isLong = body.length > 140;
  const preview = isLong && !expanded ? body.slice(0, 140).trim() + "…" : body;

  return (
    <div className="agent-explanation">
      <div className="agent-explanation-label">
        <Info size={13} />
        Why “{label}”?
      </div>
      <p className="agent-explanation-body">{preview}</p>
      {isLong && (
        <button className="agent-explanation-toggle" onClick={() => setExpanded((e) => !e)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

export default function AgentMessage({ text }) {
  const groups = groupLines(parseLines(text).map(classifyLine));

  return (
    <div className="agent-message">
      {groups.map((g, i) => {
        switch (g.type) {
          case "success":
            return (
              <div className="agent-line agent-line-success" key={i}>
                <CheckCircle2 size={16} />
                <span>{boldify(g.content)}</span>
              </div>
            );
          case "skip-group":
            return <SkipGroup items={g.items} key={i} />;
          case "exec":
            return (
              <div className="agent-line agent-line-exec" key={i}>
                <span className="agent-exec-dot" />
                <span>{boldify(g.content.replace(/^Executing agent:\s*/i, ""))}</span>
              </div>
            );
          case "suggestion":
            return <SuggestionItem name={g.name} desc={g.desc} code={g.code} key={i} />;
          case "sub":
            return (
              <div className="agent-line-sub" key={i}>
                {g.content}
              </div>
            );
          case "explanation":
            return <Explanation content={g.content} key={i} />;
          case "stat":
            return (
              <div className="agent-stat-row" key={i}>
                <BarChart3 size={14} />
                <span>{boldify(g.content)}</span>
              </div>
            );
          case "system":
            return (
              <div className="agent-line-system" key={i}>
                {g.content}
              </div>
            );
          default:
            return (
              <p className="agent-line-text" key={i}>
                {boldify(g.content)}
              </p>
            );
        }
      })}
    </div>
  );
}