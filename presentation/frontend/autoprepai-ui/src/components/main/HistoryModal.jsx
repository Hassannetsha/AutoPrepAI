import { X, Lightbulb, CheckCircle2, XCircle, Info, Play } from "lucide-react";

const SUGGESTION_RE = /^(?:\d+\.\s+)?(?<name>\w[\w\s()-]*?):\s+(?<desc>.*?)\s*\|\s*code:\s*(?<code>.*)$/;

function iconFor(message) {
  if (message.startsWith("[System] User ACCEPTED")) return { icon: CheckCircle2, cls: "history-accept" };
  if (message.startsWith("[System] User REJECTED")) return { icon: XCircle, cls: "history-reject" };
  if (message.startsWith("[System]")) return { icon: Info, cls: "history-system" };
  if (message.includes("Executing agent")) return { icon: Play, cls: "history-agent" };
  return null;
}

function cleanMessage(message) {
  return message.replace(/^\[System\] /, "");
}

function isSuggestionLog(message) {
  return message.startsWith("Generated suggestions:");
}

function SuggestionItem({ name, desc, code }) {
  return (
    <div className="history-suggestion">
      <div className="history-suggestion-name">{name}</div>
      <code className="history-suggestion-code">{code}</code>
      <p className="history-suggestion-desc">{desc}</p>
    </div>
  );
}

function SuggestionBlock({ message, time }) {
  const lines = message.split("\n");
  const suggestions = [];
  for (const line of lines) {
    const m = line.match(SUGGESTION_RE);
    if (m) suggestions.push(m.groups);
  }
  return (
    <div className="history-entry history-suggestion-block">
      <div className="history-suggestion-header-row">
        <Lightbulb size={14} />
        <span>Feature Engineering Suggestions</span>
      </div>
      <div className="history-suggestion-list">
        {suggestions.map((s, i) => (
          <SuggestionItem key={i} name={s.name} desc={s.desc} code={s.code} />
        ))}
      </div>
      <span className="history-entry-time">{time}</span>
    </div>
  );
}

function LogEntry({ message, time }) {
  const meta = iconFor(message);
  const Icon = meta?.icon;
  const cls = meta?.cls ?? "";
  return (
    <div className={`history-entry ${cls}`}>
      {Icon && <Icon size={15} className="history-entry-icon" />}
      <span className="history-entry-text">{cleanMessage(message)}</span>
      <span className="history-entry-time">{time}</span>
    </div>
  );
}

export default function HistoryModal({ onClose, logs, loadingHistory }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Operations History</h3>
          <span className="close-btn" onClick={onClose}>
            <X size={16} />
          </span>
        </div>

        <div className="history-container">
          {loadingHistory ? (
            <div className="history-loading">
              <div className="history-spinner"></div>
              <p>Loading history...</p>
            </div>

          ) : logs.length === 0 ? (
            <p className="empty-history">No operations yet.</p>
          ) : (
            logs.map((log, index) =>
              isSuggestionLog(log.message) ? (
                <SuggestionBlock key={index} message={log.message} time={log.time} />
              ) : (
                <LogEntry key={index} message={log.message} time={log.time} />
              )
            )
          )}
        </div>
      </div>
    </div>
  );
}
