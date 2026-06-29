import { X } from "lucide-react";

function getLogClass(message) {
  if (message.startsWith("[System] User ACCEPTED")) return "history-item history-accept";
  if (message.startsWith("[System] User REJECTED")) return "history-item history-reject";
  if (message.startsWith("[System]")) return "history-item history-system";
  if (message.includes("Executing agent")) return "history-item history-agent";
  return "history-item";
}

function formatMessage(message) {
  return message.replace(/^\[System\] /, "");
}

export default function HistoryModal({ onClose, logs }) {
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
          {logs.length === 0 ? (
            <p className="empty-history">No operations yet.</p>
          ) : (
            logs.map((log, index) => (
              <div key={index} className={getLogClass(log.message)}>
                <p className="history-text">{formatMessage(log.message)}</p>
                <span className="history-time">{log.time}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
