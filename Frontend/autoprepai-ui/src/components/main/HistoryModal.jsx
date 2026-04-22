export default function HistoryModal({ onClose, logs }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Operations History</h3>
          <span className="close-btn" onClick={onClose}>
            âœ•
          </span>
        </div>

        <div className="history-container">
          {logs.length === 0 ? (
            <p className="empty-history">No operations yet.</p>
          ) : (
            logs.map((log, index) => (
              <div key={index} className="history-item">
                <p className="history-text">{log.message}</p>
                <span className="history-time">{log.time}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
