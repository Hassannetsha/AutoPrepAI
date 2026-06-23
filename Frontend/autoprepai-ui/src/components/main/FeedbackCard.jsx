export default function FeedbackCard({ onAccept, onReject, disabled, title }) {
  return (
    <div className="feedback-card">
      <div className="feedback-card-header">NEEDS YOUR APPROVAL</div>
      {title && <div className="feedback-step-title">{title}</div>}
      <div className="feedback-card-actions">
        <button className="feedback-btn-accept" onClick={onAccept} disabled={disabled}>
          ✓ Accept
        </button>
        <button className="feedback-btn-reject" onClick={onReject} disabled={disabled}>
          ✕ Reject
        </button>
      </div>
    </div>
  );
}
