import React from "react";

export default function ActionList({
  actions,
  uploaded,
  selectedActions,
  onActionClick,
  onApplySelected,
  isLoading,
}) {
  return (
    <div className="actions">
      <div className="actions-left">
        {actions.map((action) => (
          <div
            key={action}
            onClick={() => { if (uploaded && !isLoading) onActionClick(action); }}
            className={`action ${selectedActions.includes(action) ? "active" : ""} ${!uploaded || isLoading ? "disabled" : ""}`}
          >
            {action}
          </div>
        ))}
      </div>

      {selectedActions.length > 0 && (
        <div
          onClick={() => !isLoading && onApplySelected()}
          className={`action apply-action ${isLoading ? "disabled" : ""}`}
        >
          ⚡ Apply {selectedActions.length} Action{selectedActions.length > 1 ? "s" : ""}
        </div>
      )}
    </div>
  );
}
