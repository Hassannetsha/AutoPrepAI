import React from "react";

export default function ActionList({
  actions,
  uploaded,
  selectedActions,
  onActionClick,
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
    </div>
  );
}
