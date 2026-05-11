import React from "react";

export default function ActionList({
  actions,
  uploaded,
  selectedActions,
  toggleAction,
  onActionClick,
}) {
  return (
    <div className="actions">
      {actions.map((action) => (
        <div
          key={action}
          onClick={() => {
            if (uploaded) {
              onActionClick ? onActionClick(action) : toggleAction(action);
            }
          }}
          className={`action ${selectedActions.includes(action) ? "active" : ""} ${!uploaded ? "disabled" : ""}`}
        >
          {action}
        </div>
      ))}
    </div>
  );
}
