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
        {actions.map((action) => {
          const disabled = !uploaded || isLoading;
          return (
            <div
              key={action}
              onClick={() => { if (!disabled) onActionClick(action); }}
              onKeyDown={(e) => { if (!disabled && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onActionClick(action); } }}
              role="button"
              tabIndex={disabled ? -1 : 0}
              aria-disabled={disabled}
              className={`action ${selectedActions.includes(action) ? "active" : ""} ${disabled ? "disabled" : ""}`}
              style={{ pointerEvents: disabled ? 'none' : 'auto', opacity: disabled ? 0.6 : 1 }}
            >
              {action}
            </div>
          );
        })}
      </div>
    </div>
  );
}
