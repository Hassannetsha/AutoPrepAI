export default function ActionList({
  actions,
  uploaded,
  selectedActions,
  toggleAction,
}) {
  return (
    <div className="actions">
      {actions.map((action) => (
        <div
          key={action}
          onClick={() => uploaded && toggleAction(action)}
          className={`action ${selectedActions.includes(action) ? "active" : ""} ${!uploaded ? "disabled" : ""}`}
        >
          {action}
        </div>
      ))}
    </div>
  );
}
