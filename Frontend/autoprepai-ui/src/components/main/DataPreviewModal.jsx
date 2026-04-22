export default function DataPreviewModal({ onClose, data, headers, datasetName }) {
  const visibleRows = data.slice(0, 50);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Preview: {datasetName}</h3>
          <span className="close-btn" onClick={onClose}>
            âœ•
          </span>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>#</th>
                {headers.map((h, i) => (
                  <th key={i}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  {headers.map((h, i) => (
                    <td key={i}>
                      {row[h] === "" ? (
                        <span className="missing">missing</span>
                      ) : (
                        row[h]
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="table-footer">
          Showing 1â€“{visibleRows.length} of {data.length} rows
        </div>
      </div>
    </div>
  );
}
