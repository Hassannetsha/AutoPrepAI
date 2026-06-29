import { useState } from "react";

export default function DataPreviewModal({ onClose, data, dataBefore, headers, headersBefore, datasetName }) {
  const [tab, setTab] = useState("after");

  const hasBefore = dataBefore?.length > 0;
  const currentData = tab === "before" ? dataBefore : data;
  const activeHeaders = tab === "before" && headersBefore?.length ? headersBefore : headers;
  const visibleRows = currentData.slice(0, 50);

  if (!hasBefore) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h3>Preview: {datasetName}</h3>
            <span className="close-btn" onClick={onClose}>&times;</span>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  {headers.map((h, i) => <th key={i}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((row, index) => (
                  <tr key={index}>
                    <td>{index + 1}</td>
                    {headers.map((h, i) => (
                      <td key={i}>
                        {row[h] === "" || row[h] === null || row[h] === undefined ? (
                          <span className="missing">missing</span>
                        ) : (
                          String(row[h])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="table-footer">
            Showing 1 to {visibleRows.length} of {currentData.length} rows
          </div>
        </div>
      </div>
    );
  }

  console.log("DataPreviewModal: dataBefore length:", dataBefore.length, "data length:", data.length);
  console.log("DataBefore :", dataBefore);
  console.log("DataAfter :", data);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Preview: {datasetName}</h3>
          <span className="close-btn" onClick={onClose}>&times;</span>
        </div>

        <div className="preview-tabs">
          <button
            className={`preview-tab${tab === "before" ? " active" : ""}`}
            onClick={() => setTab("before")}
          >
            Before Changes
          </button>
          <button
            className={`preview-tab${tab === "after" ? " active" : ""}`}
            onClick={() => setTab("after")}
          >
            After Changes
          </button>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>#</th>
                {activeHeaders.map((h, i) => <th key={i}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  {activeHeaders.map((h, i) => (
                    <td key={i}>
                      {row[h] === "" || row[h] === null || row[h] === undefined ? (
                        <span className="missing">missing</span>
                      ) : (
                        String(row[h])
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="table-footer">
          Showing 1 to {visibleRows.length} of {currentData.length} rows
          {tab === "before" && dataBefore.length > 0 && data.length > 0 && (
            <span> &middot; {dataBefore.length - data.length > 0
              ? `${dataBefore.length - data.length} rows removed`
              : data.length - dataBefore.length > 0
                ? `${data.length - dataBefore.length} rows added`
                : "Row count unchanged"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
