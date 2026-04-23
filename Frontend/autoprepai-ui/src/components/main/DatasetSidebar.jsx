import {
  Bot,
  Columns3,
  Download,
  Eye,
  Rows3,
  Upload,
  X,
} from "lucide-react";

export default function DatasetSidebar({
  fileInputRef,
  handleFileUpload,
  uploaded,
  handleUploadClick,
  uploadError,
  DatasetIcon,
  datasetName,
  rows,
  columns,
  setShowPreview,
  handleDownload,
  handleShowHistory,
  handleReset,
  handleAutoClean,
}) {
  return (
    <div className="sidebar">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.json"
        onChange={handleFileUpload}
        style={{ display: "none" }}
      />

      {!uploaded ? (
        <div className="uploadBox">
          <p>Upload your dataset to get started</p>
          <button onClick={handleUploadClick}>
            <Upload size={16} /> Upload Dataset
          </button>
          {uploadError && <p className="uploadError">{uploadError}</p>}
        </div>
      ) : (
        <div className="datasetInfo">
          <div className="datasetCard">
            <h3>
              <DatasetIcon size={18} />
              {datasetName}
            </h3>
            <p>
              <Rows3 size={15} /> Rows: {rows}
            </p>
            <p>
              <Columns3 size={15} /> Columns: {columns}
            </p>
          </div>

          <button onClick={() => setShowPreview(true)}>
            <Eye size={16} /> Preview Data
          </button>

          <button onClick={handleDownload}>
            <Download size={16} /> Download Cleaned Data
          </button>

          <button onClick={handleShowHistory}>
            <Rows3 size={16} /> Cleaning History
          </button>
          <button onClick={handleReset} className="reset">
            <X size={16} /> Reset Data
          </button>

          <div className="autoSection">
            <button className="autoCleanBtn" onClick={handleAutoClean}>
              <Bot size={16} /> Automatic Data Cleaning ðŸª„
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
