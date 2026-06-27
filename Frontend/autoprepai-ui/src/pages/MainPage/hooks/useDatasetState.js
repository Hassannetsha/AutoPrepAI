import { useState, useRef, useCallback } from "react";
import { getFileIcon } from "../utils/helpers";

export function useDatasetState() {
  const [uploaded, setUploaded] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [rows, setRows] = useState(0);
  const [columns, setColumns] = useState(0);
  const [uploadError, setUploadError] = useState("");
  const [tableData, setTableData] = useState([]);
  const [tableDataBefore, setTableDataBefore] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [headersBefore, setHeadersBefore] = useState([]);
  const [showPreview, setShowPreview] = useState(false);
  const fileInputRef = useRef(null);

  const handleUploadClick = () => fileInputRef.current?.click();

  const restoreDatasetFromPayload = useCallback((payload) => {
    if (!payload) return;
    if (payload.dataset_name) {
      setDatasetName(payload.dataset_name);
    }
    if (payload.dataset?.length) {
      const newHeaders = Object.keys(payload.dataset[0]);
      setHeaders(newHeaders);
      const beforeData = payload.data_preview_before ?? [];
      const beforeHeaders = beforeData.length ? Object.keys(beforeData[0]) : newHeaders;
      setHeadersBefore(beforeHeaders);
      setTableData(payload.dataset);
      setTableDataBefore(beforeData);
      setRows(payload.shape?.[0] ?? payload.dataset.length);
      setColumns(payload.shape?.[1] ?? newHeaders.length);
      setUploaded(true);
    }
  }, []);

  const handleReset = () => {
    const confirmed = window.confirm("Are you sure you want to reset all uploaded data? This action cannot be undone.");
    if (!confirmed) return;
    setUploaded(false);
    setDatasetName("");
    setRows(0);
    setColumns(0);
    setTableData([]);
    setTableDataBefore([]);
    setHeaders([]);
    setHeadersBefore([]);
    setUploadError("");
  };

  const csvFileName = datasetName
    ? datasetName.replace(/\.\w+$/, ".csv")
    : "data.csv";

  const DatasetIcon = getFileIcon(datasetName);

  return {
    uploaded, setUploaded,
    datasetName, setDatasetName,
    rows, setRows,
    columns, setColumns,
    uploadError, setUploadError,
    tableData, setTableData,
    tableDataBefore, setTableDataBefore,
    headers, setHeaders,
    headersBefore, setHeadersBefore,
    showPreview, setShowPreview,
    fileInputRef,
    handleUploadClick,
    handleReset,
    restoreDatasetFromPayload,
    csvFileName,
    DatasetIcon,
  };
}
