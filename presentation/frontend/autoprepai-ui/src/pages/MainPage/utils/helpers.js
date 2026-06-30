import { ALLOWED_MIME_TYPES } from "../constants.js";

export const escapeCSV = (value) =>
  `"${String(value ?? "").replace(/"/g, '""')}"`; // double quotes for CSV escaping

export const formatTime = (date) => {
  const d = date ? new Date(date) : new Date();

  if (Number.isNaN(d.getTime())) {
    return "Unknown time";
  }

  return (
    d.toLocaleDateString([], {
      day: "2-digit",
      month: "short",
      year: "numeric",
    }) +
    " " +
    d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })
  );
};

export const updateDatasetFromResponse = (result, setters) => {
  const dataset = result?.dataset;
  if (!Array.isArray(dataset) || dataset.length === 0) {
    // check for empty dataset
    return;
  }
  const newHeaders = Object.keys(dataset[0]);
  const beforeData = result?.data_preview_before ?? [];
  setters.setHeaders(newHeaders);
  setters.setHeadersBefore(
    beforeData.length ? Object.keys(beforeData[0]) : newHeaders,
  );
  setters.setTableData(dataset);
  setters.setTableDataBefore(beforeData);
  setters.setRows(result.shape?.[0] ?? dataset.length);
  setters.setColumns(result.shape?.[1] ?? newHeaders.length);
};

export const userMsg = (text, time) => ({
  role: "user",
  text,
  time,
});

export const botMsg = (text, time, downloadUrl) => ({
  role: "assistant",
  text,
  time,
  ...(downloadUrl ? { downloadUrl } : {}),
});

export const validateDatasetFile = (file) => {
  const fileName = file.name.toLowerCase();

  const isCSV = fileName.endsWith(".csv");
  const isXLSX = fileName.endsWith(".xlsx") || fileName.endsWith(".xls");

  if (!isCSV && !isXLSX) {
    throw new Error(
      `"${file.name}" is not supported. Please upload a CSV or Excel file.`,
    );
  }

  if (file.type && !ALLOWED_MIME_TYPES.includes(file.type)) {
    throw new Error(
      `"${file.name}" appears to be an image or unsupported file. Please upload a CSV or Excel file.`,
    );
  }

  return {
    isCSV,
    extension: isXLSX ? (fileName.endsWith(".xls") ? ".xls" : ".xlsx") : ".csv",
  };
};
