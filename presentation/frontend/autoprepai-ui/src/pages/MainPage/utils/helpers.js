

export const cleanError = (msg) => {
  if (!msg) return "Something went wrong. Please try again.";

  const error = msg.toLowerCase();

  if ( error.includes("does not support image") ||error.includes("cannot read") ) {
    return "Image files are not supported. Please upload a CSV, or Excel file.";
  }

  if (error.includes("failed to parse dataset")) {
    return "Could not read the file. Make sure it's a valid CSV, or Excel file.";
  }

  if (error.includes("dataset is required")) {
    return "No dataset found. Please upload a file first.";
  }

  if (
    error === "session_expired" ||
    error.includes("token") ||
    error.includes("unauthorized") ||
    error.includes("expired")
  ) {
    return "Your session has expired. Please log out and log in again.";
  }

  return msg;
};


export const escapeCSV = (value) =>
  `"${String(value ?? "").replace(/"/g, '""')}"`; // double quotes for CSV escaping

export const formatTime = (date) => {
  const d = date ? new Date(date) : new Date();

  if (isNaN(d.getTime())) {
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
