import { File as FileIcon, FileSpreadsheet } from "lucide-react";

export const cleanError = (msg) => {
  if (!msg) return "Something went wrong. Please try again.";
  if (msg.includes("does not support image") || msg.includes("Cannot read"))
    return "Image files are not supported. Please upload a CSV, or Excel file.";
  if (msg.includes("Failed to parse dataset"))
    return "Could not read the file. Make sure it's a valid CSV, or Excel file.";
  if (msg.includes("Dataset is required"))
    return "No dataset found. Please upload a file first.";
  if (
    msg === "SESSION_EXPIRED" ||
    msg.includes("token") ||
    msg.includes("unauthorized") ||
    msg.includes("expired")
  )
    return "Your session has expired. Please log out and log in again.";
  return msg;
};

export const escapeCSV = (value) =>
  `"${String(value ?? "").replace(/"/g, '""')}"`;

export const getFileIcon = (name) => {
  if (name.endsWith(".csv") || name.endsWith(".xlsx") || name.endsWith(".xls"))
    return FileSpreadsheet;
  return FileIcon;
};
