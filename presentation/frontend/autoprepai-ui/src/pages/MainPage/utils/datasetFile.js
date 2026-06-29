import * as XLSX from "xlsx";
import { escapeCSV } from "./helpers";

export function reconstructDatasetFile(headers, data, ext, datasetName = "data") {
  const isExcel = ext === ".xlsx" || ext === ".xls";
  const baseName = datasetName.replace(/\.\w+$/, "");
  const fileName = `${baseName}${isExcel ? ".xlsx" : ".csv"}`;

  if (isExcel) {
    const ws = XLSX.utils.json_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
    const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array" });
    return new File([wbout], fileName, {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
  }

  const csvContent = [
    headers.map(escapeCSV).join(","),
    ...data.map((row) => headers.map((h) => escapeCSV(row[h])).join(",")),
  ].join("\n");

  return new File([csvContent], fileName, { type: "text/csv" });
}

export function openLatestDownload(messages) {
  const last = [...messages].reverse().find((m) => m.downloadUrl);
  if (last) window.open(last.downloadUrl, "_blank");
}
