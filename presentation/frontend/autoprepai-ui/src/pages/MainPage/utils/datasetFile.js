import * as XLSX from "xlsx";
import { escapeCSV } from "./helpers";

export function reconstructDatasetFile(
  headers = [],
  data = [],
  ext,
  datasetName = "data",
) {
  const extension = ext?.toLowerCase();
  const isExcel = extension === ".xlsx" || extension === ".xls";
  const baseName = (datasetName || "data").replace(/\.\w+$/, "");
  const fileName = `${baseName}${isExcel ? ".xlsx" : ".csv"}`;

  if (isExcel) {
    const ws = XLSX.utils.json_to_sheet(data, {
      header: headers,
    });
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
    const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array" });
    return new File([wbout], fileName, {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
  }

  // For CSV, we need to escape values and join them with commas
  const csvContent = [
    headers.map(escapeCSV).join(","),
    ...data.map((row) => headers.map((h) => escapeCSV(row[h])).join(",")),
  ].join("\r\n");

  return new File([csvContent], fileName, { type: "text/csv" });
}
