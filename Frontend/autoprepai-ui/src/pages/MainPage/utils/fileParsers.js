import * as XLSX from "xlsx";

export const parseCSVLine = (line) => {
  const values = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < line.length && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      values.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  values.push(current);
  return values;
};

export const parseCSV = (text) => {
  const lines = text.replace(/\r\n/g, "\n").trim().split("\n").filter(Boolean);
  if (lines.length === 0) return { rows: 0, columns: 0, data: [], headers: [] };
  const hdrs = parseCSVLine(lines[0]);
  const data = lines.slice(1).map((line) => {
    const values = parseCSVLine(line);
    const obj = {};
    hdrs.forEach((h, i) => {
      obj[h] = values[i] ?? "";
    });
    return obj;
  });
  return { rows: data.length, columns: hdrs.length, data, headers: hdrs };
};

export const parseXLSX = async (file) => {
  const arrayBuffer = await file.arrayBuffer();

  const workbook = XLSX.read(arrayBuffer, {
    type: "array",
  });

  const sheetName = workbook.SheetNames[0];

  const worksheet = workbook.Sheets[sheetName];

  const allRows = XLSX.utils.sheet_to_json(worksheet, {
    header: 1,
    defval: "",
    raw: false,
  });

  if (allRows.length === 0) {
    return { rows: 0, columns: 0, data: [], headers: [] };
  }

  let headerIdx = 0;
  while (
    headerIdx < allRows.length &&
    allRows[headerIdx].every(
      (cell) => cell === "" || cell === null || cell === undefined
    )
  ) {
    headerIdx++;
  }

  if (headerIdx >= allRows.length) {
    return { rows: 0, columns: 0, data: [], headers: [] };
  }

  const raw = allRows[headerIdx];
  const colMap = [];
  const headers = [];
  raw.forEach((h, i) => {
    if (h !== "" && h !== null && h !== undefined) {
      headers.push(String(h).trim());
      colMap.push(i);
    }
  });

  const data = [];
  for (let r = headerIdx + 1; r < allRows.length; r++) {
    const row = allRows[r];
    if (
      row.every((cell) => cell === "" || cell === null || cell === undefined)
    ) {
      continue;
    }
    const obj = {};
    colMap.forEach((srcIdx, destIdx) => {
      obj[headers[destIdx]] = row[srcIdx] ?? "";
    });
    data.push(obj);
  }

  return {
    rows: data.length,
    columns: headers.length,
    data,
    headers,
  };
};
