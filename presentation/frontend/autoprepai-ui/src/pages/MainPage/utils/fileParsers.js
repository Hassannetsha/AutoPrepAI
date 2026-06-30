import * as XLSX from "xlsx";

const makeUniqueHeaders = (headers) => {
  const seen = {};
  return headers.map((header, index) => {
    const name = header?.trim() || `Column_${index + 1}`;

    if (!seen[name]) {
      seen[name] = 1;
      return name;
    }

    seen[name]++;

    return `${name}_${seen[name]}`;
  });
};

export const parseCSVLine = (line = "") => {
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
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        values.push(current);

        current = "";
      } else {
        current += ch;
      }
    }
  }
  values.push(current);
  return values;
};

export const parseCSV = (text = "") => {
  const cleaned = text.replace(/\r\n/g, "\n").trim();
  if (!cleaned) {
    return {
      rows: 0,
      columns: 0,
      data: [],
      headers: [],
    };
  }
  const lines = cleaned.split("\n").filter(Boolean);
  if (lines.length === 0) {
    return {
      rows: 0,
      columns: 0,
      data: [],
      headers: [],
    };
  }
  const headers = makeUniqueHeaders(parseCSVLine(lines[0]));
  const data = lines.slice(1).map((line) => {
    const values = parseCSVLine(line);
    const obj = {};
    headers.forEach((header, index) => {
      obj[header] = values[index] ?? "";
    });
    return obj;
  });

  return {
    rows: data.length,
    columns: headers.length,
    data,
    headers,
  };
};

export const parseXLSX = async (file) => {
  const arrayBuffer = await file.arrayBuffer();

  const workbook = XLSX.read(arrayBuffer, {
    type: "array",
  });

  if (!workbook.SheetNames.length) {
    return {
      rows: 0,
      columns: 0,
      data: [],
      headers: [],
    };
  }

  const sheetName = workbook.SheetNames[0];

  const worksheet = workbook.Sheets[sheetName];

  const allRows = XLSX.utils.sheet_to_json(worksheet, {
    header: 1,
    defval: "",
    raw: true,
  });

  if (allRows.length === 0) {
    return {
      rows: 0,
      columns: 0,
      data: [],
      headers: [],
    };
  }

  let headerIdx = 0;

  while (
    headerIdx < allRows.length &&
    allRows[headerIdx].every(
      (cell) => cell === "" || cell === null || cell === undefined,
    )
  ) {
    headerIdx++;
  }

  if (headerIdx >= allRows.length) {
    return { rows: 0, columns: 0, data: [], headers: [] };
  }

  const rawHeaders = allRows[headerIdx];
  const headers = makeUniqueHeaders(rawHeaders.map((h) => String(h ?? "")));
  const colMap = [];

  rawHeaders.forEach((header, index) => {
    if (header !== "" && header !== null && header !== undefined) {
      colMap.push(index);
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

    colMap.forEach((sourceIndex, headerIndex) => {
      obj[headers[headerIndex]] = row[sourceIndex] ?? "";
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
