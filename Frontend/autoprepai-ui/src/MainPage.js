import React, { useRef, useState, useEffect } from "react";
import {
  Upload,
  FileSpreadsheet,
  X,
  FileJson,
  File,
  Eye,
  Download,
  Rows3,
  Columns3,
  User,
  Database,
  Bot,
} from "lucide-react";
import "./style.css";
import { useNavigate } from "react-router-dom";

export default function MainPage() {
  const [uploaded, setUploaded] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [rows, setRows] = useState(0);
  const [columns, setColumns] = useState(0);
  const [uploadError, setUploadError] = useState("");
  const [tableData, setTableData] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [showPreview, setShowPreview] = useState(false);
  const [selectedActions, setSelectedActions] = useState([]);
  const [inputValue, setInputValue] = useState("");

  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const actions = [
    "Handle Missing Values",
    "Remove Outliers",
    "Remove Duplicates",
    "Detect Feature Inconsistency",
    "Scale Data",
    "Encode Data",
    "Select Features",
  ];

  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hello! I'm your AutoPrepAI assistant. Upload a dataset to get started.",
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
      list: [
        "Fix missing values",
        "Detect and handle outliers",
        "Detect and handle duplicates",
        "Resolve feature inconsistency",
        "Scale and encode data",
        "Feature selection",
      ],
    },
  ]);

  const parseCSV = (text) => {
    const lines = text
      .replace(/\r\n/g, "\n")
      .trim()
      .split("\n")
      .filter(Boolean);
    if (lines.length === 0)
      return { rows: 0, columns: 0, data: [], headers: [] };
    const headers = lines[0].split(",");
    const data = lines.slice(1).map((line) => {
      const values = line.split(",");
      let obj = {};
      headers.forEach((h, i) => {
        obj[h] = values[i] || "";
      });
      return obj;
    });
    return { rows: data.length, columns: headers.length, data, headers };
  };

  const parseJSON = (text) => {
    const parsed = JSON.parse(text);
    const records = Array.isArray(parsed) ? parsed : [parsed];
    if (records.length === 0)
      return { rows: 0, columns: 0, data: [], headers: [] };
    const headers = Object.keys(records[0]);
    return {
      rows: records.length,
      columns: headers.length,
      data: records,
      headers,
    };
  };

  const handleUploadClick = () => fileInputRef.current?.click();

  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    const fileName = selectedFile.name.toLowerCase();
    const isCSV = fileName.endsWith(".csv");
    const isJSON = fileName.endsWith(".json");

    if (!isCSV && !isJSON) {
      setUploadError("Please upload a CSV or JSON file.");
      event.target.value = "";
      return;
    }

    try {
      const text = await selectedFile.text();
      const parsed = isJSON ? parseJSON(text) : parseCSV(text);
      setDatasetName(selectedFile.name);
      setRows(parsed.rows);
      setColumns(parsed.columns);
      setTableData(parsed.data);
      setHeaders(parsed.headers);
      setUploadError("");
      setUploaded(true);
    } catch {
      setUploadError("Could not parse this file.");
    }

    event.target.value = "";
  };

  const handleReset = () => {
    setUploaded(false);
    setDatasetName("");
    setRows(0);
    setColumns(0);
    setTableData([]);
    setHeaders([]);
    setUploadError("");
    setSelectedActions([]);
  };

  // ✅ Download as CSV
  const handleDownload = () => {
    if (!tableData.length) return;
    const csvContent = [
      headers.join(","),
      ...tableData.map((row) => headers.map((h) => row[h]).join(",")),
    ].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cleaned_${datasetName}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ✅ Send message
  const handleSend = () => {
    if (!inputValue.trim()) return;

    const time = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    setMessages((prev) => [
      ...prev,
      { sender: "user", text: inputValue, time },
    ]);
    setInputValue("");

    setTimeout(() => {
      const botTime = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "I received your message! This feature is coming soon.",
          time: botTime,
        },
      ]);
    }, 800);
  };

  const chatEndRef = useRef(null);

  // Auto scroll when messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toggleAction = (action) => {
    setSelectedActions((prev) =>
      prev.includes(action)
        ? prev.filter((a) => a !== action)
        : [...prev, action],
    );
  };

  const getFileIcon = (name) => {
    if (name.endsWith(".json")) return FileJson;
    if (name.endsWith(".csv")) return FileSpreadsheet;
    return File;
  };

  const DatasetIcon = getFileIcon(datasetName);

  return (
    <div className="container">
      {/* ── Sidebar ── */}
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

            <button onClick={handleReset} className="reset">
              <X size={16} /> Reset Data
            </button>
          </div>
        )}
      </div>

      {/* ── Main ── */}
      <div className="main">
        <div className="header">
          <div className="header-brand">
            <div className="header-icon">
              <Database size={22} />
            </div>
            <div>
              <h2>AutoPrepAI</h2>
              <p>AI-Powered Data Cleaning & Preparation</p>
            </div>
          </div>

          <button onClick={() => navigate("/login")}>Login</button>
        </div>

        <div className="chat">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message-row ${msg.sender === "user" ? "row-user" : "row-bot"}`}
            >
              {/* Bot avatar — left */}
              {msg.sender === "bot" && (
                <div className="avatar avatar-bot">
                  <Bot size={16} />
                </div>
              )}

              {/* Bubble */}
              <div
                className={`message ${msg.sender === "user" ? "message-user" : "message-bot"}`}
              >
                <p>{msg.text}</p>
                {msg.list && (
                  <ul>
                    {msg.list.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                )}
                <span className="msg-time">{msg.time}</span>
              </div>

              {/* User avatar — right */}
              {msg.sender === "user" && (
                <div className="avatar avatar-user">
                  <User size={16} />
                </div>
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {/* ── Action Pills ── */}
        <div className="actions">
          {actions.map((action) => (
            <div
              key={action}
              onClick={() => uploaded && toggleAction(action)}
              className={`action ${selectedActions.includes(action) ? "active" : ""} ${!uploaded ? "disabled" : ""}`}
            >
              {action}
            </div>
          ))}
        </div>

        {/* ── Input ── */}
        <div className="inputArea">
          <input
            placeholder="Ask me about your data..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button onClick={handleSend} disabled={!inputValue.trim()}>
            Send
          </button>
        </div>
      </div>

      {/* ── Modal ── */}
      {showPreview && (
        <DataPreviewModal
          onClose={() => setShowPreview(false)}
          data={tableData}
          headers={headers}
          datasetName={datasetName}
        />
      )}
    </div>
  );
}

/* ─── Modal Component ───────────────────────────────── */
function DataPreviewModal({ onClose, data, headers, datasetName }) {
  const visibleRows = data.slice(0, 50);

  return (
    // ✅ Close on outside click
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Preview: {datasetName}</h3>
          <span className="close-btn" onClick={onClose}>
            ✕
          </span>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>#</th>
                {headers.map((h, i) => (
                  <th key={i}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  {headers.map((h, i) => (
                    <td key={i}>
                      {row[h] === "" ? (
                        <span className="missing">missing</span>
                      ) : (
                        row[h]
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="table-footer">
          Showing 1–{visibleRows.length} of {data.length} rows
        </div>
      </div>
    </div>
  );
}
