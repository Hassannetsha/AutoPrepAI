import React, { useRef, useState, useEffect } from "react";
import { File, FileJson, FileSpreadsheet } from "lucide-react";
import { useNavigate } from "react-router-dom";
import "../../styles/style.css";
import ActionList from "../../components/main/ActionList";
import AppHeader from "../../components/main/AppHeader";
import ChatInput from "../../components/main/ChatInput";
import ChatSidebar from "../../components/main/ChatSidebar";
import ChatWindow from "../../components/main/ChatWindow";
import DataPreviewModal from "../../components/main/DataPreviewModal";
import DatasetSidebar from "../../components/main/DatasetSidebar";
import HistoryModal from "../../components/main/HistoryModal";

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
  const [showHistory, setShowHistory] = useState(false);
  const [historyLogs, setHistoryLogs] = useState([]);
  // Toggle this to true to show a single demo operation when opening History.
  // Set to false when you want to rely only on the backend API.
  const USE_DEMO_HISTORY = true;

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

  const [chats, setChats] = useState([
    {
      id: 1,
      title: "Chat 1",
      messages: [
        {
          sender: "bot",
          time: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
          text: "Hello! I'm your AutoPrepAI assistant. Upload a dataset to get started.",
          list: [
            "Fix missing values",
            "Detect and handle outliers",
            "Detect and handle duplicates",
            "Resolve feature inconsistency",
            "Scale and encode data",
            "Feature selection",
          ],
        },
      ],
    },
  ]);

  const [activeChatId, setActiveChatId] = useState(1);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Get current active chat
  const activeChat = chats.find((c) => c.id === activeChatId);

  const handleNewChat = () => {
    const newId = Date.now();
    const newChat = {
      id: newId,
      title: `Chat ${chats.length + 1}`,
      messages: [
        {
          sender: "bot",
          time: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
          text: "Hello! Start a new conversation.",
        },
      ],
    };
    setChats((prev) => [...prev, newChat]);
    setActiveChatId(newId);
  };

  const handleRenameChat = (chatId, title) => {
    const trimmedTitle = title.trim();
    if (!trimmedTitle) return;

    setChats((prev) =>
      prev.map((chat) =>
        chat.id === chatId ? { ...chat, title: trimmedTitle } : chat,
      ),
    );
  };

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

  // âœ… Download as CSV
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

  const handleShowHistory = async () => {
    // Production flow: open modal and try to fetch logs from backend.
    setShowHistory(true);
    try {
      const response = await fetch("YOUR_BACKEND_LOGS_ENDPOINT"); // replace with backend endpoint
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setHistoryLogs(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error("Failed to fetch history logs:", err);
      setHistoryLogs([]);
    }
  };
  const handleAutoClean = async () => {
    try {
      const response = await fetch("YOUR_API_ENDPOINT", {
        // <----- replace with backend endpoint
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: tableData }),
      });

      const result = await response.json();

      // update table with cleaned data
      setTableData(result.cleanedData);

      // optional: show message
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "âœ¨ Your data has been automatically cleaned!",
          time: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        },
      ]);
    } catch (error) {
      console.error(error);
    }
  };

  // âœ… Send message
  const handleSend = () => {
    if (!inputValue.trim()) return;

    const time = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    const userMessage = { sender: "user", text: inputValue, time };

    // Add to active chat
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === activeChatId
          ? { ...chat, messages: [...chat.messages, userMessage] }
          : chat,
      ),
    );
    setInputValue("");

    setTimeout(() => {
      const botTime = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      const botMessage = {
        sender: "bot",
        text: "I received your message!",
        time: botTime,
      };

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === activeChatId
            ? { ...chat, messages: [...chat.messages, botMessage] }
            : chat,
        ),
      );
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
      <ChatSidebar
        sidebarCollapsed={sidebarCollapsed}
        setSidebarCollapsed={setSidebarCollapsed}
        handleNewChat={handleNewChat}
        chats={chats}
        activeChatId={activeChatId}
        setActiveChatId={setActiveChatId}
        handleRenameChat={handleRenameChat}
      />

      <DatasetSidebar
        fileInputRef={fileInputRef}
        handleFileUpload={handleFileUpload}
        uploaded={uploaded}
        handleUploadClick={handleUploadClick}
        uploadError={uploadError}
        DatasetIcon={DatasetIcon}
        datasetName={datasetName}
        rows={rows}
        columns={columns}
        setShowPreview={setShowPreview}
        handleDownload={handleDownload}
        handleShowHistory={handleShowHistory}
        handleReset={handleReset}
        handleAutoClean={handleAutoClean}
      />

      <div className="main">
        <AppHeader onLoginClick={() => navigate("/login")} />
        <ChatWindow activeChat={activeChat} chatEndRef={chatEndRef} />
        <ActionList
          actions={actions}
          uploaded={uploaded}
          selectedActions={selectedActions}
          toggleAction={toggleAction}
        />
        <ChatInput
          inputValue={inputValue}
          setInputValue={setInputValue}
          handleSend={handleSend}
        />
      </div>

      {showPreview && (
        <DataPreviewModal
          onClose={() => setShowPreview(false)}
          data={tableData}
          headers={headers}
          datasetName={datasetName}
        />
      )}
      {showHistory && (
        <HistoryModal
          onClose={() => setShowHistory(false)}
          logs={historyLogs}
        />
      )}
    </div>
  );
}
