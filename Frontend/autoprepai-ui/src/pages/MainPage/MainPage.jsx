import { useRef, useState, useEffect, useCallback } from "react";
import { File as FileIcon, FileSpreadsheet } from "lucide-react";
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
import { LOGOUT_EVENT } from "../../components/main/AppHeader";
import {
  sendChatMessage,
  deleteConversation,
  renameConversation,
  getConversation,
  listConversations,
  sendFeedback,
} from "../../api/chat";
import { getAuthToken } from "../../api/auth";
import * as XLSX from "xlsx";

// Map display action labels → backend snake_case intents
const ACTION_TO_INTENT = {
  "Handle Missing Values": "handle_missing_values",
  "Remove Outliers": "remove_outliers",
  "Remove Duplicates": "remove_duplicates",
  "Detect Feature Inconsistency": "remove_inconsistencies",
  "Scale Data": "scale_numerical",
  "Encode Data": "encode_categorical",
  "Feature Engineering": "feature_engineering",
};

const INITIAL_BOT_MESSAGE = {
  sender: "bot",
  text: "Hello! I'm your AutoPrepAI assistant. Upload a dataset to get started.\n\n- Fix missing values\n- Detect and handle outliers\n- Detect and handle duplicates\n- Resolve feature inconsistency\n- Scale and encode data\n- Feature selection with a focus on the target variable\n- Features engineering",
  time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
};

export default function MainPage() {
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
  const [selectedActions, setSelectedActions] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [showHistory, setShowHistory] = useState(false);
  const [historyLogs, setHistoryLogs] = useState([]);

  const cleanError = (msg) => {
    if (!msg) return "Something went wrong. Please try again.";
    if (msg.includes("does not support image") || msg.includes("Cannot read")) return "Image files are not supported. Please upload a CSV, or Excel file.";
    if (msg.includes("Failed to parse dataset")) return "Could not read the file. Make sure it's a valid CSV, or Excel file.";
    if (msg.includes("Dataset is required")) return "No dataset found. Please upload a file first.";
    if (msg === "SESSION_EXPIRED" || msg.includes("token") || msg.includes("unauthorized") || msg.includes("expired")) return "Your session has expired. Please log out and log in again.";
    return msg;
  };

  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [isLoadingChat, setIsLoadingChat] = useState(false);
  const [chatError, setChatError] = useState("");
  const [pendingFeedback, setPendingFeedback] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [stepTitle, setStepTitle] = useState("");

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const navigate = useNavigate();

  const actions = [
    "Handle Missing Values",
    "Remove Outliers",
    "Remove Duplicates",
    "Detect Feature Inconsistency",
    "Scale Data",
    "Encode Data",
    "Feature Engineering",
  ];

  const [chats, setChats] = useState([
    { id: 1, title: "Chat 1", messages: [INITIAL_BOT_MESSAGE] },
  ]);
  const [activeChatId, setActiveChatId] = useState(1);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const activeChat = chats.find((c) => c.id === activeChatId);

  // ─── Restore dataset state from a payload ────────────────────────────────────
  const restoreDatasetFromPayload = useCallback((payload) => {
    if (!payload) return;
    if (payload.dataset_name) {
      setDatasetName(payload.dataset_name);
    }
    if (payload.dataset?.length) {
      const newHeaders = Object.keys(payload.dataset[0]);
      console.log("Restoring dataset from payload:", payload);
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

  // ─── Load messages for a conversation ────────────────────────────────────────
  const loadChatMessages = useCallback(
    async (conversationId) => {
      try {
        const data = await getConversation(conversationId);
        if (!data?.messages) return;

        const mapped = data.messages.length === 0
          ? [{
              ...INITIAL_BOT_MESSAGE,
              time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            }]
          : data.messages.map((msg) => {
              const sender = msg.sender ?? msg.role;
              return {
                sender: sender === "user" ? "user" : "bot",
                text: msg.content,
                time: new Date(msg.created_at).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                }),
                downloadUrl: msg.payload?.download_url ?? null,
              };
            });

        setChats((prev) =>
          prev.map((chat) =>
            chat.id === conversationId ? { ...chat, messages: mapped } : chat
          )
        );

        // Restore dataset name from the upload message (first assistant with dataset_name)
        const uploadMessage = data.messages.find((m) => {
          const sender = m.sender ?? m.role;
          return (sender === "assistant" || sender === "bot") && m.payload?.dataset_name;
        });

        if (uploadMessage?.payload?.dataset_name) {
          setDatasetName(uploadMessage.payload.dataset_name);
        }

        // Restore data preview from last assistant payload
        const lastAssistant = [...data.messages]
          .reverse()
          .find((m) => {
            const sender = m.sender ?? m.role;
            return (sender === "assistant" || sender === "bot") && m.payload;
          });

        if (lastAssistant?.payload) {
          restoreDatasetFromPayload(lastAssistant.payload);
        }

        setPendingFeedback(lastAssistant?.payload?.finished === false);
      } catch (error) {
        console.error("Failed to load messages:", error);
        const friendly = cleanError(error.message);
        if (friendly.includes("log out")) {
          setChatError(friendly);
        }
      }
    },
    [restoreDatasetFromPayload]
  );

  // ─── Load all conversations on mount ─────────────────────────────────────────
  const loadConversations = useCallback(async () => {
    try {
      const data = await listConversations();
      if (data && data.length > 0) {
        const loadedChats = data.map((conv) => ({
          id: conv.id,
          title: conv.title || "Untitled Chat",
          messages: [],
          backendId: conv.id,
        }));
        setChats(loadedChats);
        setActiveChatId(loadedChats[0].id);
        setCurrentConversationId(loadedChats[0].id);
        await loadChatMessages(loadedChats[0].id);
      }
    } catch (error) {
      console.error("Failed to load conversations:", error);
      const friendly = cleanError(error.message);
      if (friendly.includes("log out")) {
        setChatError(friendly);
      }
    }
  }, [loadChatMessages]);

  // ─── Auth check + load conversations on mount ─────────────────────────────────
  useEffect(() => {
    if (!getAuthToken()) {
      return;
    }
    loadConversations();
  }, [navigate, loadConversations]);

  // ─── Reset state on logout ───────────────────────────────────────────────────
  useEffect(() => {
    const handleLogout = () => {
      setUploaded(false);
      setDatasetName("");
      setRows(0);
      setColumns(0);
      setTableData([]);
      setTableDataBefore([]);
      setHeaders([]);
      setHeadersBefore([]);
      setUploadError("");
      setSelectedActions([]);
      setCurrentConversationId(null);
      setHistoryLogs([]);
      setChats([{ id: 1, title: "Chat 1", messages: [INITIAL_BOT_MESSAGE] }]);
      setActiveChatId(1);
    };
    window.addEventListener(LOGOUT_EVENT, handleLogout);
    return () => window.removeEventListener(LOGOUT_EVENT, handleLogout);
  }, []);

  // ─── Auto scroll on new messages ──────────────────────────────────────────────
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

  // ─── Sync backend conversation id into chats list ─────────────────────────────
  // Takes thisChatId as param to avoid stale closure on activeChatId
  const syncConversationId = useCallback(
    (responseConvId, thisChatId) => {
      if (!currentConversationId && responseConvId) {
        setCurrentConversationId(responseConvId);
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === thisChatId
              ? { ...chat, id: responseConvId, backendId: responseConvId }
              : chat
          )
        );
        setActiveChatId(responseConvId);
      }
    },
    [currentConversationId]
  );

  // ─── Switch chat ──────────────────────────────────────────────────────────────
  const handleSwitchChat = useCallback(
    async (chatId) => {
      setActiveChatId(chatId);
      setCurrentConversationId(chatId);
      setUploaded(false);
      setTableData([]);
      setTableDataBefore([]);
      setHeaders([]);
      setHeadersBefore([]);
      setDatasetName("");
      setRows(0);
      setColumns(0);
      setPendingFeedback(false);
      await loadChatMessages(chatId);
    },
    [loadChatMessages]
  );

  // ─── New chat ─────────────────────────────────────────────────────────────────
  const handleNewChat = () => {
    const tempId = Date.now();
    const newChat = {
      id: tempId,
      title: "New Chat",
      messages: [
        {
          ...INITIAL_BOT_MESSAGE,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ],
    };
    setChats((prev) => [...prev, newChat]);
    setActiveChatId(tempId);
    setCurrentConversationId(null);
    setUploaded(false);
    setTableData([]);
    setTableDataBefore([]);
    setHeaders([]);
    setHeadersBefore([]);
    setDatasetName("");
    setRows(0);
    setColumns(0);
    setSelectedActions([]);
    setChatError("");
    setPendingFeedback(false);
  };

  // ─── Rename chat ──────────────────────────────────────────────────────────────
  const handleRenameChat = async (chatId, title) => {
    const trimmedTitle = title.trim();
    if (!trimmedTitle) return;

    setChats((prev) =>
      prev.map((chat) =>
        chat.id === chatId ? { ...chat, title: trimmedTitle } : chat
      )
    );

    // Only persist to backend if this chat has been synced (has a string UUID)
    if (typeof chatId === "string") {
      try {
        await renameConversation(chatId, trimmedTitle);
      } catch (error) {
        console.error("Failed to rename conversation:", error);
        const message = error?.message || "Could not rename conversation";
        setChatError(message);
        if (/token|unauthorized|expired|invalid/i.test(message)) {
          navigate("/login");
        }
      }
    }
  };

  // ─── Delete chat ──────────────────────────────────────────────────────────────
  const handleDeleteChat = async (chatId) => {
    const remaining = chats.filter((c) => c.id !== chatId);
    setChats(remaining);

    if (activeChatId === chatId) {
      if (remaining.length > 0) {
        await handleSwitchChat(remaining[0].id);
      } else {
        handleNewChat();
      }
    }

    try {
      await deleteConversation(chatId);
    } catch (error) {
      console.error("Failed to delete conversation:", error);
    }
  };

  // ─── CSV / Excel parsers ───────────────────────────────────────────────────────
  const parseCSVLine = (line) => {
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

  const parseCSV = (text) => {
    const lines = text.replace(/\r\n/g, "\n").trim().split("\n").filter(Boolean);
    if (lines.length === 0) return { rows: 0, columns: 0, data: [], headers: [] };
    const hdrs = parseCSVLine(lines[0]);
    const data = lines.slice(1).map((line) => {
      const values = parseCSVLine(line);
      const obj = {};
      hdrs.forEach((h, i) => { obj[h] = values[i] ?? ""; });
      return obj;
    });
    return { rows: data.length, columns: hdrs.length, data, headers: hdrs };
  };

  const parseXLSX = async (file) => {
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

  // ─── File upload ──────────────────────────────────────────────────────────────
  const handleUploadClick = () => fileInputRef.current?.click();

  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    const fileName = selectedFile.name.toLowerCase();
    const isCSV = fileName.endsWith(".csv");
    const isXLSX = fileName.endsWith(".xlsx") || fileName.endsWith(".xls");
    const allowedMimeTypes = [
      "text/csv",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/vnd.ms-excel",
    ];

    if (!isCSV && !isXLSX) {
      setUploadError(`"${selectedFile.name}" is not supported. Please upload a CSV, or Excel file.`);
      event.target.value = "";
      return;
    }
    if (selectedFile.type && !allowedMimeTypes.includes(selectedFile.type)) {
      setUploadError(`"${selectedFile.name}" appears to be an image or unsupported file. Please upload a CSV, or Excel file.`);
      event.target.value = "";
      return;
    }

    // Capture ids before any async/state changes
    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;

    setIsLoadingChat(true);
    setChatError("");

    try {
      let parsed;

      parsed = isCSV ? parseCSV(await selectedFile.text())
        : parsed = await parseXLSX(selectedFile);

      setDatasetName(selectedFile.name);
      setRows(parsed.rows);
      setColumns(parsed.columns);
      setTableData(parsed.data);
      setHeaders(parsed.headers);
      setHeadersBefore(parsed.headers);
      setUploadError("");
      setUploaded(true);

      const response = await sendChatMessage({
        message: `I've uploaded a dataset: ${selectedFile.name}. It has ${parsed.rows} rows and ${parsed.columns} columns.`,
        mode: "chat",
        selectedIntents: [],
        conversationId: thisConvId,
        dataset: selectedFile,
      });

      const realConvId = response.conversation_id ?? thisConvId;
      syncConversationId(response.conversation_id, thisChatId);

      const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const userUploadText = `I've uploaded a dataset: ${selectedFile.name}. It has ${parsed.rows} rows and ${parsed.columns} columns.`;
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId || chat.id === realConvId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  { sender: "user", text: userUploadText, time },
                  {
                    sender: "bot",
                    text: response.assistant_message || `✓ Dataset loaded: ${selectedFile.name}`,
                    time,
                    downloadUrl: response.result?.download_url ?? null,
                  },
                ],
              }
            : chat
        )
      );
    } catch (error) {
      console.error("File upload error:", error);
      setChatError(cleanError(error.message));
      setUploadError(cleanError(error.message));
    } finally {
      setIsLoadingChat(false);
      event.target.value = "";
    }
  };

  // ─── Reset ────────────────────────────────────────────────────────────────────
  const handleReset = () => {
    const confirmed = window.confirm('Are you sure you want to reset all uploaded data? This action cannot be undone.');
    if (!confirmed) return;
    setUploaded(false);
    setDatasetName('');
    setRows(0);
    setColumns(0);
    setTableData([]);
    setTableDataBefore([]);
    setHeaders([]);
    setHeadersBefore([]);
    setUploadError('');
    setSelectedActions([]);
  };

  // ─── Download current table as CSV ────────────────────────────────────────────

  const handleDownload = () => {
    const lastWithDownload = activeChat.messages
      .slice()
      .reverse()
      .find((m) => m.downloadUrl);

    if (!lastWithDownload) {
      console.error("No download URL found");
      return;
    }

    // Open pre-signed B2 URL directly – bypasses CORS
    window.open(lastWithDownload.downloadUrl, "_blank");
  };

  // ─── History ──────────────────────────────────────────────────────────────────
  const handleShowHistory = async () => {
    setShowHistory(true);
    setChatError("");
    try {
      let logs = [];
      if (currentConversationId) {
        const data = await getConversation(currentConversationId);
        if (data?.messages) {
          const seen = new Set();
          for (const msg of data.messages) {
            const sender = msg.sender ?? msg.role;
            if ((sender === "assistant" || sender === "bot") && msg.payload?.logs) {
              const time = new Date(msg.created_at).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              });
              for (const log of msg.payload.logs) {
                if (!seen.has(log) && !log.startsWith("Skipping")) {
                  seen.add(log);
                  logs.push({ message: log, time });
                }
              }
            }
          }
        }
      }
      setHistoryLogs(logs);
    } catch (err) {
      console.error("Failed to fetch history logs:", err);
      setChatError("Could not load history");
      setHistoryLogs([]);
    }
  };

  const csvFileName = datasetName
    ? datasetName.replace(/\.\w+$/, ".csv")
    : "data.csv";

  const escapeCSV = (value) =>
    `"${String(value ?? "").replace(/"/g, '""')}"`;

  // ─── Auto clean ───────────────────────────────────────────────────────────────
  const handleAutoClean = async () => {
    if (!uploaded) {
      setChatError("Upload a dataset first");
      return;
    }
    if (isLoadingChat || pendingFeedback) {
      return;
    }

    setChatError("");
    setIsLoadingChat(true);

    // Capture ids before any async/state changes
    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === thisChatId
          ? {
              ...chat,
              messages: [
                ...chat.messages,
                { sender: "user", text: "🤖 Auto-clean my dataset", time },
              ],
            }
          : chat
      )
    );

    try {
      const file = new File(
        [
          [
            headers.map(escapeCSV).join(","),
            ...tableData.map((row) =>
              headers.map((h) => escapeCSV(row[h])).join(",")
            ),
          ].join("\n"),
        ],
        csvFileName,
        { type: "text/csv" }
      );

      const response = await sendChatMessage({
        message: "🤖 Auto-clean my dataset",
        mode: "full_auto",
        selectedIntents: [],
        conversationId: thisConvId,
        dataset: file,
      });

      // Capture real conv id BEFORE syncing (sync mutates activeChatId)
      const realConvId = response.conversation_id ?? thisConvId;
      syncConversationId(response.conversation_id, thisChatId);

      if (response.result?.dataset?.length) {
        const newHeaders = Object.keys(response.result.dataset[0]);
        setHeaders(newHeaders);
        const beforeData = response.result.data_preview_before ?? [];
        const beforeHeaders = beforeData.length ? Object.keys(beforeData[0]) : newHeaders;
        setHeadersBefore(beforeHeaders);
        setTableData(response.result.dataset);
        setTableDataBefore(beforeData);
        setRows(response.result.shape?.[0] ?? response.result.dataset.length);
        setColumns(response.result.shape?.[1] ?? newHeaders.length);
      }

       const botTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

       setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId || chat.id === realConvId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  {
                    sender: "bot",
                    text: response.assistant_message || "✨ Auto-clean completed successfully.",
                    time: botTime,
                    downloadUrl: response.result?.download_url ?? null,
                  },
                ],
              }
            : chat
        ));
    } catch (error) {
      console.error(error);
      const friendlyMsg = cleanError(error.message);
      setChatError(friendlyMsg);
      const botTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  { sender: "bot", text: `❌ ${friendlyMsg}`, time: botTime },
                ],
              }
            : chat
        )
      );
    } finally {
      setIsLoadingChat(false);
    }
  };

  // ─── Send message ─────────────────────────────────────────────────────────────
  const handleSend = async () => {
    if (!inputValue.trim() && selectedActions.length === 0) return;

    // Capture ids before any async/state changes
    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    const messageText = inputValue.trim();

    let displayedMessage = messageText;

    if (selectedActions.length > 0) {
      displayedMessage +=
        (displayedMessage ? "\n\n" : "") +
        `Please apply these actions: ${selectedActions.join(", ")}`;
    }

    const userMessage = {
      sender: "user",
      text: displayedMessage,
      time,
    };

    setChats((prev) =>
      prev.map((chat) =>
        chat.id === thisChatId
          ? { ...chat, messages: [...chat.messages, userMessage] }
          : chat
      )
    );

    setInputValue("");
    setChatError("");
    setIsLoadingChat(true);

    try {
      let datasetFile = null;

      const sourceData = tableData;

      if (sourceData.length > 0 && headers.length > 0) {
        const csvContent = [
          headers.map(escapeCSV).join(","),
          ...sourceData.map((row) =>
            headers.map((h) => escapeCSV(row[h])).join(",")
          ),
        ].join("\n");

        datasetFile = new File([csvContent], csvFileName, { type: "text/csv" });
      }

      const backendMessage =`${messageText}\n\nPlease apply these actions: ${selectedActions.join(", ")}`

      const response = await sendChatMessage({
        message: backendMessage,
        mode: inputValue.trim() === "" && selectedActions.length > 0 ? "manual" : "chat",
        selectedIntents: selectedActions.map((a) => ACTION_TO_INTENT[a] || a),
        conversationId: thisConvId,
        dataset: datasetFile,
      });

      const realConvId = response.conversation_id ?? thisConvId;
      syncConversationId(response.conversation_id, thisChatId);

      const botTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      setPendingFeedback(!response.finished);
      setStepTitle(response.step_title || "");
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId || chat.id === realConvId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  {
                    sender: "bot",
                    text: response.assistant_message || "I received your message!",
                    time: botTime,
                    downloadUrl: response.result?.download_url ?? null,
                  },
                ],
              }
            : chat
        )
      );

      if (response.result?.dataset?.length) {
        const newHeaders = Object.keys(response.result.dataset[0]);
        setHeaders(newHeaders);
        const beforeData = response.result.data_preview_before ?? [];
        const beforeHeaders = beforeData.length ? Object.keys(beforeData[0]) : newHeaders;
        setHeadersBefore(beforeHeaders);
        setTableData(response.result.dataset);
        setTableDataBefore(beforeData);
        setRows(response.result.shape?.[0] ?? response.result.dataset.length);
        setColumns(response.result.shape?.[1] ?? newHeaders.length);
      }
      setSelectedActions([]);
    } catch (error) {
      console.error("Chat error:", error);
      setChatError(cleanError(error.message));
      const botTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  { sender: "bot", text: `Error: ${error.message}`, time: botTime },
                ],
              }
            : chat
        )
      );
    } finally {
      setIsLoadingChat(false);
    }
  };


  // ─── Handle feedback (accept/reject) ──────────────────────────────────────────
  const handleFeedback = async (accept) => {
    if (!currentConversationId || submittingFeedback) return;

    setSubmittingFeedback(true);
    setPendingFeedback(false);
    setChatError("");

    try {
      const response = await sendFeedback({
        conversationId: currentConversationId,
        accept,
      });

      const botTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === currentConversationId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  {
                    sender: "bot",
                    text: response.assistant_message,
                    time: botTime,
                    downloadUrl: response.result?.download_url ?? null,
                  },
                ],
              }
            : chat
        )
      );

      setPendingFeedback(!response.finished);
      setStepTitle(response.step_title || "");

      if (response.result?.dataset?.length) {
        const newHeaders = Object.keys(response.result.dataset[0]);
        setHeaders(newHeaders);
        const beforeData = response.result.data_preview_before ?? [];
        const beforeHeaders = beforeData.length ? Object.keys(beforeData[0]) : newHeaders;
        setHeadersBefore(beforeHeaders);
        setTableData(response.result.dataset);
        setTableDataBefore(beforeData);
        setRows(response.result.shape?.[0] ?? response.result.dataset.length);
        setColumns(response.result.shape?.[1] ?? newHeaders.length);
      }
    } catch (error) {
      console.error("Feedback error:", error);
      setChatError(cleanError(error.message));
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const toggleAction = (action) => {
    setSelectedActions((prev) =>
      prev.includes(action) ? prev.filter((a) => a !== action) : [...prev, action]
    );
  };

  const getFileIcon = (name) => {
    if (name.endsWith(".csv") || name.endsWith(".xlsx") || name.endsWith(".xls")) return FileSpreadsheet;
    return FileIcon;
  };

  const DatasetIcon = getFileIcon(datasetName);
  const canSend = uploaded && (inputValue.trim() !== "" || selectedActions.length > 0);

  return (
    <div className="container">
      <ChatSidebar
        sidebarCollapsed={sidebarCollapsed}
        setSidebarCollapsed={setSidebarCollapsed}
        handleNewChat={handleNewChat}
        chats={chats}
        activeChatId={activeChatId}
        setActiveChatId={handleSwitchChat}
        handleRenameChat={handleRenameChat}
        handleDeleteChat={handleDeleteChat}
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
        autoCleanDisabled={!uploaded || isLoadingChat || pendingFeedback}
        pendingFeedback={pendingFeedback}
      />

      <div className="main">
        <AppHeader />
        <ChatWindow
          activeChat={activeChat}
          chatEndRef={chatEndRef}
          pendingFeedback={pendingFeedback}
          onAcceptFeedback={() => handleFeedback(true)}
          onRejectFeedback={() => handleFeedback(false)}
          feedbackDisabled={submittingFeedback}
          stepTitle={stepTitle}
        />
        <ActionList
          actions={actions}
          uploaded={uploaded}
          selectedActions={selectedActions}
          toggleAction={toggleAction}
          onActionClick={toggleAction}
          isLoading={isLoadingChat || pendingFeedback}
        />
        {chatError && (
          <div className="chat-error">
            {chatError}
          </div>
        )}
        {isLoadingChat && (
          <div className="chat-loading">
            Processing...
          </div>
        )}
        <ChatInput
          inputValue={inputValue}
          setInputValue={setInputValue}
          handleSend={handleSend}
          canSend={canSend}
          isLoading={isLoadingChat || pendingFeedback }
        />
      </div>

      {showPreview && (
        <DataPreviewModal
          onClose={() => setShowPreview(false)}
          data={tableData}
          dataBefore={tableDataBefore}
          headers={headers}
          headersBefore={headersBefore}
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