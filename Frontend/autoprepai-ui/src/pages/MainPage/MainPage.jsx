import React, { useRef, useState, useEffect, useCallback } from "react";
import { File as FileIcon, FileJson, FileSpreadsheet } from "lucide-react";
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
import {
  sendChatMessage,
  deleteConversation,
  renameConversation,
  getConversation,
  listConversations,
} from "../../api/chat";
import { getAuthToken } from "../../api/auth";

// Map display action labels → backend snake_case intents
const ACTION_TO_INTENT = {
  "Handle Missing Values": "handle_missing_values",
  "Remove Outliers": "remove_outliers",
  "Remove Duplicates": "remove_duplicates",
  "Detect Feature Inconsistency": "remove_inconsistencies",
  "Scale Data": "scale_numerical",
  "Encode Data": "encode_categorical",
  "Select Features": "select_features",
};

const INITIAL_BOT_MESSAGE = {
  sender: "bot",
  text: "Hello! I'm your AutoPrepAI assistant. Upload a dataset to get started.",
  time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  list: [
    "Fix missing values",
    "Detect and handle outliers",
    "Detect and handle duplicates",
    "Resolve feature inconsistency",
    "Scale and encode data",
    "Feature selection",
  ],
};

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
  const [showHistory, setShowHistory] = useState(false);
  const [historyLogs, setHistoryLogs] = useState([]);

  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [isLoadingChat, setIsLoadingChat] = useState(false);
  const [chatError, setChatError] = useState("");

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
    "Select Features",
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
    if (payload.data_preview?.length) {
      const newHeaders = Object.keys(payload.data_preview[0]);
      setHeaders(newHeaders);
      setTableData(payload.data_preview);
      setRows(payload.shape?.[0] ?? payload.data_preview.length);
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

        const mapped = data.messages.map((msg) => {
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

        // Restore dataset from last assistant payload
        const lastAssistant = [...data.messages]
          .reverse()
          .find((m) => {
            const sender = m.sender ?? m.role;
            return (sender === "assistant" || sender === "bot") && m.payload;
          });

        if (lastAssistant?.payload) {
          restoreDatasetFromPayload(lastAssistant.payload);
        }
      } catch (error) {
        console.error("Failed to load messages:", error);
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
    }
  }, [loadChatMessages]);

  // ─── Auth check + load conversations on mount ─────────────────────────────────
  useEffect(() => {
    if (!getAuthToken()) {
      navigate("/login");
      return;
    }
    loadConversations();
  }, [navigate, loadConversations]);

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
      setHeaders([]);
      setDatasetName("");
      setRows(0);
      setColumns(0);
      await loadChatMessages(chatId);
    },
    [loadChatMessages]
  );

  // ─── New chat ─────────────────────────────────────────────────────────────────
  const handleNewChat = () => {
    const tempId = Date.now();
    const newChat = {
      id: tempId,
      title: `Chat ${chats.length + 1}`,
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
    setHeaders([]);
    setDatasetName("");
    setRows(0);
    setColumns(0);
    setSelectedActions([]);
    setChatError("");
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

    if (currentConversationId && chatId === activeChatId) {
      try {
        await renameConversation(currentConversationId, trimmedTitle);
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

  // ─── CSV / JSON parsers ───────────────────────────────────────────────────────
  const parseCSV = (text) => {
    const lines = text.replace(/\r\n/g, "\n").trim().split("\n").filter(Boolean);
    if (lines.length === 0) return { rows: 0, columns: 0, data: [], headers: [] };
    const hdrs = lines[0].split(",");
    const data = lines.slice(1).map((line) => {
      const values = line.split(",");
      const obj = {};
      hdrs.forEach((h, i) => { obj[h] = values[i] || ""; });
      return obj;
    });
    return { rows: data.length, columns: hdrs.length, data, headers: hdrs };
  };

  const parseJSON = (text) => {
    const parsed = JSON.parse(text);
    const records = Array.isArray(parsed) ? parsed : [parsed];
    if (records.length === 0) return { rows: 0, columns: 0, data: [], headers: [] };
    const hdrs = Object.keys(records[0]);
    return { rows: records.length, columns: hdrs.length, data: records, headers: hdrs };
  };

  // ─── File upload ──────────────────────────────────────────────────────────────
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

    // ✅ Capture ids before any async/state changes
    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;

    setIsLoadingChat(true);
    setChatError("");

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
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId || chat.id === realConvId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  {
                    sender: "bot",
                    text: response.assistant_message || `✓ Dataset loaded: ${selectedFile.name}`,
                    time,
                  },
                ],
              }
            : chat
        )
      );
    } catch (error) {
      console.error("File upload error:", error);
      setChatError(error.message);
      setUploadError(error.message || "Could not upload file.");
    } finally {
      setIsLoadingChat(false);
      event.target.value = "";
    }
  };

  // ─── Reset ────────────────────────────────────────────────────────────────────
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

  // ─── Download current table as CSV ────────────────────────────────────────────
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

  // ─── History ──────────────────────────────────────────────────────────────────
  const handleShowHistory = async () => {
    setShowHistory(true);
    setChatError("");
    try {
      const data = await listConversations();
      setHistoryLogs(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error("Failed to fetch history logs:", err);
      setChatError("Could not load history");
      setHistoryLogs([]);
    }
  };

  // ─── Auto clean ───────────────────────────────────────────────────────────────
  const handleAutoClean = async () => {
    if (!tableData.length) {
      setChatError("No data to clean");
      return;
    }

    setChatError("");
    setIsLoadingChat(true);

    // ✅ Capture ids before any async/state changes
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
      const csvContent = [
        headers.join(","),
        ...tableData.map((row) => headers.map((h) => row[h]).join(",")),
      ].join("\n");
      const file = new File([csvContent], datasetName || "data.csv", { type: "text/csv" });

      const response = await sendChatMessage({
        message: "🤖 Auto-clean my dataset",
        mode: "full_auto",
        selectedIntents: [],
        conversationId: thisConvId,
        dataset: file,
      });

      // ✅ Capture real conv id BEFORE syncing (sync mutates activeChatId)
      const realConvId = response.conversation_id ?? thisConvId;
      syncConversationId(response.conversation_id, thisChatId);

      if (response.result?.data_preview?.length) {
        const newHeaders = Object.keys(response.result.data_preview[0]);
        setHeaders(newHeaders);
        setTableData(response.result.data_preview);
        setRows(response.result.shape?.[0] ?? response.result.data_preview.length);
        setColumns(response.result.shape?.[1] ?? newHeaders.length);
      }

      const logs = response.result?.logs ?? [];
      const logSummary = logs.length ? logs.join("\n") : "Auto-clean completed successfully.";
      const botTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

      // ✅ Match on either old temp id or new backend id
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId || chat.id === realConvId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  {
                    sender: "bot",
                    text: `✨ Auto-clean done!\n\n${logSummary}`,
                    time: botTime,
                    downloadUrl: response.result?.download_url ?? null,
                  },
                ],
              }
            : chat
        )
      );
    } catch (error) {
      console.error(error);
      setChatError(error.message);
      const botTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  { sender: "bot", text: `❌ Auto-clean failed: ${error.message}`, time: botTime },
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
    if (!inputValue.trim()) return;

    // ✅ Capture ids before any async/state changes
    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const userMessage = { sender: "user", text: inputValue, time };

    setChats((prev) =>
      prev.map((chat) =>
        chat.id === thisChatId
          ? { ...chat, messages: [...chat.messages, userMessage] }
          : chat
      )
    );

    const messageText = inputValue;
    setInputValue("");
    setChatError("");
    setIsLoadingChat(true);

    try {
      let datasetFile = null;
      if (tableData.length > 0 && headers.length > 0) {
        const csvContent = [
          headers.join(","),
          ...tableData.map((row) => headers.map((h) => row[h]).join(",")),
        ].join("\n");
        datasetFile = new File([csvContent], datasetName || "data.csv", { type: "text/csv" });
      }

      const response = await sendChatMessage({
        message: messageText,
        mode: "chat",
        selectedIntents: selectedActions.map((a) => ACTION_TO_INTENT[a] || a),
        conversationId: thisConvId,
        dataset: datasetFile,
      });

      const realConvId = response.conversation_id ?? thisConvId;
      syncConversationId(response.conversation_id, thisChatId);

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
                    text: response.assistant_message || "I received your message!",
                    time: botTime,
                    downloadUrl: response.result?.download_url ?? null,
                  },
                ],
              }
            : chat
        )
      );

      if (response.result?.data_preview?.length) {
        const newHeaders = Object.keys(response.result.data_preview[0]);
        setHeaders(newHeaders);
        setTableData(response.result.data_preview);
        setRows(response.result.shape?.[0] ?? response.result.data_preview.length);
        setColumns(response.result.shape?.[1] ?? newHeaders.length);
      }
    } catch (error) {
      console.error("Chat error:", error);
      setChatError(error.message);
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

  // ─── Action button click ──────────────────────────────────────────────────────
  const handleActionClick = async (action) => {
    const isRemoving = selectedActions.includes(action);
    const newSelectedActions = isRemoving
      ? selectedActions.filter((a) => a !== action)
      : [...selectedActions, action];
    setSelectedActions(newSelectedActions);

    if (isRemoving || !tableData.length || !headers.length) return;

    // ✅ Capture ids before any async/state changes
    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;

    setChatError("");
    setIsLoadingChat(true);

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === thisChatId
          ? {
              ...chat,
              messages: [
                ...chat.messages,
                { sender: "user", text: `Please ${action}`, time },
              ],
            }
          : chat
      )
    );

    try {
      const csvContent = [
        headers.join(","),
        ...tableData.map((row) => headers.map((h) => row[h]).join(",")),
      ].join("\n");
      const datasetFile = new File([csvContent], datasetName || "data.csv", { type: "text/csv" });

      const response = await sendChatMessage({
        message: `Please ${action.toLowerCase()}`,
        mode: "manual",
        selectedIntents: [ACTION_TO_INTENT[action]],
        conversationId: thisConvId,
        dataset: datasetFile,
      });

      const realConvId = response.conversation_id ?? thisConvId;
      syncConversationId(response.conversation_id, thisChatId);

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
                    text: response.assistant_message || `✓ ${action} applied`,
                    time: botTime,
                    downloadUrl: response.result?.download_url ?? null,
                  },
                ],
              }
            : chat
        )
      );

      if (response.result?.data_preview?.length) {
        const newHeaders = Object.keys(response.result.data_preview[0]);
        setHeaders(newHeaders);
        setTableData(response.result.data_preview);
        setRows(response.result.shape?.[0] ?? response.result.data_preview.length);
        setColumns(response.result.shape?.[1] ?? newHeaders.length);
      }
    } catch (error) {
      console.error("Action error:", error);
      setChatError(error.message);
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

  const toggleAction = (action) => {
    setSelectedActions((prev) =>
      prev.includes(action) ? prev.filter((a) => a !== action) : [...prev, action]
    );
  };

  const getFileIcon = (name) => {
    if (name.endsWith(".json")) return FileJson;
    if (name.endsWith(".csv")) return FileSpreadsheet;
    return FileIcon;
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
      />

      <div className="main">
        <AppHeader onLoginClick={() => navigate("/login")} />
        <ChatWindow activeChat={activeChat} chatEndRef={chatEndRef} />
        <ActionList
          actions={actions}
          uploaded={uploaded}
          selectedActions={selectedActions}
          toggleAction={toggleAction}
          onActionClick={handleActionClick}
        />
        {chatError && (
          <div
            className="chat-error"
            style={{ color: "red", padding: "4px 12px", fontSize: "13px" }}
          >
            ⚠️ {chatError}
          </div>
        )}
        {isLoadingChat && (
          <div
            className="chat-loading"
            style={{ padding: "4px 12px", fontSize: "13px", color: "#888" }}
          >
            ⏳ Processing...
          </div>
        )}
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