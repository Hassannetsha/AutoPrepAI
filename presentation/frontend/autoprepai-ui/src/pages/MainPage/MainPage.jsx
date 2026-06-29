import { useRef, useState, useEffect, useCallback } from "react";
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
  createConversation,
  sendFeedback,
} from "../../api/chat";
import { getAuthToken } from "../../api/auth";
import { ACTION_TO_INTENT, ACTIONS, ALLOWED_MIME_TYPES } from "./constants";
import { parseCSV, parseXLSX } from "./utils/fileParsers";
import * as XLSX from "xlsx";
import { cleanError, escapeCSV } from "./utils/helpers";
import { useChatManager } from "./hooks/useChatManager";
import { useDatasetState } from "./hooks/useDatasetState";

export default function MainPage() {
  const {
    chats, setChats, activeChat, activeChatId, setActiveChatId,
    currentConversationId, setCurrentConversationId,
    sidebarCollapsed, setSidebarCollapsed,
    syncConversationId,
  } = useChatManager();

  const {
    uploaded, setUploaded, datasetName, setDatasetName,
    rows, setRows, columns, setColumns,
    uploadError, setUploadError,
    tableData, setTableData, tableDataBefore, setTableDataBefore,
    headers, setHeaders, headersBefore, setHeadersBefore,
    showPreview, setShowPreview,
    originalExtension, setOriginalExtension,
    fileInputRef, handleUploadClick, handleReset,
    restoreDatasetFromPayload,
    DatasetIcon,
  } = useDatasetState();

  const handleResetWithBackend = async () => {
    if (currentConversationId && tableDataBefore.length > 0) {
      try {
        await sendFeedback({ conversationId: currentConversationId, accept: false });
      } catch {
        // backend session may not exist — still reset frontend
      }
    }
    handleReset();
  };

  const [selectedActions, setSelectedActions] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [showHistory, setShowHistory] = useState(false);
  const [historyLogs, setHistoryLogs] = useState([]);

  const [isLoadingChat, setIsLoadingChat] = useState(false);
  const [chatError, setChatError] = useState("");
  const [pendingFeedback, setPendingFeedback] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [stepTitle, setStepTitle] = useState("");

  const chatEndRef = useRef(null);
  const navigate = useNavigate();

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

        const uploadMessage = data.messages.find((m) => {
          const sender = m.sender ?? m.role;
          return (sender === "assistant" || sender === "bot") && m.payload?.dataset_name;
        });

        if (uploadMessage?.payload?.dataset_name) {
          setDatasetName(uploadMessage.payload.dataset_name);
          const ext = uploadMessage.payload.dataset_name.match(/\.\w+$/)?.[0]?.toLowerCase() || ".csv";
          setOriginalExtension(ext);
        }

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
        setStepTitle(lastAssistant?.payload?.step_title ?? "");
      } catch (error) {
        console.error("Failed to load messages:", error);
        const friendly = cleanError(error.message);
        if (friendly.includes("log out")) {
          setChatError(friendly);
        }
      }
    },
    [restoreDatasetFromPayload, setChats, setDatasetName, setOriginalExtension]
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
  }, [loadChatMessages, setChats, setActiveChatId, setCurrentConversationId]);

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
      setChats([{ id: 1, title: "Chat 1", messages: [] }]);
      setActiveChatId(1);
    };
    window.addEventListener(LOGOUT_EVENT, handleLogout);
    return () => window.removeEventListener(LOGOUT_EVENT, handleLogout);
  }, [setUploaded, setDatasetName, setRows, setColumns, setTableData, setTableDataBefore, setHeaders, setHeadersBefore, setUploadError, setCurrentConversationId, setChats, setActiveChatId]);

  // ─── Auto scroll on new messages ──────────────────────────────────────────────
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

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
      setOriginalExtension(".csv");
      setPendingFeedback(false);
      await loadChatMessages(chatId);
    },
    [loadChatMessages, setActiveChatId, setCurrentConversationId, setUploaded, setTableData, setTableDataBefore, setHeaders, setHeadersBefore, setDatasetName, setRows, setColumns, setOriginalExtension]
  );

  // ─── New chat ─────────────────────────────────────────────────────────────────
  const handleNewChat = async () => {
    const tempId = Date.now();
    let messages = [];
    let backendId = null;

    if (getAuthToken()) {
      try {
        const conv = await createConversation();
        backendId = conv.id;
        messages = conv.messages.map((msg) => ({
          sender: msg.sender === "user" ? "user" : "bot",
          text: msg.content,
          time: new Date(msg.created_at).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        }));
      } catch {
        // fall back to empty chat
      }
    }

    const newChat = {
      id: backendId || tempId,
      title: "New Chat",
      messages,
      ...(backendId ? { backendId } : {}),
    };
    setChats((prev) => [...prev, newChat]);
    setActiveChatId(backendId || tempId);
    setCurrentConversationId(backendId);
    setUploaded(false);
    setTableData([]);
    setTableDataBefore([]);
    setHeaders([]);
    setHeadersBefore([]);
    setDatasetName("");
    setRows(0);
    setColumns(0);
    setOriginalExtension(".csv");
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

  // ─── File upload ──────────────────────────────────────────────────────────────
  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    if (!getAuthToken()) {
      setUploadError("You must be logged in to upload files. Please log in first.");
      event.target.value = "";
      return;
    }

    const fileName = selectedFile.name.toLowerCase();
    const isCSV = fileName.endsWith(".csv");
    const isXLSX = fileName.endsWith(".xlsx") || fileName.endsWith(".xls");
    if (!isCSV && !isXLSX) {
      setUploadError(`"${selectedFile.name}" is not supported. Please upload a CSV, or Excel file.`);
      event.target.value = "";
      return;
    }
    if (selectedFile.type && !ALLOWED_MIME_TYPES.includes(selectedFile.type)) {
      setUploadError(`"${selectedFile.name}" appears to be an image or unsupported file. Please upload a CSV, or Excel file.`);
      event.target.value = "";
      return;
    }

    setOriginalExtension(isXLSX ? (fileName.endsWith(".xls") ? ".xls" : ".xlsx") : ".csv");

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

  // ─── Download current table as CSV ────────────────────────────────────────────
  const hasDownloadUrl = activeChat?.messages?.some((m) => m.downloadUrl) ?? false;

  const handleDownload = () => {
    const lastWithDownload = activeChat.messages
      .slice()
      .reverse()
      .find((m) => m.downloadUrl);

    if (!lastWithDownload) return;

    window.open(lastWithDownload.downloadUrl, "_blank");
  };

  // ─── Reconstruct dataset file preserving original format ──────────────────────
  const reconstructDatasetFile = (hdrs, data, ext) => {
    const csvContent = [
      hdrs.map(escapeCSV).join(","),
      ...data.map((row) => hdrs.map((h) => escapeCSV(row[h])).join(",")),
    ].join("\n");

    const isExcel = ext === ".xlsx" || ext === ".xls";
    const baseName = datasetName ? datasetName.replace(/\.\w+$/, "") : "data";
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

    return new File([csvContent], fileName, { type: "text/csv" });
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
      const file = reconstructDatasetFile(headers, tableData, originalExtension);

      const response = await sendChatMessage({
        message: "🤖 Auto-clean my dataset",
        mode: "full_auto",
        selectedIntents: [],
        conversationId: thisConvId,
        dataset: file,
      });

      setPendingFeedback(false); // auto-clean is always fully done, no feedback needed

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
        datasetFile = reconstructDatasetFile(headers, sourceData, originalExtension);
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
      setPendingFeedback(response.finished === false);
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

      setPendingFeedback(response.finished === false);
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
        hasDownloadUrl={hasDownloadUrl}
        handleShowHistory={handleShowHistory}
        handleReset={handleResetWithBackend}
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
          actions={ACTIONS}
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