import { useState, useCallback, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  sendChatMessage,
  deleteConversation,
  renameConversation,
  getConversation,
  listConversations,
  createConversation,
  sendFeedback,
} from "../../../api/chat";
import { getAuthToken, removeAuthToken } from "../../../api/auth";
import {
  formatTime,
  updateDatasetFromResponse,
  userMsg,
  botMsg,
} from "../utils/helpers";
import { cleanError, handleAuthError } from "../utils/handleErrors";
import {
  reconstructDatasetFile,
  resetDatasetState,
} from "../utils/datasetFile";
import { ACTION_TO_INTENT } from "../constants";
import { LOGOUT_EVENT } from "../../../components/main/AppHeader";
import {
  updateChatById,
  appendMessage,
  appendMessageToChats,
} from "../utils/chatState";

export function useChatManager({
  setUploaded,
  setTableData,
  setTableDataBefore,
  setHeaders,
  setHeadersBefore,
  setDatasetName,
  setRows,
  setColumns,
  setOriginalExtension,
  setUploadError,
  restoreDatasetFromPayload,
  setSelectedActions,
} = {}) {
  const [chats, setChats] = useState([
    { id: 1, title: "Chat 1", messages: [] },
  ]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isLoadingChat, setIsLoadingChat] = useState(false);
  const [chatError, setChatError] = useState("");
  const [pendingFeedback, setPendingFeedback] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [stepTitle, setStepTitle] = useState("");
  const [isDeletingChat, setIsDeletingChat] = useState(false);

  const chatEndRef = useRef(null);
  const navigate = useNavigate();

  const activeChat = chats.find((c) => c.id === activeChatId);

  const safe = (fn) => (typeof fn === "function" ? fn : () => {}); // check if setter functions are provided before calling them

  const bumpChatToTop = useCallback((chatId) => {
    setChats((prev) => {
      const idx = prev.findIndex((c) => c.id === chatId);
      if (idx <= 0) return prev;
      const chat = prev[idx];
      return [chat, ...prev.slice(0, idx), ...prev.slice(idx + 1)];
    });
  }, []);

  // syncs the conversation ID from the backend response with the local chat state
  const syncConversationId = useCallback(
    (responseConvId, thisChatId) => {
      if (!currentConversationId && responseConvId) {
        setCurrentConversationId(responseConvId);
        updateChatById(setChats, thisChatId, (chat) => ({
          ...chat,
          id: responseConvId,
          backendId: responseConvId,
        }));
        setActiveChatId(responseConvId);
      }
    },
    [currentConversationId],
  );
  // loads messages for a specific conversation
  const loadChatMessages = useCallback(
    async (conversationId) => {
      setIsLoadingChat(true);
      try {
        const data = await getConversation(conversationId);
        if (!data?.messages) return;

        const mapped = data.messages.map((msg) => ({
          role: msg.role,
          text: msg.content,
          time: formatTime(msg.created_at),
          downloadUrl: msg.payload?.download_url ?? null,
        }));
        // Update the chat with the loaded messages
        updateChatById(setChats, conversationId, (chat) => ({
          ...chat,
          messages: mapped,
        }));

        // restore dataset state from the last assistant payload first
        const lastAssistant = [...data.messages]
          .reverse()
          .find((m) => m.role === "assistant" && m.payload);

        if (lastAssistant?.payload) {
          safe(restoreDatasetFromPayload)(lastAssistant.payload);
          setPendingFeedback(lastAssistant.payload?.finished === false);
          setStepTitle(lastAssistant.payload?.step_title ?? "");
        }

        // then ensure dataset name is set from the upload message if available
        const uploadMessage = [...data.messages]
          .reverse()
          .find((m) => m.role === "assistant" && m.payload?.dataset_name);

        if (uploadMessage?.payload?.dataset_name) {
          safe(setDatasetName)(uploadMessage.payload.dataset_name);
          const ext =
            uploadMessage.payload.dataset_name
              .match(/\.\w+$/)?.[0]
              ?.toLowerCase() || ".csv";
          safe(setOriginalExtension)(ext);
        }
      } catch (err) {
        console.error("Failed to load messages:", err);
        const friendly = cleanError(err.message);
        if (friendly.includes("log out")) {
          setChatError(friendly);
        }
        if (err.message === "SESSION_EXPIRED") {
          removeAuthToken();
          window.dispatchEvent(new Event(LOGOUT_EVENT));
        }
      } finally {
        setIsLoadingChat(false);
      }
    },
    [
      setChats,
      setDatasetName,
      setOriginalExtension,
      restoreDatasetFromPayload,
      setChatError,
      setIsLoadingChat,
    ],
  );

  // loads all conversations
  const loadConversations = useCallback(async () => {
    try {
      const data = await listConversations();
      // Sort conversations by updated_at, then created_at
      if (data && data.length > 0) {
        const sorted = [...data].sort((a, b) => {
          const diff = new Date(b.updated_at) - new Date(a.updated_at);
          if (diff !== 0) return diff;
          return new Date(b.created_at) - new Date(a.created_at);
        });

        const loadedChats = sorted.map((conv) => ({
          id: conv.id,
          title: conv.title || "Untitled Chat",
          messages: [],
          backendId: conv.id,
        }));
        setChats(loadedChats);

        setActiveChatId(sorted[0].id);
        setCurrentConversationId(sorted[0].id);
        await loadChatMessages(sorted[0].id);
      } else {
        setChats([]);
        setActiveChatId(null);
        setCurrentConversationId(null);
      }
    } catch (error) {
      console.error("Failed to load conversations:", error);
      const friendly = cleanError(error.message);
      if (friendly.includes("log out")) {
        setChatError(friendly);
      }
      if (error.message === "SESSION_EXPIRED") {
        removeAuthToken();
        window.dispatchEvent(new Event(LOGOUT_EVENT));
      }
    }
  }, [
    loadChatMessages,
    setChats,
    setActiveChatId,
    setCurrentConversationId,
    setChatError,
  ]);

  useEffect(() => {
    if (!getAuthToken()) return;
    loadConversations();
  }, [navigate, loadConversations]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

  useEffect(() => {
    const handleLogout = () => {
      resetDatasetState({
        setUploaded,
        setDatasetName,
        setRows,
        setColumns,
        setTableData,
        setTableDataBefore,
        setHeaders,
        setHeadersBefore,
        setUploadError,
        setOriginalExtension,
      });
      setCurrentConversationId(null);
      setChats([]);
      setActiveChatId(null);
      setPendingFeedback(false);
      setStepTitle("");
    };
    window.addEventListener(LOGOUT_EVENT, handleLogout);
    return () => window.removeEventListener(LOGOUT_EVENT, handleLogout);
  }, [
    setUploaded,
    setDatasetName,
    setRows,
    setColumns,
    setTableData,
    setTableDataBefore,
    setHeaders,
    setHeadersBefore,
    setUploadError,
    setOriginalExtension,
    setCurrentConversationId,
    setChats,
    setActiveChatId,
  ]);

  const handleSwitchChat = useCallback(
    async (chatId) => {
      setActiveChatId(chatId);
      setCurrentConversationId(chatId);
      resetDatasetState({
        setUploaded,
        setDatasetName,
        setRows,
        setColumns,
        setTableData,
        setTableDataBefore,
        setHeaders,
        setHeadersBefore,
        setUploadError,
        setOriginalExtension,
      });
      setPendingFeedback(false);
      await loadChatMessages(chatId);
    },
    [
      loadChatMessages,
      setActiveChatId,
      setCurrentConversationId,
      setUploaded,
      setDatasetName,
      setRows,
      setColumns,
      setTableData,
      setTableDataBefore,
      setHeaders,
      setHeadersBefore,
      setUploadError,
      setOriginalExtension,
    ],
  );

  const handleNewChat = useCallback(async () => {
    const tempId = Date.now();
    let messages = [];
    let backendId = null;

    if (getAuthToken()) {
      try {
        const conv = await createConversation();
        backendId = conv.id;
        messages = conv.messages.map((msg) => ({
          role: msg.role,
          text: msg.content,
          time: formatTime(msg.created_at),
        }));
      } catch (err) {
        console.error("Failed to create new conversation:", err);
        setChatError("Could not create a new chat. Please try again.");
      }
    }

    const newChat = {
      id: backendId || tempId,
      title: "New Chat",
      messages,
      ...(backendId ? { backendId } : {}),
    };
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(backendId || tempId);
    setCurrentConversationId(backendId);
    resetDatasetState({
      setUploaded,
      setDatasetName,
      setRows,
      setColumns,
      setTableData,
      setTableDataBefore,
      setHeaders,
      setHeadersBefore,
      setUploadError,
      setOriginalExtension,
    });
    safe(setChatError)("");
    setPendingFeedback(false);
    safe(setSelectedActions)([]);
  }, [
    setChats,
    setActiveChatId,
    setCurrentConversationId,
    setUploaded,
    setDatasetName,
    setRows,
    setColumns,
    setTableData,
    setTableDataBefore,
    setHeaders,
    setHeadersBefore,
    setUploadError,
    setOriginalExtension,
    setChatError,
    setSelectedActions,
  ]);

  const handleRenameChat = useCallback(
    async (chatId, title) => {
      const trimmedTitle = title.trim();
      if (!trimmedTitle) return;

      updateChatById(setChats, chatId, (chat) => ({
        ...chat,
        title: trimmedTitle,
      }));

      if (typeof chatId === "string") {
        try {
          await renameConversation(chatId, trimmedTitle);
        } catch (error) {
          console.error("Failed to rename conversation:", error);
          const message = error?.message || "Could not rename conversation";
          setChatError(message);
          handleAuthError(error, navigate);
        }
      }
    },
    [setChats, setChatError, navigate],
  );

  const handleDeleteChat = useCallback(
    async (chatId) => {
      setIsDeletingChat(true);
      try {
        await deleteConversation(chatId);

        const remaining = chats.filter((c) => c.id !== chatId);
        setChats(remaining);

        if (activeChatId === chatId) {
          if (remaining.length > 0) {
            await handleSwitchChat(remaining[0].id);
          } else {
            handleNewChat();
          }
        }
      } catch (error) {
        console.error("Failed to delete conversation:", error);
      } finally {
        setIsDeletingChat(false);
      }
    },
    [chats, activeChatId, handleSwitchChat, handleNewChat],
  );

  const handleFeedback = useCallback(
    async (accept) => {
      if (!currentConversationId || submittingFeedback) return;

      setSubmittingFeedback(true);
      setChatError("");

      try {
        const response = await sendFeedback({
          conversationId: currentConversationId,
          accept,
        });

        appendMessage(
          setChats,
          currentConversationId,
          botMsg(
            response.assistant_message,
            formatTime(),
            response.result?.download_url,
          ),
        );

        setPendingFeedback(response.finished === false);
        setStepTitle(response.step_title || "");

        updateDatasetFromResponse(response.result, {
          setHeaders,
          setHeadersBefore,
          setTableData,
          setTableDataBefore,
          setRows,
          setColumns,
        });
        bumpChatToTop(currentConversationId);
      } catch (error) {
        console.error("Feedback error:", error);
        setChatError(cleanError(error.message));
      } finally {
        setSubmittingFeedback(false);
      }
    },
    [
      currentConversationId,
      submittingFeedback,
      setChats,
      setHeaders,
      setHeadersBefore,
      setTableData,
      setTableDataBefore,
      setRows,
      setColumns,
      setChatError,
      bumpChatToTop,
    ],
  );

  const handleSend = useCallback(
    async ({
      inputValue,
      selectedActions,
      setInputValue,
      headers,
      tableData,
      originalExtension,
      datasetName,
    }) => {
      if (!inputValue.trim() && selectedActions.length === 0) return;

      const thisChatId = activeChatId;
      const thisConvId = currentConversationId;
      const time = formatTime();
      const messageText = inputValue.trim();
      let displayedMessage = messageText;

      if (selectedActions.length > 0) {
        displayedMessage +=
          (displayedMessage ? "\n\n" : "") +
          `Please apply these actions: ${selectedActions.join(", ")}`;
      }

      appendMessage(setChats, thisChatId, userMsg(displayedMessage, time));

      safe(setInputValue)("");
      setChatError("");
      setIsLoadingChat(true);

      try {
        let datasetFile = null;
        if (tableData?.length > 0 && headers?.length > 0) {
          datasetFile = reconstructDatasetFile(
            headers,
            tableData,
            originalExtension || ".csv",
            datasetName,
          );
        }

        const backendMessage = `${messageText}\n\nPlease apply these actions: ${selectedActions.join(", ")}`;
        const mode =
          inputValue.trim() === "" && selectedActions.length > 0
            ? "manual"
            : "chat";

        console.log("Sending message to backend:", backendMessage);
        console.log(mode);
        const response = await sendChatMessage({
          message: backendMessage,
          mode,
          selectedIntents: selectedActions.map((a) => ACTION_TO_INTENT[a] || a),
          conversationId: thisConvId,
          dataset: datasetFile,
        });

        const realConvId = response.conversation_id ?? thisConvId;
        syncConversationId(response.conversation_id, thisChatId);

        setPendingFeedback(response.finished === false);
        setStepTitle(response.step_title || "");

        appendMessageToChats(
          setChats,
          [thisChatId, realConvId],
          botMsg(
            response.assistant_message || "I received your message!",
            formatTime(),
            response.result?.download_url ?? null,
          ),
        );

        updateDatasetFromResponse(response.result, {
          setHeaders,
          setHeadersBefore,
          setTableData,
          setTableDataBefore,
          setRows,
          setColumns,
        });
        safe(setSelectedActions)([]);
        bumpChatToTop(thisChatId);
      } catch (error) {
        console.error("Chat error:", error);
        setChatError(cleanError(error.message));
        appendMessage(
          setChats,
          thisChatId,
          botMsg(`Error: ${error.message}`, formatTime()),
        );
        bumpChatToTop(thisChatId);
      } finally {
        setIsLoadingChat(false);
      }
    },
    [
      activeChatId,
      currentConversationId,
      syncConversationId,
      setChats,
      setChatError,
      setIsLoadingChat,
      setHeaders,
      setHeadersBefore,
      setTableData,
      setTableDataBefore,
      setRows,
      setColumns,
      setSelectedActions,
      bumpChatToTop,
    ],
  );

  return {
    chats,
    setChats,
    activeChat,
    activeChatId,
    setActiveChatId,
    currentConversationId,
    setCurrentConversationId,
    sidebarCollapsed,
    setSidebarCollapsed,
    syncConversationId,
    isLoadingChat,
    setIsLoadingChat,
    chatError,
    setChatError,
    chatEndRef,
    pendingFeedback,
    setPendingFeedback,
    submittingFeedback,
    setSubmittingFeedback,
    stepTitle,
    setStepTitle,
    isDeletingChat,
    setIsDeletingChat,
    handleFeedback,
    loadChatMessages,
    loadConversations,
    handleSwitchChat,
    handleNewChat,
    handleRenameChat,
    handleDeleteChat,
    handleSend,
    bumpChatToTop,
  };
}
