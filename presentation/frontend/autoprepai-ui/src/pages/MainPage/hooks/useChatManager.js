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
import { cleanError, formatTime, updateDatasetFromResponse, userMsg, botMsg } from "../utils/helpers";
import { reconstructDatasetFile } from "../utils/datasetFile";
import { ACTION_TO_INTENT } from "../constants";
import { LOGOUT_EVENT } from "../../../components/main/AppHeader";

export function useChatManager({
  setUploaded, setTableData, setTableDataBefore,
  setHeaders, setHeadersBefore, setDatasetName,
  setRows, setColumns, setOriginalExtension, setUploadError,
  restoreDatasetFromPayload,
  setSelectedActions,
} = {}) {
  const [chats, setChats] = useState([
    { id: 1, title: "Chat 1", messages: [] },
  ]);
  const [activeChatId, setActiveChatId] = useState(1);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isLoadingChat, setIsLoadingChat] = useState(false);
  const [chatError, setChatError] = useState("");
  const [pendingFeedback, setPendingFeedback] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [stepTitle, setStepTitle] = useState("");

  const chatEndRef = useRef(null);
  const navigate = useNavigate();

  const activeChat = chats.find((c) => c.id === activeChatId);

  const safe = (fn) => (typeof fn === "function" ? fn : () => {});

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

  const loadChatMessages = useCallback(
    async (conversationId) => {
      try {
        const data = await getConversation(conversationId);
        if (!data?.messages) return;

        const mapped = data.messages.map((msg) => ({
          role: msg.role,
          text: msg.content,
          time: formatTime(msg.created_at),
          downloadUrl: msg.payload?.download_url ?? null,
        }));

        setChats((prev) =>
          prev.map((chat) =>
            chat.id === conversationId ? { ...chat, messages: mapped } : chat
          )
        );

        const uploadMessage = [...data.messages].reverse().find((m) =>
          m.role === "assistant" && m.payload?.dataset_name
        );

        if (uploadMessage?.payload?.dataset_name) {
          safe(setDatasetName)(uploadMessage.payload.dataset_name);
          const ext = uploadMessage.payload.dataset_name.match(/\.\w+$/)?.[0]?.toLowerCase() || ".csv";
          safe(setOriginalExtension)(ext);
        }

        const lastAssistant = [...data.messages]
          .reverse()
          .find((m) => m.role === "assistant" && m.payload);

        if (lastAssistant?.payload) {
          safe(restoreDatasetFromPayload)(lastAssistant.payload);
          setPendingFeedback(lastAssistant.payload?.finished === false);
          setStepTitle(lastAssistant.payload?.step_title ?? "");
        }
      } catch (error) {
        console.error("Failed to load messages:", error);
        const friendly = cleanError(error.message);
        if (friendly.includes("log out")) {
          setChatError(friendly);
        }
        if (error.message === "SESSION_EXPIRED") {
          removeAuthToken();
          window.dispatchEvent(new Event(LOGOUT_EVENT));
        }
      }
    },
    [setChats, setDatasetName, setOriginalExtension, restoreDatasetFromPayload, setChatError]
  );

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
      if (error.message === "SESSION_EXPIRED") {
        removeAuthToken();
        window.dispatchEvent(new Event(LOGOUT_EVENT));
      }
    }
  }, [loadChatMessages, setChats, setActiveChatId, setCurrentConversationId, setChatError]);

  useEffect(() => {
    if (!getAuthToken()) return;
    loadConversations();
  }, [navigate, loadConversations]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

  useEffect(() => {
    const handleLogout = () => {
      safe(setUploaded)(false);
      safe(setDatasetName)("");
      safe(setRows)(0);
      safe(setColumns)(0);
      safe(setTableData)([]);
      safe(setTableDataBefore)([]);
      safe(setHeaders)([]);
      safe(setHeadersBefore)([]);
      safe(setUploadError)("");
      setCurrentConversationId(null);
      setChats([{ id: 1, title: "Chat 1", messages: [] }]);
      setActiveChatId(1);
      setPendingFeedback(false);
      setStepTitle("");
    };
    window.addEventListener(LOGOUT_EVENT, handleLogout);
    return () => window.removeEventListener(LOGOUT_EVENT, handleLogout);
  }, [setUploaded, setDatasetName, setRows, setColumns, setTableData, setTableDataBefore, setHeaders, setHeadersBefore, setUploadError, setCurrentConversationId, setChats, setActiveChatId]);

  const handleSwitchChat = useCallback(
    async (chatId) => {
      setActiveChatId(chatId);
      setCurrentConversationId(chatId);
      safe(setUploaded)(false);
      safe(setTableData)([]);
      safe(setTableDataBefore)([]);
      safe(setHeaders)([]);
      safe(setHeadersBefore)([]);
      safe(setDatasetName)("");
      safe(setRows)(0);
      safe(setColumns)(0);
      safe(setOriginalExtension)(".csv");
      setPendingFeedback(false);
      await loadChatMessages(chatId);
    },
    [loadChatMessages, setActiveChatId, setCurrentConversationId, setUploaded, setTableData, setTableDataBefore, setHeaders, setHeadersBefore, setDatasetName, setRows, setColumns, setOriginalExtension]
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
    safe(setUploaded)(false);
    safe(setTableData)([]);
    safe(setTableDataBefore)([]);
    safe(setHeaders)([]);
    safe(setHeadersBefore)([]);
    safe(setDatasetName)("");
    safe(setRows)(0);
    safe(setColumns)(0);
    safe(setOriginalExtension)(".csv");
    safe(setChatError)("");
    setPendingFeedback(false);
    safe(setSelectedActions)([]);
  }, [setChats, setActiveChatId, setCurrentConversationId, setUploaded, setTableData, setTableDataBefore, setHeaders, setHeadersBefore, setDatasetName, setRows, setColumns, setOriginalExtension, setChatError, setSelectedActions]);

  const handleRenameChat = useCallback(
    async (chatId, title) => {
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
    },
    [setChats, setChatError, navigate]
  );

  const handleDeleteChat = useCallback(
    async (chatId) => {
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
    },
    [chats, activeChatId, setChats, handleSwitchChat, handleNewChat]
  );

  const handleFeedback = useCallback(
    async (accept) => {
      if (!currentConversationId || submittingFeedback) return;

      setSubmittingFeedback(true);
      setChatError("");

      try {
        const response = await sendFeedback({ conversationId: currentConversationId, accept });

        const botTime = formatTime();

        setChats((prev) =>
          prev.map((chat) =>
            chat.id === currentConversationId
              ? {
                  ...chat,
                  messages: [
                    ...chat.messages,
                    botMsg(response.assistant_message, botTime, response.result?.download_url),
                  ],
                }
              : chat
          )
        );

        setPendingFeedback(response.finished === false);
        setStepTitle(response.step_title || "");

        updateDatasetFromResponse(response.result, { setHeaders, setHeadersBefore, setTableData, setTableDataBefore, setRows, setColumns });
      } catch (error) {
        console.error("Feedback error:", error);
        setChatError(cleanError(error.message));
      } finally {
        setSubmittingFeedback(false);
      }
    },
    [currentConversationId, submittingFeedback, setChats, setHeaders, setHeadersBefore, setTableData, setTableDataBefore, setRows, setColumns, setChatError]
  );

  const handleSend = useCallback(
    async ({ inputValue, selectedActions, setInputValue, headers, tableData, originalExtension, datasetName }) => {
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

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId
            ? { ...chat, messages: [...chat.messages, userMsg(displayedMessage, time)] }
            : chat
        )
      );

      safe(setInputValue)("");
      setChatError("");
      setIsLoadingChat(true);

      try {
        let datasetFile = null;
        if (tableData?.length > 0 && headers?.length > 0) {
          datasetFile = reconstructDatasetFile(headers, tableData, originalExtension || ".csv", datasetName);
        }

        const backendMessage = `${messageText}\n\nPlease apply these actions: ${selectedActions.join(", ")}`;

        console.log("Sending message to backend:", backendMessage);
        console.log("mode:", inputValue.trim() === "" && selectedActions.length > 0 ? "manual" : "chat");
        const response = await sendChatMessage({
          message: backendMessage,
          mode: inputValue.trim() === "" && selectedActions.length > 0 ? "manual" : "chat",
          selectedIntents: selectedActions.map((a) => ACTION_TO_INTENT[a] || a),
          conversationId: thisConvId,
          dataset: datasetFile,
        });

        const realConvId = response.conversation_id ?? thisConvId;
        syncConversationId(response.conversation_id, thisChatId);

        const botTime = formatTime();
        setPendingFeedback(response.finished === false);
        setStepTitle(response.step_title || "");

        setChats((prev) =>
          prev.map((chat) =>
            chat.id === thisChatId || chat.id === realConvId
              ? {
                  ...chat,
                  messages: [
                    ...chat.messages,
                    botMsg(response.assistant_message || "I received your message!", botTime, response.result?.download_url ?? null),
                  ],
                }
              : chat
          )
        );

        updateDatasetFromResponse(response.result, { setHeaders, setHeadersBefore, setTableData, setTableDataBefore, setRows, setColumns });
        safe(setSelectedActions)([]);
      } catch (error) {
        console.error("Chat error:", error);
        setChatError(cleanError(error.message));
        const botTime = formatTime();
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === thisChatId
              ? {
                  ...chat,
                  messages: [
                    ...chat.messages,
                    botMsg(`Error: ${error.message}`, botTime),
                  ],
                }
              : chat
          )
        );
      } finally {
        setIsLoadingChat(false);
      }
    },
    [activeChatId, currentConversationId, syncConversationId, setChats, setChatError, setIsLoadingChat, setHeaders, setHeadersBefore, setTableData, setTableDataBefore, setRows, setColumns, setSelectedActions]
  );

  return {
    chats, setChats, activeChat, activeChatId, setActiveChatId,
    currentConversationId, setCurrentConversationId,
    sidebarCollapsed, setSidebarCollapsed,
    syncConversationId,
    isLoadingChat, setIsLoadingChat,
    chatError, setChatError,
    chatEndRef,
    pendingFeedback, setPendingFeedback,
    submittingFeedback, setSubmittingFeedback,
    stepTitle, setStepTitle,
    handleFeedback,
    loadChatMessages,
    loadConversations,
    handleSwitchChat,
    handleNewChat,
    handleRenameChat,
    handleDeleteChat,
    handleSend,
  };
}
