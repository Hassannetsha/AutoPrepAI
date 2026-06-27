import { useState, useCallback } from "react";
import { INITIAL_BOT_MESSAGE } from "../constants";

export function useChatManager() {
  const [chats, setChats] = useState([
    { id: 1, title: "Chat 1", messages: [INITIAL_BOT_MESSAGE] },
  ]);
  const [activeChatId, setActiveChatId] = useState(1);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const activeChat = chats.find((c) => c.id === activeChatId);

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

  return {
    chats, setChats, activeChat, activeChatId, setActiveChatId,
    currentConversationId, setCurrentConversationId,
    sidebarCollapsed, setSidebarCollapsed,
    syncConversationId,
  };
}
