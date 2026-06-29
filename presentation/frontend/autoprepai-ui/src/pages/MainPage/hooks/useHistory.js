import { useState } from "react";
import { getConversation } from "../../../api/chat";
import { formatTime } from "../utils/helpers";

/**
 * Manages fetching and displaying the processing history log
 * for the current conversation.
 *
 * @param {object} params
 * @param {string|null} params.currentConversationId
 * @param {Function} params.setChatError
 */
export function useHistory({ currentConversationId, setChatError }) {
  const [showHistory, setShowHistory] = useState(false);
  const [historyLogs, setHistoryLogs] = useState([]);

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
              const time = formatTime(msg.created_at);
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

  return { showHistory, setShowHistory, historyLogs, handleShowHistory };
}
