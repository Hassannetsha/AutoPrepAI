import { useState } from "react";
import { getConversation } from "../../../api/chat";
import { formatTime } from "../utils/helpers";

function extractHistoryLogs(messages = []) {
  const seen = new Set();
  const formattedLogs = [];

  for (const msg of messages) {
    if (msg.role !== "assistant") {
      continue;
    }
    const logs = msg.payload?.logs;
    if (!logs) {
      continue;
    }
    const time = formatTime(msg.created_at);
    for (const log of logs) {
      // Remove duplicates and unnecessary logs
      if (seen.has(log) || log.startsWith("Skipping")) {
        continue;
      }
      seen.add(log);
      formattedLogs.push({
        message: log,
        time,
      });
    }
  }
  return formattedLogs;
}
/**
 * Manages fetching and displaying the processing history log
 * for the current conversation.
 *
 * @param {object} params
 * @param {string|null} params.currentConversationId # The ID of the current conversation to fetch history for.
 * @param {Function} params.setChatError # Function to set an error message in the chat context.
 */
export function useHistory({ currentConversationId, setChatError }) {
  const [showHistory, setShowHistory] = useState(false);
  const [historyLogs, setHistoryLogs] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const handleShowHistory = async () => {
    setShowHistory(true);
    setChatError("");

    if (!currentConversationId) {
      setHistoryLogs([]);
      return;
    }
    setLoadingHistory(true);
    try {
      const data = await getConversation(currentConversationId);
      const logs = extractHistoryLogs(data?.messages ?? []);
      setHistoryLogs(logs);
    } catch (err) {
      console.error(err);
      setChatError("Could not load history");
      setHistoryLogs([]);
    } finally {
      setLoadingHistory(false);
    }
  };

  return {
    showHistory,
    setShowHistory,
    historyLogs,
    loadingHistory,
    handleShowHistory,
  };
}
