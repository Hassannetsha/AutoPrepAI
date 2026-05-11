import { getAuthToken } from "./auth";

const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8022";

async function chatRequest(path, options = {}) {
  const token = getAuthToken();
  const headers = {
    ...options.headers,
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      "Something went wrong. Please try again.";
    throw new Error(
      Array.isArray(message)
        ? message.map((item) => item.msg).join(" ")
        : message,
    );
  }

  return data;
}

/**
 * Send a chat message
 * @param {Object} params
 * @param {string} params.message - The message text
 * @param {string} [params.mode="chat"] - The mode (e.g., "chat", "analyze")
 * @param {Array<string>} [params.selectedIntents] - List of intents to apply
 * @param {string} [params.conversationId] - Existing conversation ID
 * @param {File} [params.dataset] - Optional dataset file
 */
export async function sendChatMessage({
  message,
  mode = "chat",
  selectedIntents = [],
  conversationId = null,
  dataset = null,
}) {
  const formData = new FormData();
  formData.append("message", message);
  formData.append("mode", mode);
  formData.append("selected_intents", JSON.stringify(selectedIntents));

  if (conversationId) {
    formData.append("conversation_id", conversationId);
  }

  if (dataset) {
    formData.append("dataset", dataset);
  }

  const token = getAuthToken();
  const headers = {};

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers,
    body: formData,
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message = data?.detail || data?.message || "Failed to send message";
    throw new Error(
      Array.isArray(message)
        ? message.map((item) => item.msg).join(" ")
        : message,
    );
  }

  return data;
}

/**
 * Get conversation history
 * @param {string} conversationId
 */
export async function getConversation(conversationId) {
  return chatRequest(`/conversations/${conversationId}`);
}

/**
 * Delete a conversation
 * @param {string} conversationId
 */
export async function deleteConversation(conversationId) {
  return chatRequest(`/conversations/${conversationId}`, {
    method: "DELETE",
  });
}

/**
 * Rename a conversation
 * @param {string} conversationId
 * @param {string} title
 */
export async function renameConversation(conversationId, title) {
  return chatRequest(`/conversations/${conversationId}/rename`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
  });
}

/**
 * List all conversations for the current user
 */
export async function listConversations() {
  return chatRequest("/conversations");
}
