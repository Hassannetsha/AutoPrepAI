import { userMsg, botMsg, formatTime } from "./helpers";

export const updateChatById = (setChats, chatId, updater) => {
  setChats((prev) =>
    prev.map((chat) => (chat.id === chatId ? updater(chat) : chat)),
  );
};

export const appendMessage = (setChats, chatId, message) => {
  updateChatById(setChats, chatId, (chat) => ({
    ...chat,
    messages: [...chat.messages, message],
  }));
};

const appendMessagesToMatchingChats = (setChats, matchFn, newMessages) => {
  setChats((prev) =>
    prev.map((chat) =>
      matchFn(chat)
        ? { ...chat, messages: [...chat.messages, ...newMessages] }
        : chat,
    ),
  );
};

export const appendUploadMessages = ({
  setChats,
  activeChatId,
  realConvId,
  uploadMessage,
  assistantMessage,
  downloadUrl,
}) => {
  const time = formatTime();
  const matchFn = (chat) => chat.id === activeChatId || chat.id === realConvId;

  appendMessagesToMatchingChats(setChats, matchFn, [
    userMsg(uploadMessage, time),
    botMsg(assistantMessage, time, downloadUrl),
  ]);
};

export const appendMessageToChats = (setChats, chatIds, message) => {
  const matchFn = (chat) => chatIds.includes(chat.id);
  appendMessagesToMatchingChats(setChats, matchFn, [message]);
};
