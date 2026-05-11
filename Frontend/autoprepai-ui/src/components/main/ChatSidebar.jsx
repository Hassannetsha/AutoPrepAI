import { Check, ChevronLeft, ChevronRight, MessageSquare, Pencil, Trash2, X } from "lucide-react";
import { useState } from "react";

export default function ChatSidebar({
  sidebarCollapsed,
  setSidebarCollapsed,
  handleNewChat,
  chats,
  activeChatId,
  setActiveChatId,
  handleRenameChat,
  handleDeleteChat,
}) {
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingTitle, setEditingTitle] = useState("");

  const startRename = (event, chat) => {
    event.stopPropagation();
    setEditingChatId(chat.id);
    setEditingTitle(chat.title);
  };

  const cancelRename = (event) => {
    event.stopPropagation();
    setEditingChatId(null);
    setEditingTitle("");
  };

  const saveRename = (event) => {
    event.stopPropagation();
    handleRenameChat(editingChatId, editingTitle);
    setEditingChatId(null);
    setEditingTitle("");
  };

  const handleRenameKeyDown = (event) => {
    event.stopPropagation();
    if (event.key === "Enter") {
      saveRename(event);
    }
    if (event.key === "Escape") {
      cancelRename(event);
    }
  };

  return (
    <div className={`chat-sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
      >
        {sidebarCollapsed ? (
          <ChevronRight size={16} />
        ) : (
          <ChevronLeft size={16} />
        )}
      </button>

      {!sidebarCollapsed && (
        <>
          <button className="new-chat-btn" onClick={handleNewChat}>
            <MessageSquare size={16} />
            New Chat
          </button>

          <p className="chats-label">CHATS</p>

          <div className="chat-list-container">
            {chats.map((chat) => {
              const isEditing = editingChatId === chat.id;

              return (
                <div
                  key={chat.id}
                  onClick={() => setActiveChatId(chat.id)}
                  className={`chat-item ${chat.id === activeChatId ? "chat-item-active" : ""} ${isEditing ? "chat-item-editing" : ""}`}
                >
                  <MessageSquare size={14} />

                  {isEditing ? (
                    <>
                      <input
                        className="chat-title-input"
                        value={editingTitle}
                        onChange={(event) => setEditingTitle(event.target.value)}
                        onClick={(event) => event.stopPropagation()}
                        onKeyDown={handleRenameKeyDown}
                        autoFocus
                      />
                      <button
                        className="chat-title-action"
                        onClick={saveRename}
                        aria-label="Save chat name"
                      >
                        <Check size={14} />
                      </button>
                      <button
                        className="chat-title-action"
                        onClick={cancelRename}
                        aria-label="Cancel rename"
                      >
                        <X size={14} />
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="chat-title">{chat.title}</span>
                      <button
                        className="chat-rename-btn"
                        onClick={(event) => startRename(event, chat)}
                        aria-label="Rename chat"
                      >
                        <Pencil size={13} />
                      </button>
                      {handleDeleteChat && (
                        <button
                          className="chat-delete-btn"
                          onClick={(event) => {
                            event.stopPropagation();
                            if (window.confirm(`Delete "${chat.title}"?`)) {
                              handleDeleteChat(chat.id);
                            }
                          }}
                          aria-label="Delete chat"
                        >
                          <Trash2 size={13} />
                        </button>
                      )}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
