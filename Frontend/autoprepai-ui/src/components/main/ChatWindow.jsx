import { Bot, User } from "lucide-react";

export default function ChatWindow({ activeChat, chatEndRef }) {
  return (
    <div className="chat">
      {activeChat?.messages.map((msg, index) => (
        <div
          key={index}
          className={`message-row ${msg.sender === "user" ? "row-user" : "row-bot"}`}
        >
          {msg.sender === "bot" && (
            <div className="avatar avatar-bot">
              <Bot size={16} />
            </div>
          )}

          <div
            className={`message ${msg.sender === "user" ? "message-user" : "message-bot"}`}
          >
            <p>{msg.text}</p>
            {msg.list && (
              <ul>
                {msg.list.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            )}
            <span className="msg-time">{msg.time}</span>
          </div>

          {msg.sender === "user" && (
            <div className="avatar avatar-user">
              <User size={16} />
            </div>
          )}
        </div>
      ))}
      <div ref={chatEndRef} />
    </div>
  );
}
