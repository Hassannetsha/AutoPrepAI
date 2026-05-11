import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";

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
            {msg.sender === "bot" ? (
              <ReactMarkdown
                components={{
                  p: ({ children }) => (
                    <p style={{ margin: "4px 0", whiteSpace: "pre-wrap" }}>{children}</p>
                  ),
                  ul: ({ children }) => (
                    <ul style={{ paddingLeft: "16px", margin: "4px 0" }}>{children}</ul>
                  ),
                  li: ({ children }) => (
                    <li style={{ margin: "2px 0" }}>{children}</li>
                  ),
                }}
              >
                {msg.text}
              </ReactMarkdown>
            ) : (
              <p style={{ whiteSpace: "pre-wrap" }}>{msg.text}</p>
            )}

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
