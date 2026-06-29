import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import FeedbackCard from "./FeedbackCard";
import AgentMessage from "./AgentMessage";

// Heuristic: does this bot message look like a structured pipeline log
// (the ✅ / • / 📊 / [System] format), or is it a normal conversational reply?
const AGENT_LOG_PATTERN = /(^•|\n•|\[System\]|Executing agent|Skipping agent)/;

export default function ChatWindow({
  activeChat,
  chatEndRef,
  pendingFeedback,
  onAcceptFeedback,
  onRejectFeedback,
  feedbackDisabled,
  stepTitle,
}) {
  return (
    <div className="chat">
      {activeChat?.messages.map((msg, index) => {
        const isAgentLog = msg.sender === "bot" && AGENT_LOG_PATTERN.test(msg.text);

        return (
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
                isAgentLog ? (
                  <AgentMessage text={msg.text} />
                ) : (
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => (
                        <p style={{ margin: "4px 0", whiteSpace: "pre-wrap" }}>{children}</p>
                      ),
                      ul: ({ children }) => (
                        <ul style={{ paddingLeft: "16px", margin: "4px 0" }}>{children}</ul>
                      ),
                      li: ({ children }) => <li style={{ margin: "2px 0" }}>{children}</li>,
                    }}
                  >
                    {msg.text}
                  </ReactMarkdown>
                )
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
        );
      })}

      {pendingFeedback && (
        <div className="message-row row-bot">
          <div className="avatar avatar-bot">
            <Bot size={16} />
          </div>
          <div className="message message-bot">
            <FeedbackCard
              onAccept={onAcceptFeedback}
              onReject={onRejectFeedback}
              disabled={feedbackDisabled}
              title={stepTitle}
            />
          </div>
        </div>
      )}

      <div ref={chatEndRef} />
    </div>
  );
}