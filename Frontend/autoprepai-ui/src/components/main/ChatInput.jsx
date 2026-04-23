export default function ChatInput({ inputValue, setInputValue, handleSend }) {
  return (
    <div className="inputArea">
      <input
        placeholder="Ask me about your data..."
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
      />
      <button onClick={handleSend} disabled={!inputValue.trim()}>
        Send
      </button>
    </div>
  );
}
