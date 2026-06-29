export default function ChatInput({ inputValue, setInputValue, handleSend, canSend, isLoading }) {
  return (
    <div className="inputArea">
      <input
        placeholder="Ask me about your data..."
        value={inputValue}
        onChange={(e) => { if (!isLoading) setInputValue(e.target.value); }}
        onKeyDown={(e) => { if (e.key === "Enter" && !isLoading) handleSend(); }}
      />
      <button onClick={handleSend} disabled={isLoading || !canSend}>
        {isLoading ? "Processing" : "Send"}
      </button>
    </div>
  );
}
