import React, { useState } from "react";
import "./ChatInput.css";

function ChatInput({ onSend }) {
  const [input, setInput] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim()) {
      onSend(input);
      setInput("");
    }
  };

  return (
    <form className="chat-input-container" onSubmit={handleSubmit}>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask anything about your farm ..."
      />
      <button type="submit">
        <img src="/images/send_icon.png" alt="Send" />
      </button>
    </form>
  );
}

export default ChatInput;