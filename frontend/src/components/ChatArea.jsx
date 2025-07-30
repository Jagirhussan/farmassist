import React from "react";
import "./ChatArea.css";

function ChatArea({ messages, loading }) {
  return (
    <div className="chat-area">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`chat-message ${
            message.sender === "user" ? "user" : "bot"
          }`}
        >
          {message.text}
        </div>
      ))}
      {loading && (
        <div className="typing-indicator">
          <span className="bubble"></span>
          <span className="bubble"></span>
          <span className="bubble"></span>
        </div>
      )}
    </div>
  );
}

export default ChatArea;