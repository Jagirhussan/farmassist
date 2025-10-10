// src/components/ChatArea.jsx
import React, { useEffect, useRef } from "react";
import "./ChatArea.css";

function ChatArea({ messages, loading }) {
  const containerRef = useRef(null);

  // scroll to bottom when messages change
  useEffect(() => {
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  return (
    <div className="chat-area">
      <div className="chat-messages-box" ref={containerRef}>
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
    </div>
  );
}

export default ChatArea;
