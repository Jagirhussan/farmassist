import React, { useState } from "react";
import Sidebar from "./Sidebar";
import Header from "./Header";
import ChatArea from "./ChatArea";
import ChatInput from "./ChatInput";
import "../App.css";

function App() {
  const [messages, setMessages] = useState([]);

  const handleSend = async (userMessage) => {
    const newMessages = [...messages, { sender: "user", text: userMessage }];
    setMessages(newMessages);

    try {
      const response = await fetch("http://172.23.104.234:5050/ask_llm", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: userMessage }),
      });
      const data = await response.json();
      setMessages([...newMessages, { sender: "bot", text: data.output }]);
    } catch (error) {
      setMessages([
        ...newMessages,
        { sender: "bot", text: `Error: ${error.message}` },
      ]);
    }
  };

  return (
    <div className="app">
      <Sidebar />
      <div className="main">
        <Header />
        <ChatArea messages={messages} />
        <ChatInput onSend={handleSend} />
      </div>
    </div>
  );
}

export default App;