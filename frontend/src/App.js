import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import ChatArea from "./components/ChatArea";
import ChatInput from "./components/ChatInput";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(false);

  // Get alexIp from the .env file
  const alexIp = process.env.REACT_APP_ALEX_IP;

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  // Function to handle sending messages
  const handleSend = async (userMessage) => {
    if (!alexIp) {
      setMessages([
        ...messages,
        { sender: "bot", text: "Config not loaded yet" },
      ]);
      return;
    }

    const newMessages = [...messages, { sender: "user", text: userMessage }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const response = await fetch(`http://${alexIp}:5050/ask_llm`, {
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
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
      <div
        className={`main ${isSidebarOpen ? "sidebar-open" : "sidebar-closed"}`}
      >
        <Header />
        <ChatArea messages={messages} loading={loading} />
        <ChatInput onSend={handleSend} />
      </div>
    </div>
  );
}

export default App;
