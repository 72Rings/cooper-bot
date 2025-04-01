import React, { useState, useRef } from "react";
import axios from "axios";

const API_URL = "https://confident-flexibility-production.up.railway.app/chat";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const sessionId = useRef(Date.now().toString()); // ← ✅ Now it's inside the component

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "You", text: input };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await axios.post(API_URL, {
        question: input,
        session_id: sessionId.current,
      });

      if (response.status === 200 && response.data.response) {
        const botMessage = { sender: "Bot", text: response.data.response };
        setMessages((prev) => [...prev, userMessage, botMessage]);
      } else {
        setMessages((prev) => [
          ...prev,
          userMessage,
          { sender: "Bot", text: "No response from server." },
        ]);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        userMessage,
        { sender: "Bot", text: "Error fetching response." },
      ]);
    }

    setInput("");
  };

  return (
    <div style={{ width: "400px", margin: "auto", textAlign: "center" }}>
      <h2>Cooper Bot</h2>
      <div
        style={{
          height: "300px",
          overflowY: "scroll",
          border: "1px solid #ccc",
          padding: "10px",
        }}
      >
        {messages.map((msg, index) => (
          <div
            key={index}
            style={{ textAlign: msg.sender === "You" ? "right" : "left" }}
          >
            <strong>{msg.sender}:</strong>{" "}
            {typeof msg.text === "object"
              ? JSON.stringify(msg.text)
              : msg.text}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        style={{ width: "80%", padding: "10px", marginTop: "10px" }}
      />
      <button
        onClick={sendMessage}
        style={{ padding: "10px", marginLeft: "5px" }}
      >
        Send
      </button>
    </div>
  );
};

export default Chatbot;
