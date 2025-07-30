import React, { useState } from "react";
import "./Sidebar.css";

function Sidebar({ isOpen, toggleSidebar }) {
  const [showInfo, setShowInfo] = useState(false); // State to control the pop-up

  const handleNewChat = () => {
    window.location.reload(); // Refresh the page
  };

  const handleInfoClick = () => {
    setShowInfo(true); // Show the pop-up
  };

  const closeInfo = () => {
    setShowInfo(false); // Hide the pop-up
  };

  return (
    <>
      <div className={`sidebar ${isOpen ? "open" : "closed"}`}>
        <button className="toggle-sidebar" onClick={toggleSidebar}>
          <img
            src={isOpen ? "/images/sidebar_close.png" : "/images/sidebar_open.png"}
            alt="Toggle Sidebar"
          />
        </button>
        <div className="nav">
          <div className="nav-item" onClick={handleNewChat}>
            <img src="/images/newchaticon.png" alt="New Chat" />
            {isOpen && <span>New Chat</span>}
          </div>
          <div className="nav-item" onClick={handleInfoClick}>
            <img src="/images/info_icon.png" alt="Info" />
            {isOpen && <span>Info</span>}
          </div>
        </div>
        <div className="settings">
          <img src="/images/settings_icon.png" alt="Settings" />
          {isOpen && <span>Settings</span>}
        </div>
      </div>

      {/* Pop-Up Modal */}
      {showInfo && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>Disclaimer</h2>
            <p>
              The information provided by this application is for general informational purposes only. 
              We make no guarantees about the accuracy or reliability of the information.
            </p>
            <button className="close-button" onClick={closeInfo}>
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default Sidebar;