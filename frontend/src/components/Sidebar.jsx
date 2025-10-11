import React, { useState } from "react";
import "./Sidebar.css";

function Sidebar({ isOpen, toggleSidebar }) {
  const [showInfo, setShowInfo] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  const AMY_JETSON_URL = process.env.REACT_APP_AMY_IP;

  const handleNewChat = () => window.location.reload();
  const handleInfoClick = () => setShowInfo(true);
  const handleUploadClick = () => setShowUpload(true);
  const closeInfo = () => setShowInfo(false);
  const closeUpload = () => {
    setShowUpload(false);
    setSelectedFile(null);
    setUploadMessage("");
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) {
      alert("Please select a file first");
      return;
    }

    setIsUploading(true);
    setUploadMessage("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${AMY_JETSON_URL}/upload_video`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      setUploadMessage(`Upload complete: ${data.filename}`);
    } catch (err) {
      console.error(err);
      setUploadMessage("Error uploading video");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <>
      <div className={`sidebar ${isOpen ? "open" : "closed"}`}>
        <button className="toggle-sidebar" onClick={toggleSidebar}>
          <img
            src={
              isOpen ? "/images/sidebar_close.png" : "/images/sidebar_open.png"
            }
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

          <div className="nav-item" onClick={handleUploadClick}>
            <img src="/images/upload.png" alt="Upload" />
            {isOpen && <span>Upload Video</span>}
          </div>
        </div>

        <div className="settings">
          <img src="/images/settings_icon.png" alt="Settings" />
          {isOpen && <span>Settings</span>}
        </div>
      </div>
      {/* Info Modal */}
      {showInfo && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>Disclaimer</h2>
            <p>
              The information provided by this application is for general
              informational purposes only. We make no guarantees about the
              accuracy or reliability of the information.
            </p>
            <button className="close-button" onClick={closeInfo}>
              Close
            </button>
          </div>
        </div>
      )}
      {/* Upload Modal */}
      {showUpload && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>Upload Video</h2>

            {!selectedFile && (
              <input
                type="file"
                accept="video/*"
                style={{ marginBottom: "15px" }}
                onChange={(e) => setSelectedFile(e.target.files[0])}
              />
            )}

            {selectedFile && (
              <>
                <div className="upload-status">
                  {isUploading && <div className="spinner"></div>}
                  <span>
                    {isUploading
                      ? "Uploading ..."
                      : `Ready to upload: ${selectedFile.name}`}
                  </span>
                </div>

                {uploadMessage && (
                  <div className="upload-message">
                    {uploadMessage}
                    {uploadMessage.startsWith("Upload complete") && (
                      <div className="upload-note">
                        Please wait a few more minutes for the video to be fully
                        processed.
                      </div>
                    )}
                  </div>
                )}

                <div className="upload-buttons">
                  <button
                    className="upload-button"
                    onClick={handleUploadSubmit}
                    disabled={isUploading}
                  >
                    Upload
                  </button>
                  <button
                    className="close-button"
                    onClick={closeUpload}
                    disabled={isUploading}
                  >
                    Close
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default Sidebar;
