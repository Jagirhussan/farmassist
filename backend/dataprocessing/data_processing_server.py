import os
import re
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from framer3 import framer
from dotenv import load_dotenv
import asyncio

# --- Load environment variables from .env ---
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# Get IP and port for Alex Jetson storage server from .env
DATA_STORAGE_IP = os.getenv("REACT_APP_ALEX_IP", "127.0.0.1")
DATA_STORAGE_PORT = int(os.getenv("REACT_APP_ALEX_PORT", 5051))

# Relative path to videos folder (relative to this script)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "videos")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"[Debug] Upload folder path: {UPLOAD_FOLDER}", flush=True)
print(
    f"[Debug] Storage server IP: {DATA_STORAGE_IP}, Port: {DATA_STORAGE_PORT}",
    flush=True,
)
print(f"[Debug] Current working directory: {os.getcwd()}", flush=True)

# --- FastAPI app ---
app = FastAPI()

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# --- Helper functions ---
def secure_filename(filename):
    """Sanitize filename to remove unsafe characters."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename)


def send_to_storage(data):
    """Send processed data to the storage server on Alex Jetson."""
    try:
        url = f"http://{DATA_STORAGE_IP}:{DATA_STORAGE_PORT}/store_data"
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"[Debug] Data sent successfully to {url}", flush=True)
        else:
            print(
                f"[Debug] Failed to send data: {response.status_code} - {response.text}",
                flush=True,
            )
    except Exception as e:
        print(f"[Debug] Error sending data to storage server: {e}", flush=True)


def process_video(video_path):
    """Process a video and send results to storage."""
    abs_path = os.path.abspath(video_path)
    print(f"[Amy] Processing video: {abs_path}", flush=True)
    processed_data = framer(abs_path)

    data = {
        "id": [processed_data[i]["id"] for i in range(len(processed_data))],
        "captions": [processed_data[i]["caption"] for i in range(len(processed_data))],
        "embeddings": [processed_data[i]["embedding"] for i in range(len(processed_data))]
    }
    send_to_storage(data)


async def async_process_video(video_path):
    """Run process_video asynchronously."""
    await asyncio.to_thread(process_video, video_path)


# --- FastAPI endpoint ---
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    save_path = os.path.abspath(save_path)

    print(f"[Debug] Absolute save path: {save_path}", flush=True)
    print(f"[Debug] Incoming filename: {file.filename}", flush=True)

    # Save uploaded file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print(f"[Amy] Received new video: {file.filename}", flush=True)

    # Process in background
    asyncio.create_task(async_process_video(save_path))

    return {"message": "Upload successful", "filename": file.filename}


# Optional: periodic processing of existing videos
async def periodic_processing_loop():
    while True:
        for filename in os.listdir(UPLOAD_FOLDER):
            full_path = os.path.join(UPLOAD_FOLDER, filename)
            process_video(full_path)
        await asyncio.sleep(60)


# --- Run server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")
