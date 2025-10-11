import os
import re
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from framer3 import framer
import asyncio

# ---------------------
# CONFIGURATION
# ---------------------
DATA_STORAGE_IP = "172.23.117.196"  # Hardcoded Alex Jetson IP
DATA_STORAGE_PORT = 5051
UPLOAD_FOLDER = "/videos"  # Hardcoded absolute path

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"[Debug] Upload folder path: {UPLOAD_FOLDER}", flush=True)

# ---------------------
# FASTAPI SETUP
# ---------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------
# UTILITY FUNCTIONS
# ---------------------
def secure_filename(filename: str) -> str:
    """Replace unsafe characters in filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename)


def send_to_storage(data: dict):
    """Send processed data to storage server."""
    try:
        response = requests.post(
            f"http://{DATA_STORAGE_IP}:{DATA_STORAGE_PORT}/store_data", json=data
        )
        if response.status_code == 200:
            print("[Debug] Data sent successfully!", flush=True)
        else:
            print(
                f"[Debug] Failed to send data: {response.status_code} - {response.text}",
                flush=True,
            )
    except Exception as e:
        print(f"[Debug] Error sending data: {e}", flush=True)


def process_video(video_path: str):
    """Process video and send results to storage."""
    print(f"[Amy] Processing video: {video_path}", flush=True)
    framer(video_path)

    data = {
        "video_path": video_path,
        "captions": ["Caption 1", "Caption 2"],  # Replace with real captions
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],  # Replace with real embeddings
    }

    send_to_storage(data)


# ---------------------
# FASTAPI ENDPOINT
# ---------------------
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    print(f"[Debug] Absolute save path: {save_path}", flush=True)
    print(f"[Debug] Incoming filename: {file.filename}", flush=True)

    # Save file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print(f"[Amy] Received new video: {filename}", flush=True)

    # Process in background
    asyncio.create_task(async_process_video(save_path))

    return {"message": "Upload successful", "filename": filename}


async def async_process_video(video_path: str):
    """Async wrapper for process_video."""
    await asyncio.to_thread(process_video, video_path)


# ---------------------
# OPTIONAL: PERIODIC LOOP
# ---------------------
async def periodic_processing_loop():
    while True:
        for filename in os.listdir(UPLOAD_FOLDER):
            full_path = os.path.join(UPLOAD_FOLDER, filename)
            process_video(full_path)
        await asyncio.sleep(60)


# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    uvicorn.run(
        "data_processing_server:app",
        host="0.0.0.0",
        port=5050,
        reload=True,  # Automatically reloads and prints are visible
        log_level="info",
    )
