import os
import time
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from framer3 import framer
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()
DATA_STORAGE_IP = os.getenv("REACT_APP_ALEX_IP")  # IP of Alex Jetson
DATA_STORAGE_PORT = 5051  # Port for the data storage server

UPLOAD_FOLDER = "backend/dataprocessing/videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def send_to_storage(data):
    """Send processed data to the storage server on Alex Jetson."""
    try:
        response = requests.post(
            f"http://{DATA_STORAGE_IP}:{DATA_STORAGE_PORT}/store_data", json=data
        )
        if response.status_code == 200:
            print("Data sent successfully!")
        else:
            print(f"Failed to send data: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending data to storage server: {e}")


def process_video(video_path):
    """Process a video and send the results to storage."""
    print(f"[Amy] Processing video: {video_path}")
    framer(video_path)  # Your existing processing

    # Example data to send to Alex Jetson
    data = {
        "video_path": video_path,
        "captions": ["Caption 1", "Caption 2"],  # Replace with actual captions
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],  # Replace with actual embeddings
    }
    send_to_storage(data)


# --- FastAPI endpoint for frontend uploads ---
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    # Save file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print(f"[Amy] Received new video: {file.filename}")

    # Process immediately in background
    asyncio.create_task(async_process_video(save_path))

    return {"message": "Upload successful", "filename": file.filename}


async def async_process_video(video_path):
    """Run process_video in an async-friendly way."""
    await asyncio.to_thread(process_video, video_path)


# --- Optional: background loop to process existing videos every minute ---
async def periodic_processing_loop():
    while True:
        for filename in os.listdir(UPLOAD_FOLDER):
            full_path = os.path.join(UPLOAD_FOLDER, filename)
            # Skip files already processed? Add a check if needed
            process_video(full_path)
        await asyncio.sleep(60)


if __name__ == "__main__":
    # Run FastAPI on Amy Jetson
    uvicorn.run(app, host="0.0.0.0", port=5050)
    # If you want the periodic loop too, you can start it as a background task:
    # asyncio.create_task(periodic_processing_loop())
