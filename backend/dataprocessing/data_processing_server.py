import os
import time
import requests
from framer3 import framer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATA_STORAGE_IP = os.getenv("REACT_APP_ALEX_IP")  # IP of the data storage server
DATA_STORAGE_PORT = 5051  # Port for the data storage server


def send_to_storage(data):
    """
    Send processed data to the data storage server.
    Args:
        data (dict): The data to send (captions, embeddings, etc.).
    """
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
    """
    Process the video and send data to the storage server.
    Args:
        video_path (str): Path to the video file.
    """
    # Call the framer function to process the video
    print(f"Processing video: {video_path}")
    framer(video_path)  # This will process the video and save data to ChromaDB locally

    # Simulate sending data to the storage server (replace with actual data from ChromaDB)
    data = {
        "video_path": video_path,
        "captions": ["Caption 1", "Caption 2"],  # Replace with actual captions
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],  # Replace with actual embeddings
    }
    send_to_storage(data)


if __name__ == "__main__":
    # Example: Process a video every minute
    while True:
        video_path = "/path/to/video.mp4"  # Replace with the actual video path
        process_video(video_path)
        time.sleep(60)  # Wait for 1 minute before processing the next video
