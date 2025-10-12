# framer3.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from extract_time import get_video_creation_time
import cv2
from datetime import timedelta
from PIL import Image
import torch
import sys
import chromadb
from sentence_transformers import SentenceTransformer

# Global variables to store the model (loaded once)
processor = None
model = None
db = None
model_encoder = None


def load_models():
    """Load the LLM, database, and caption encoding models once when the server starts"""
    global processor, model, db, model_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if processor is None or model is None or db is None or model_encoder is None:
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        model_encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)


def framer(video_path):

    # load the video from the path and initialize the video capture
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    # initialise the model
    load_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialise the time
    start_time = get_video_creation_time(video_path)

    # being splitting the video into frames and processing each frame
    processed_data = []  # Store processed captions and embeddings

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % int(fps * 30) == 0:

            # calculate the timestamp of the frame
            seconds = frame_count / fps
            frame_time = start_time + timedelta(seconds=seconds)
            timestamp_str = frame_time.strftime("%Y-%m-%d_%H-%M-%S")

            # save the frame as a rbg array
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)
                response = processor.decode(output[0], skip_special_tokens=True)

            # embed the response and save it to a database.
            embedded_response = model_encoder.encode(response)
            processed_data.append({
                "id": timestamp_str,
                "caption": response,
                "embedding": embedded_response
            })

            # Debug print
            print(
                f"[Debug] Frame {frame_count} @ {timestamp_str}: {response} | Embedding length: {len(embedded_response)}"
            )

        frame_count += 1

    video.release()
    return processed_data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 framer3.py <video/path/to.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    framer(video_path)
    print("Finished extracting frames and metadata.")
