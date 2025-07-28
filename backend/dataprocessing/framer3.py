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
    global processor, model, db

    if processor is None or model is None or db is None or model_encoder is None:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")
        db = chromadb.PersistentClient(path="video_db")
        model_encoder = SentenceTransformer('all-MiniLM-L6-v2')

def framer(video_path):

    # load the video from the path and initialize the video capture
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frame_index = 0

    # initialise the model
    load_models()

    # initialise the time
    start_time = get_video_creation_time(video_path)

    # being splitting the video into frames and processing each frame
    while video.isOpened():
    
        ret, frame = video.read()
        if not ret:
            break
        
        # every 30 seconds, save the frame and process it
        if frame_count % int(fps * 30) == 0:
            # save the frame as a rbg array
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            inputs = processor(images=image, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(**inputs)
                response = processor.decode(output[0], skip_special_tokens=True)   
        
            # embed the response and save it to a database.
            embedded_response = model_encoder.encode(response).tolist()

            collection = db.get_or_create_collection(name="video_frames")

            collection.add(documents=response,
                   embeddings=embedded_response,
                   ids=[f"{start_time + timedelta(seconds=saved_frame_index)}"])

        
        saved_frame_index += 1
        frame_count += 1

    video.release()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 framer3.py <video/path/to.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    framer(video_path)
    print("Finished extracting frames and metadata.")




        





