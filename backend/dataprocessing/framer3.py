# framer3.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from extract_time import get_video_creation_time
import cv2
from datetime import datetime, timedelta
import os
import json
from PIL import Image
import torch
import shutil
import sys


def load_model(model_name: str = "princeton-nlp/Sheared-LLaMA-2.7B"):
    """Load the LLM model once when the server starts"""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set up pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
def framer(video_path):

    if os.path.exists("frames"):
        # clear the directory if it exists
        shutil.rmtree("frames")
    os.makedirs("frames", exist_ok=True)

    # input model name
    model_name = "princeton-nlp/Sheared-LLaMA-2.7B"

    # load the video from the path and initialize the video capture
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frame_index = 0

    # initialise the model
    load_model(model_name)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."
        "Provide one answer ONLY the following query based on the context provided below. "
        "Do not generate or answer any other questions. "
        "Do not make up or infer any information that is not directly stated in the context. "
        "Provide a concise answer."},
        {"role": "user", "content": "Explain what is happening in the image."}
    ]

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

            inputs = tokenizer(
                messages,
                image=image,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # save the .json metadata of the frame with the response
            metadata = {
                "frame_index": saved_frame_index,
                "timestamp": (start_time + timedelta(seconds=saved_frame_index+30)).isoformat(),
                "response": response
            }

            # save the metadata as a .json file
            with open(f"frame_{saved_frame_index}.json", "w") as f:
                json.dump(metadata, f)
        
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




        





