import torch
import open_clip
import os
import json
from PIL import Image
import chromadb
import time
from memory_profiler import profile

def embed_frame(frame_path, model, preprocess, device):
    """
    Embed a single image using the OpenCLIP model.

    Args:
        frame_path (str): Path to the image file.
        model: Loaded OpenCLIP model.
        preprocess: Preprocessing transform for the model.
        device: Torch device.

    Returns:
        np.ndarray: The embedding of the image.
    """
    img = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img)
    return features.squeeze().cpu().numpy()

@profile
def process_frames_in_folder(folder_path, model, preprocess, device, collection_name="video_frames"):
    """
    Process all valid image frames in a folder and store their embeddings + metadata in ChromaDB.

    Args:
        folder_path (str): Path to the folder containing image and JSON metadata files.
        model: Loaded OpenCLIP model.
        preprocess: Preprocessing transform for the model.
        device: Torch device.
        collection_name (str): Name of the ChromaDB collection to store in.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path is not a directory: {folder_path}")

    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path="video_db")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Only look for .jpg files
    image_extensions = {".jpg"}
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

    for idx, image_file in enumerate(sorted(image_files)):
        image_path = os.path.join(folder_path, image_file)
        frame_id = os.path.splitext(image_file)[0]
        json_path = os.path.join(folder_path, frame_id + ".json")

        if not os.path.isfile(json_path):
            print(f"Warning: JSON metadata not found for {image_file}, skipping.")
            continue

        # Embed image
        embedding = embed_frame(image_path, model, preprocess, device)

        with open(json_path, "r") as f:
            metadata = json.load(f)

        collection.add(ids=[frame_id], embeddings=[embedding], metadatas=[metadata])
        print(f"Embedded and stored: {image_file}")



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 embed_frames.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    # Load model once
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Run processing
    starttime = time.time()
    process_frames_in_folder(folder_path, model, preprocess, device)
    elapsed_time = time.time() - starttime
    print(f"Total Processing Time: {elapsed_time:.2f} seconds")
