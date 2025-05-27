import torch
import open_clip
import sys
import chromadb
import os
from PIL import Image
import json

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

chroma_client = chromadb.PersistentClient(path="video_db")
collection = chroma_client.get_or_create_collection(name="video_frames")


def embed_frame(frame):
	img = preprocess(Image.open(frame)).unsqueeze(0).to(device)
	with torch.no_grad():
		features = model.encode_image(img)
	return features.squeeze().cpu().numpy() 

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python3 embed_frames.py <folder_path>")
		sys.exit(1)

	folder_path = sys.argv[1]
	if not os.path.isdir(folder_path):
		print(f"Error: {folder_path} is not a valid directory.")
		sys.exit(1)

	image_extensions = {".jpg"}

	image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

	for idx, image_file in enumerate(sorted(image_files)):
		image_path = os.path.join(folder_path, image_file)
		embedding = embed_frame(image_path)

		# Example: use filename as ID, and store filename as metadata
		frame_id = os.path.splitext(image_file)[0]
		json_path = os.path.join(folder_path, frame_id + ".json")
		
		with open(json_path, "r") as f:
			metadata = json.load(f)

		# Add to ChromaDB
		collection.add(ids=[frame_id], embeddings=[embedding], metadatas=[metadata])

		print(f"Embedded and stored: {image_file}")
