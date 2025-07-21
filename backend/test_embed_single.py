# test_embed_single.py - Test embedding a single frame to debug the issue
import torch
import open_clip
import os
import json
from PIL import Image
import chromadb
import numpy as np

def test_single_frame_embedding():
    """Test embedding a single frame to debug issues"""
    
    # Load the same model as embed_frame.py
    print("Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Test frame path
    frame_path = "dataprocessing/frames/frame_0001.jpg"
    json_path = "dataprocessing/frames/frame_0001.json"
    
    if not os.path.exists(frame_path):
        print(f"Frame not found: {frame_path}")
        return
    
    if not os.path.exists(json_path):
        print(f"JSON not found: {json_path}")
        return
    
    # Load and embed the image
    print(f"Embedding frame: {frame_path}")
    img = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.encode_image(img)
        # Normalize the features for proper cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
    
    embedding = features.squeeze().cpu().numpy()
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    print(f"Embedding norm: {np.linalg.norm(embedding):.3f}")
    
    # Load metadata
    with open(json_path, "r") as f:
        metadata = json.load(f)
    
    print(f"Metadata: {metadata}")
    
    # Test ChromaDB storage
    print("\nTesting ChromaDB storage...")
    chroma_client = chromadb.PersistentClient(path="video_db")
    
    # Delete and recreate collection to start fresh
    try:
        chroma_client.delete_collection(name="test_collection")
    except:
        pass
    
    collection = chroma_client.create_collection(name="test_collection")
    
    # Add the embedding
    frame_id = "frame_0001"
    collection.add(
        ids=[frame_id], 
        embeddings=[embedding.tolist()], 
        metadatas=[metadata]
    )
    
    print("Successfully added to ChromaDB")
    
    # Test querying back
    print("\nTesting query...")
    
    # Embed a text query
    query = "person typing"
    text_tokens = open_clip.tokenize([query]).to(device)
    
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        # Normalize the text embedding too
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    
    query_embedding = text_embedding.squeeze().cpu().numpy()
    
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding range: [{query_embedding.min():.3f}, {query_embedding.max():.3f}]")
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1
    )
    
    if results['ids'] and len(results['ids'][0]) > 0:
        distance = results['distances'][0][0]
        similarity = 1 - distance
        print(f"Distance: {distance:.3f}")
        print(f"Similarity: {similarity:.3f}")
        print(f"Found frame: {results['ids'][0][0]}")
        print(f"Metadata: {results['metadatas'][0][0]}")
    else:
        print("No results found")

if __name__ == "__main__":
    test_single_frame_embedding()
