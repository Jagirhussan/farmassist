# this file embeds the text query using the OpenCLIP model.
import sys
import os
import open_clip
import torch

# Load the OpenCLIP model and tokenizer - MUST match embed_frame.py
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')    
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def embed_query(query):
    """
    Embed a text query using the OpenCLIP model.
    
    Args:
        query (str): The text query to embed.
        
    Returns:
        np.ndarray: The embedding of the text query. 
    
        
    Author: Alex Foster
    Date: 2023-10-01
    """

    # tokenize the query
    text_tokens = open_clip.tokenize([query]).to(device)

    # get the embedding
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)

    return text_embedding.squeeze().cpu().numpy()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 embed_query.py <query.txt>")
        sys.exit(1)

    # Get the query from command line arguments
    query = sys.argv[1]
    
    # Embed the query
    embedding = embed_query(query)
    
    # Print the embedding
    print("Embedding:", embedding)