# video_query.py - Query the video frame database for contextual LLM responses
import chromadb
import open_clip
import torch
import numpy as np
from typing import List, Dict, Any
import json

class VideoFrameRetriever:
    def __init__(self, db_path="video_db", collection_name="video_frames"):
        """Initialize the video frame retriever with ChromaDB and OpenCLIP model"""
        # Setup ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # Load the same OpenCLIP model used for embedding
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='laion2b_s34b_b79k'  # Same as embed_frame.py
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        print(f"[VideoRetriever] Initialized with device: {self.device}")
    
    def embed_text_query(self, query: str) -> np.ndarray:
        """Embed a text query using OpenCLIP"""
        text_tokens = open_clip.tokenize([query]).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
        
        return text_embedding.squeeze().cpu().numpy()
    
    def search_similar_frames(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for frames similar to the text query
        
        Args:
            query (str): Text description to search for
            n_results (int): Number of similar frames to return
            
        Returns:
            List of dictionaries containing frame metadata and similarity scores
        """
        # Embed the query
        query_embedding = self.embed_text_query(query)
        
        # Search ChromaDB for similar frames
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        similar_frames = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                frame_info = {
                    'frame_id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                similar_frames.append(frame_info)
        
        return similar_frames
    
    def get_contextual_info(self, query: str, n_results: int = 3) -> str:
        """
        Get contextual information from video frames for LLM prompting
        
        Args:
            query (str): User's query
            n_results (int): Number of frames to retrieve
            
        Returns:
            Formatted string with contextual information
        """
        similar_frames = self.search_similar_frames(query, n_results)
        
        if not similar_frames:
            return "No relevant video context found."
        
        context_parts = ["RELEVANT VIDEO CONTEXT:"]
        
        for i, frame in enumerate(similar_frames, 1):
            metadata = frame['metadata']
            similarity = 1 - frame['distance']  # Convert distance to similarity
            
            context_parts.append(
                f"\n{i}. Frame from '{metadata.get('source_video', 'unknown')}' "
                f"at {metadata.get('timestamp', 'unknown time')} "
                f"(similarity: {similarity:.2f})"
            )
        
        context_parts.append("\nBased on this video context, please provide your response.")
        
        return "\n".join(context_parts)

# Global retriever instance
video_retriever = None

def get_video_context(query: str, n_results: int = 3) -> str:
    """
    Get video context for a query - used by LLM
    
    Args:
        query (str): User's query
        n_results (int): Number of similar frames to retrieve
        
    Returns:
        Contextual information string
    """
    global video_retriever
    
    try:
        if video_retriever is None:
            print("[VideoQuery] Initializing video retriever...")
            video_retriever = VideoFrameRetriever()
        
        return video_retriever.get_contextual_info(query, n_results)
    
    except Exception as e:
        print(f"[VideoQuery] Error retrieving context: {e}")
        return "Error retrieving video context."

if __name__ == "__main__":
    # Test the retriever
    retriever = VideoFrameRetriever()
    test_query = "person walking"
    
    print(f"Testing query: '{test_query}'")
    context = retriever.get_contextual_info(test_query)
    print(f"Context:\n{context}")
