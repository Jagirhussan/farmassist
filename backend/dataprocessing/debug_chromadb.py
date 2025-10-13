# debug_chromadb.py - Check what's in your ChromaDB collection
import chromadb
import json

def inspect_chromadb():
    """Inspect the contents of the ChromaDB collection"""
    try:
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path="video_db")
        collection = chroma_client.get_or_create_collection(
            name="video_frames"
        )
        
        results = collection.get(include=['documents', 'embeddings', 'metadatas'])

        print(f"Total items in collection: {len(results['ids'])}")

        for i in range(min(50, len(results['ids']))):
            print(f"Item {i+1}:")
            print(f"  ID: {results['ids'][i]}")
            print(f"  Embedding: {results['embeddings'][i][:5]}...")
            print(f"  Document: {results['documents'][i][:100]}...")
            print(f"  Metadata: {json.dumps(results['metadatas'][i], indent=2)}")


        
    except Exception as e:
        print(f"Error inspecting ChromaDB: {e}")
        return {}

if __name__ == "__main__":
    inspect_chromadb()
