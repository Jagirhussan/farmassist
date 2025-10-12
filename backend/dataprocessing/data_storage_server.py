from fastapi import FastAPI, Request
import chromadb
from chromadb.config import Settings

app = FastAPI()

# Initialize ChromaDB with the new client configuration
client = chromadb.PersistentClient(path="video_db")
collection = client.get_or_create_collection(name="video_frames")


@app.post("/store_data")
async def store_data(request: Request):
    """
    Endpoint to receive and store processed data.
    """
    try:
        data = await request.json()
        captions = data.get("captions", [])
        embeddings = data.get("embeddings", [])
        video_path = data.get("video_path", "unknown")

        # Add data to ChromaDB
        for i, caption in enumerate(captions):
            collection.add(
                documents=[caption],
                embeddings=[embeddings[i]],
                ids=[f"{video_path}_frame_{i}"],
            )
        return {"message": "Data stored successfully"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5051)
