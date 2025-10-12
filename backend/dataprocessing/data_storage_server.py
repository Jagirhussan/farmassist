from xmlrpc import client
from fastapi import FastAPI, Request
import chromadb
from chromadb.config import Settings

app = FastAPI()


@app.post("/store_data")
async def store_data(request: Request):
    """
    Endpoint to receive and store processed data.
    """

    # Initialize ChromaDB with the new client configuration
    client = chromadb.PersistentClient(path="dataprocessing/video_db")
    # get or create the collection
    collection = client.get_or_create_collection(name="video_frames")

    try:
        data = await request.json()

        print(f"[Storage] Received data: {data}")
        print(f"[Storage] Storing {len(data)} items to ChromaDB...")
        for item in data:
            print(f"[Storage] Sample item: {item}")

        # Add data to ChromaDB
        for item in data:
            collection.add(
                ids=[item["ids"][0]],
                documents=[item["documents"][0]],
                embeddings=[item["embeddings"][0]],
            )
        return {"message": "Data stored successfully"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5051)
