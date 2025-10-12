from fastapi import FastAPI, Request
import chromadb
from chromadb.config import Settings

app = FastAPI()

# Initialize ChromaDB with the new client configuration
client = chromadb.PersistentClient(path="dataprocessing/video_db")
    # get or create the collection
collection = client.get_or_create_collection(name="video_frames")


@app.post("/store_data")
async def store_data(request: Request):
    """
    Endpoint to receive and store processed data.
    """
    try:
        data = await request.json()

        # Add data to ChromaDB
        for i in range(len(data['id'])):
            collection.add(
                id=[data[i]['id']],
                caption=[data[i]['captions']],
                embeddings=[data[i]['embeddings']],
            )
        return {"message": "Data stored successfully"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5051)
