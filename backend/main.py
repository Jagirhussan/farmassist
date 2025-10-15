from fastapi import FastAPI, Request
from fastapi import File, UploadFile
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from llm_utils import run_llm, load_models
from contextlib import asynccontextmanager
import time

UPLOAD_FOLDER = "backend/dataprocessing/videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code here (e.g., load models)
    start_time = time.time()
    load_models()
    end_time = time.time()
    print(f"[LLM] Models loaded in {end_time - start_time:.2f} seconds", flush=True)
    yield
    # Shutdown code here


app = FastAPI(lifespan=lifespan)

# Enable CORS for all origins and allow POST with Content-Type headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (for dev/testing)
    allow_credentials=True,
    allow_methods=["POST"],  # Only allow POST methods
    allow_headers=["Content-Type"],  # Allow Content-Type header in requests
)


@app.post("/ask_llm")
async def ask_llm(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    print(f"[Backend] Got prompt: {prompt}")

    try:
        start_time = time.time()
        output = run_llm(prompt)
        end_time = time.time()
        print(f"[LLM] Inference time: {end_time - start_time:.2f} seconds", flush=True)
        return {"output": output}
    except Exception as e:
        print(f"[Backend] Error calling LLM: {e}")
        return {"output": f"Backend error: {e}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
