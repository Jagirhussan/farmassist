# backend_main.py (FastAPI)
from fastapi import FastAPI, Request
import requests
import uvicorn

app = FastAPI()

JETSON_URL = "http://172.23.98.136:8000/run_llm"

@app.post("/ask_llm")
async def ask_llm(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    
    #call llm here i guess?

    print(f"[Backend] Got prompt from frontend: {prompt}")
    
    try:
        res = requests.post(JETSON_URL, json={"prompt": prompt})
        res.raise_for_status()  # Catch HTTP errors
        return res.json()
    except Exception as e:
        print(f"[Backend] Error calling Jetson: {e}")
        return {"output": f"Backend error: {e}"}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
