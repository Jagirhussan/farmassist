# backend_main.py (FastAPI)
from fastapi import FastAPI, Request
import requests
import uvicorn

app = FastAPI()

JETSON_URL = "http://172.23.98.136:8000/run_llm"

@app.post("/ask_llm")  # Must match what the frontend calls
async def ask_llm(request: Request):
	data = await request.json()
	prompt = data.get("prompt", "")
	try:
		res = requests.post(JETSON_URL, json={"prompt": prompt})
		return res.json()
	except Exception as e:
		return {"error": str(e)}

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=5000)

