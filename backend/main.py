# backend/main.py
from fastapi import FastAPI, Request
import uvicorn
from llm_utils import run_llm

app = FastAPI()

@app.post("/ask_llm")
async def ask_llm(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    print(f"[Backend] Got prompt: {prompt}")

    try:
        # Call your LLM here
        output = run_llm(prompt)
        return {"output": output}
    except Exception as e:
        print(f"[Backend] Error calling LLM: {e}")
        return {"output": f"Backend error: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
