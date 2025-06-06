# Import necessary modules for FastAPI, HTTP requests, and running the app
from fastapi import FastAPI, Request
import requests
import uvicorn

# Create a FastAPI app instance
app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL of the Jetson server to handle LLM requests
JETSON_URL = "http://172.23.98.136:8000/run_llm"

# Define a POST endpoint to handle LLM prompts ie sending the request to the Jetson which calles the LLM
@app.post("/ask_llm")
async def ask_llm(request: Request):
    # Get JSON data from the request
    data = await request.json()
    # Extract the "prompt" field, default to empty if not provided
    prompt = data.get("prompt", "")
    
    # Log the received prompt - this is debugging info 
    print(f"[Backend] Got prompt from frontend: {prompt}")
    
    try:
        # Send the prompt to the Jetson server and get the response
        res = requests.post(JETSON_URL, json={"prompt": prompt})
        res.raise_for_status()  # Handle HTTP errors
        return res.json()  # get the response from the Jetson server as JSON format
    except Exception as e:
        # error catching and debugging
        print(f"[Backend] Error calling Jetson: {e}")
        return {"output": f"Backend error: {e}"}

# Run the app if this script is executed directly, its on port 5050 and will be called on the frontend
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
