#llm_server.py

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/run_llm") 
async def run_llm(request: Request):
	data = await request.json()
	
	#get the prompt from the data
	prompt = data.get("prompt","")
	
	#get the response from the LLM (this needs to be edited)
	response = {"output": f"LLM processed: {prompt}"}
	
	
	#debugging checking to make sure this is working
	print(f"Recieved prompt: {prompt}") 
	return {"output": f"LLM processed: {prompt}"}
	
	#return response
	
if __name__ == "__main__":
	uvicorn.run(app, host = "0.0.0.0", port = 8000)
