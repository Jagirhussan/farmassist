#llm_server.py

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/run_llm") 
async def run_llm(request: Request):
	data = await request.json()
	
	#get the prompt from the data
	prompt = data.get("prompt","")
	
	#get the response from the LLM (this needs to be edited/ a response actually got /llm actually called idk)
	response = {"output": f"LLM processed: {prompt}"}
	print("LLM Server used")
	
	
	#debugging checking to make sure this is working - don't return this dummy if you're actually geting a response from the llm 
	print(f"Recieved prompt: {prompt}") 
	return {"output": f"LLM processed: {prompt}"}
	
	# if you're not debugging you want to be returning the actual reponse from the llm
	#return response
	
if __name__ == "__main__":
	uvicorn.run(app, host = "0.0.0.0", port = 8000)
