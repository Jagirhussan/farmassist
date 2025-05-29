from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# the changed port
BACKEND_URL = "http://localhost:5050/ask_llm"  # Backend running on your laptop



@app.route('/', methods=['GET', 'POST'])
def home():
	user_input = ""
	response = ""

	if request.method == 'POST':
		user_input = request.form['input_text']
        
	try:
		# Send prompt to backend
		res = requests.post(BACKEND_URL, json={"prompt": user_input})
		response_data = res.json()
		response = response_data.get("output", "No response from backend.")
	except Exception as e:
		response = f"Error: {e}"

	return render_template('index.html', input=user_input, output=response)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3000)  # Run frontend on a different port

