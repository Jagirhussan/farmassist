from flask import Flask, render_template, request
import requests

# Create a Flask app instance
app = Flask(__name__)

# URL of the backend service
BACKEND_URL = "http://localhost:5050/ask_llm"  # Backend runs on port 5050

# Define the home route to handle GET and POST requests
# a home route that serves the main page and handles form submissions
@app.route('/', methods=['GET', 'POST'])

def home():
    #store user input and response 
    user_input = "" 
    response = ""

    if request.method == 'POST':  # Handle form submission
        user_input = request.form['input_text']  # Get user input from the UI
        print(f"[Frontend] User input: {user_input}")
        try:
            # Send the user input to the backend and get the response
            res = requests.post(BACKEND_URL, json={"prompt": user_input})
            
            response_data = res.json()  # Parse the backend's JSON response
            print(f"[Frontend] Parsed backend response: {response_data}")
            
            # Extract the "output" field from the backend response or no response if not present
            response = response_data.get("output", "No response from backend.")
        except Exception as e:
            # Handle error
            response = f"Error: {e}"

    # Render the HTML template with user input and backend response
    return render_template('index.html', input=user_input, output=response)

# Run the Flask app if this script is executed directly which it needs to be
if __name__ == '__main__':
    print("[Frontend] Starting Flask app on http://localhost:3000 ...")
    app.run(host='0.0.0.0', port=3000, debug=True)


