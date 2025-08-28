import os
from dotenv import load_dotenv
from flask import Flask, jsonify

app = Flask(__name__)
load_dotenv()


@app.route("/config")
def get_config():
    return jsonify({"alex_ip": os.getenv("ALEX_IP")})  # e.g., 172.23.6.60
