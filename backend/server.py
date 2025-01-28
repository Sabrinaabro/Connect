from flask import Flask, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)

@app.route('/start-gesture', methods=['GET'])
def start_gesture():
    try:
        subprocess.Popen(["python", "scripts/real_time_recognition.py"])
        return jsonify({"message": "Real-time gesture model started!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔥 Move this block outside the function  
if __name__ == '__main__':
    print("🔥 Flask server is starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)  # Debug mode enabled
