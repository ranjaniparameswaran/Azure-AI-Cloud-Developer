import os
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)

# Read configuration from environment variables
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_API_URL")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_MODEL_NAME")

# Persistent storage path in Azure App Service
CHAT_HISTORY_DIR = "/home/chat_history"

# Create directory if it does not exist
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Azure OpenAI v1 client
client = OpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({
            "error": "Request body must contain a 'message' field."
        }), 400

    user_message = data.get("message", "")

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=500,
            temperature=0.7,
        )

        assistant_reply = response.choices[0].message.content

        # Create timestamp
        now = datetime.now(timezone.utc)

        # Create chat record
        chat_record = {
            "timestamp": now.isoformat(),
            "user_message": user_message,
            "assistant_reply": assistant_reply
        }

        # Save chat history to persistent storage
        filename = f"chat_{now.strftime('%Y%m%d%H%M%S%f')}.json"
        file_path = os.path.join(CHAT_HISTORY_DIR, filename)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(chat_record, file, indent=4, ensure_ascii=False)

        return jsonify({
            "model": response.model,
            "reply": assistant_reply,
            "saved_file": file_path
        })

    except Exception as e:
        app.logger.exception("Azure OpenAI request failed")

        return jsonify({
            "error": "Azure OpenAI request failed",
            "details": str(e)
        }), 500


@app.route("/history", methods=["GET"])
def history():

    try:
        files = os.listdir(CHAT_HISTORY_DIR)

        return jsonify({
            "stored_files": files
        })

    except Exception as e:
        app.logger.exception("Failed to retrieve chat history")

        return jsonify({
            "error": "Failed to retrieve chat history",
            "details": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy"
    }), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )