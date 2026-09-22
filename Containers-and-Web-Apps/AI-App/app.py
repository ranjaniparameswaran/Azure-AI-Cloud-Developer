import os
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)

# Read configuration from environment variables
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_API_URL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_MODEL_NAME")

# Create Azure OpenAI client using the v1 endpoint
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
                },
            ],
            temperature=0.7,
        )

        return jsonify({
            "model": response.model,
            "reply": response.choices[0].message.content
        })

    except Exception as e:
        app.logger.exception("Azure OpenAI request failed")

        return jsonify({
            "error": "Azure OpenAI request failed",
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