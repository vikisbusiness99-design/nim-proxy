from flask import Flask, request, jsonify, Response
import requests
import json
import os

app = Flask(__name__)

# Get NVIDIA API key from environment variable
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        # Get the OpenAI-formatted request
        openai_request = request.json
        
        # Extract parameters
        messages = openai_request.get('messages', [])
        model = openai_request.get('model', 'deepseek/deepseek-r1')
        temperature = openai_request.get('temperature', 0.7)
        max_tokens = openai_request.get('max_tokens', 1024)
        stream = openai_request.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nim_request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add optional parameters if present
        if 'top_p' in openai_request:
            nim_request['top_p'] = openai_request['top_p']
        if 'frequency_penalty' in openai_request:
            nim_request['frequency_penalty'] = openai_request['frequency_penalty']
        if 'presence_penalty' in openai_request:
            nim_request['presence_penalty'] = openai_request['presence_penalty']
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            # Handle streaming response
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nim_request,
                stream=True
            )
            
            def generate():
                for line in response.iter_lines():
                    if line:
                        yield line + b'\n'
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # Handle non-streaming response
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nim_request
            )
            
            return jsonify(response.json()), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}"
        }
        
        response = requests.get(
            f"{NVIDIA_BASE_URL}/models",
            headers=headers
        )
        
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({
            "object": "list",
            "data": [
                {"id": "deepseek/deepseek-r1", "object": "model"},
                {"id": "deepseek/deepseek-r1-distill-llama-70b", "object": "model"},
                {"id": "deepseek/deepseek-r1-distill-qwen-32b", "object": "model"},
                {"id": "deepseek/deepseek-r1-distill-llama-8b", "object": "model"},
                {"id": "nvidia/llama-3.1-nemotron-70b-instruct", "object": "model"},
                {"id": "meta/llama-3.1-8b-instruct", "object": "model"},
                {"id": "meta/llama-3.1-70b-instruct", "object": "model"}
            ]
        }), 200

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "NVIDIA NIM Proxy is running"})

# Vercel serverless function handler
handler = app
