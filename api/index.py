from flask import Flask, request, jsonify, Response
import requests
import os

app = Flask(__name__)

# Add CORS support for mobile apps
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/', methods=['GET'])
@app.route('/api', methods=['GET'])
@app.route('/api/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "NVIDIA NIM Proxy is running"})

@app.route('/api/v1/chat/completions', methods=['POST', 'OPTIONS'])
@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        openai_request = request.get_json()
        
        messages = openai_request.get('messages', [])
        model = openai_request.get('model', 'deepseek-ai/deepseek-r1')
        temperature = openai_request.get('temperature', 0.7)
        max_tokens = openai_request.get('max_tokens', 1024)
        stream = openai_request.get('stream', False)
        
        nim_request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
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
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nim_request,
                timeout=60
            )
            
            # Check if request was successful
            if response.status_code != 200:
                error_response = {
                    "error": {
                        "message": f"NVIDIA API error: {response.status_code} - {response.text}",
                        "type": "nvidia_api_error",
                        "code": response.status_code
                    }
                }
                return jsonify(error_response), response.status_code
            
            nvidia_response = response.json()
            
            # Log the response for debugging (you can check Vercel logs)
            print(f"NVIDIA Response: {nvidia_response}")
            
            # Fix DeepSeek R1 reasoning_content issue
            if 'choices' in nvidia_response:
                for choice in nvidia_response['choices']:
                    if 'message' in choice:
                        msg = choice['message']
                        # If content is empty but reasoning_content exists, combine them
                        if msg.get('content') == '' and msg.get('reasoning_content'):
                            # Combine reasoning and content
                            msg['content'] = msg.get('reasoning_content', '')
                        elif msg.get('reasoning_content'):
                            # If both exist, combine them
                            msg['content'] = f"{msg.get('reasoning_content', '')}\n\n{msg.get('content', '')}"
            
            # Return the response as-is (NVIDIA API should already be OpenAI compatible)
            return jsonify(nvidia_response), 200
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timeout - model took too long to respond"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/api/v1/models', methods=['GET', 'OPTIONS'])
@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
        response = requests.get(f"{NVIDIA_BASE_URL}/models", headers=headers)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({
            "object": "list",
            "data": [
                {"id": "deepseek-ai/deepseek-r1", "object": "model"},
                {"id": "deepseek-ai/deepseek-r1-distill-llama-8b", "object": "model"},
                {"id": "deepseek-ai/deepseek-r1-distill-qwen-32b", "object": "model"},
                {"id": "deepseek-ai/deepseek-r1-distill-qwen-14b", "object": "model"}
            ]
        }), 200
