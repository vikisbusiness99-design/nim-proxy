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
        
        # UPDATED: Force minimum tokens to prevent cutoff issues
        requested_max_tokens = openai_request.get('max_tokens')
        if requested_max_tokens is None:
            max_tokens = 8192  # Default to 8K for good roleplay responses
        elif requested_max_tokens < 500:
            max_tokens = 2048  # Force at least 2K tokens minimum to prevent cutoff
        else:
            max_tokens = min(requested_max_tokens, 65536)  # Cap at 64K max
        
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
                timeout=300  # UPDATED: 5 minutes timeout for long responses
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
            
            # IMPROVED: Fix DeepSeek R1 reasoning_content issue
            if 'choices' in nvidia_response:
                for choice in nvidia_response['choices']:
                    if 'message' in choice:
                        msg = choice['message']
                        # FIXED: Always combine reasoning_content properly
                        if msg.get('reasoning_content'):
                            reasoning = msg.get('reasoning_content', '').strip()
                            content = msg.get('content', '').strip()
                            
                            # Combine reasoning and content - ALWAYS prefer showing both
                            if reasoning and content:
                                # Both exist - combine them
                                msg['content'] = f"{reasoning}\n\n{content}"
                            elif reasoning:
                                # Only reasoning exists
                                msg['content'] = reasoning
                            elif content:
                                # Only content exists (shouldn't happen but just in case)
                                msg['content'] = content
                            
                            # Debug logging
                            print(f"Original content: {content[:100]}...")
                            print(f"Reasoning length: {len(reasoning)} chars")
                            print(f"Final combined content length: {len(msg['content'])} characters")
                            
                            # Remove reasoning_content to avoid confusion
                            msg.pop('reasoning_content', None)
                        
                        # Additional check: if content is still empty, provide fallback
                        if not msg.get('content') or msg.get('content').strip() == '':
                            print("WARNING: Empty content detected, using fallback")
                            msg['content'] = "I apologize, but I couldn't generate a proper response. Please try again."
            
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
