from flask import Flask, request, jsonify, Response
import requests
import os
import time
import json

app = Flask(__name__)

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
    start_time = time.time()
    print(f"[TIMING] Request started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        openai_request = request.get_json()
        
        messages = openai_request.get('messages', [])
        model = openai_request.get('model', 'deepseek-ai/deepseek-r1')
        temperature = openai_request.get('temperature', 0.7)
        
        requested_max_tokens = openai_request.get('max_tokens')
        if requested_max_tokens is None:
            max_tokens = 8192
        elif requested_max_tokens < 500:
            max_tokens = 2048
        else:
            max_tokens = min(requested_max_tokens, 65536)
        
        stream = openai_request.get('stream', False)
        print(f"[STREAMING] Stream mode: {stream}")
        
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
        
        print(f"[TIMING] Sending request to NVIDIA API...")
        api_start = time.time()
        
        if stream:
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nim_request,
                stream=True,
                timeout=300
            )
            
            print(f"[TIMING] Stream connection established in {time.time() - api_start:.2f}s")
            
            def generate():
                try:
                    chunk_count = 0
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        try:
                            decoded_line = line.decode('utf-8').strip()
                        except:
                            continue
                        
                        if not decoded_line:
                            continue
                        
                        if decoded_line.startswith('data: '):
                            decoded_line = decoded_line[6:]
                        
                        if decoded_line == '[DONE]':
                            print(f"[STREAMING] Received [DONE], sent {chunk_count} chunks")
                            yield b'data: [DONE]\n\n'
                            break
                        
                        try:
                            chunk_data = json.loads(decoded_line)
                            
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                for choice in chunk_data['choices']:
                                    if 'delta' in choice:
                                        delta = choice['delta']
                                        
                                        if 'reasoning_content' in delta:
                                            del delta['reasoning_content']
                                        
                                        if not delta.get('content') and delta.keys() == {'role'}:
                                            continue
                                    
                                    for field in ['mm_embedding_handle', 'disaggregated_params', 'avg_decoded_tokens_per_iter', 'stop_reason']:
                                        if field in choice:
                                            del choice[field]
                            
                            if 'prompt_token_ids' in chunk_data:
                                del chunk_data['prompt_token_ids']
                            
                            chunk_count += 1
                            
                            chunk_json = json.dumps(chunk_data)
                            yield f'data: {chunk_json}\n\n'.encode('utf-8')
                            
                            if chunk_count == 1:
                                print(f"[STREAMING] First chunk sent successfully")
                            
                        except json.JSONDecodeError as e:
                            print(f"[WARNING] Failed to parse chunk: {decoded_line[:100]}")
                            continue
                    
                    elapsed = time.time() - start_time
                    print(f"[TIMING] Stream completed in {elapsed:.2f} seconds, sent {chunk_count} chunks")
                    
                except Exception as e:
                    print(f"[ERROR] Stream error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    error_chunk = {
                        "id": "error",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n\n[Error: {str(e)}]"},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f'data: {json.dumps(error_chunk)}\n\n'.encode('utf-8')
                    yield b'data: [DONE]\n\n'
            
            return Response(generate(), 
                          mimetype='text/event-stream',
                          headers={
                              'Cache-Control': 'no-cache',
                              'X-Accel-Buffering': 'no'
                          })
        else:
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nim_request,
                timeout=300
            )
            
            api_time = time.time() - api_start
            print(f"[TIMING] NVIDIA API responded in {api_time:.2f}s")
            
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
            
            print("=" * 50)
            print(f"Full NVIDIA Response: {nvidia_response}")
            print("=" * 50)
            
            try:
                if 'choices' in nvidia_response and len(nvidia_response['choices']) > 0:
                    for choice in nvidia_response['choices']:
                        if 'message' in choice:
                            msg = choice['message']
                            
                            if 'content' in msg and msg['content']:
                                msg['content'] = msg['content'].strip()
                            
                            if 'refusal' not in msg:
                                msg['refusal'] = None
                            
                            if 'reasoning_content' in msg:
                                reasoning = msg.get('reasoning_content', '').strip()
                                content = msg.get('content', '').strip()
                                
                                if content:
                                    msg['content'] = content
                                elif reasoning:
                                    msg['content'] = reasoning
                                
                                print(f"Original content: '{content}'")
                                print(f"Final content: '{msg['content']}'")
                                
                                del msg['reasoning_content']
                            
                            if 'tool_calls' in msg:
                                if not msg['tool_calls'] or len(msg['tool_calls']) == 0:
                                    del msg['tool_calls']
                            
                            for field in ['mm_embedding_handle', 'disaggregated_params']:
                                if field in msg:
                                    del msg[field]
                            
                            if not msg.get('content'):
                                msg['content'] = "I apologize, but I couldn't generate a response. Please try again."
                                print("WARNING: Empty content, using fallback")
                        
                        for field in ['mm_embedding_handle', 'disaggregated_params', 'avg_decoded_tokens_per_iter', 'stop_reason']:
                            if field in choice:
                                del choice[field]
                
                if 'prompt_token_ids' in nvidia_response:
                    del nvidia_response['prompt_token_ids']
                
            except Exception as e:
                print(f"Error processing response: {e}")
                import traceback
                traceback.print_exc()
            
            print("=" * 50)
            print(f"Sending back to Janitor: {nvidia_response}")
            if nvidia_response.get('choices'):
                print(f"First choice content: '{nvidia_response['choices'][0]['message'].get('content', '')}'")
            print("=" * 50)
            
            elapsed = time.time() - start_time
            print(f"[TIMING] Total request time: {elapsed:.2f} seconds")
            
            return jsonify(nvidia_response), 200
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"[TIMING] Request timed out after {elapsed:.2f} seconds")
        return jsonify({"error": "Request timeout"}), 504
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - start_time
        print(f"[TIMING] Request failed after {elapsed:.2f} seconds")
        return jsonify({"error": f"Request failed: {str(e)}"}), 500
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[TIMING] Unexpected error after {elapsed:.2f} seconds")
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
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
