from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
import threading
import time
import uuid
import json
import re
from datetime import datetime
from model import MedBriefTransformer, block_size, embedding_dim, n_head, n_layer, n_embd
from utils import load_tokenizer, encode, decode, restore_text, detect_domain

app = Flask(__name__)
CORS(
    app,
    origins=['https://medbriefai.vercel.app', 'http://localhost:3000', 'http://127.0.0.1:3000'],
    supports_credentials=True,
)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["30 per minute"]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
generation_lock = threading.Lock()

try:
    tokenizer = load_tokenizer()
    print(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
except FileNotFoundError:
    print("Tokenizer files not found. Please run training first.")
    tokenizer = None

try:
    checkpoint = torch.load("model.pth", map_location=device)
    if isinstance(checkpoint, dict):
        vocab_size = checkpoint.get('vocab_size', 6000)
        model = MedBriefTransformer(
            vocab_size=vocab_size,
            n_embd=embedding_dim,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = MedBriefTransformer(
            vocab_size=6000,
            n_embd=embedding_dim,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size
        ).to(device)
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model file not found. Please run training first.")
    model = None

if model is not None and hasattr(torch, 'compile'):
    model = torch.compile(model)

def get_mode_params(mode):
    params = {
        'psych': {'temperature': 0.75, 'top_k': 35, 'top_p': 0.9, 'repetition_penalty': 1.15},
        'health': {'temperature': 0.6, 'top_k': 25, 'top_p': 0.9, 'repetition_penalty': 1.2},
        'crisis': {'temperature': 0.5, 'top_k': 20, 'top_p': 0.8, 'repetition_penalty': 1.1},
        'portfolio': {'temperature': 0.7, 'top_k': 30, 'top_p': 0.9, 'repetition_penalty': 1.2},
        'general': {'temperature': 0.85, 'top_k': 45, 'top_p': 0.95, 'repetition_penalty': 1.1}
    }
    return params.get(mode, params['general'])

def detect_crisis_mode(text):
    crisis_keywords = ['suicide', 'kill myself', 'end my life', 'want to die', 'self harm', 'hurt myself']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in crisis_keywords)

def clean_response(text):
    text = text.replace('<eos>', '').replace('<eos>', '')
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) > 1:
        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punct > len(text) * 0.6:
            text = text[:last_punct + 1]
    
    return text.strip()

def build_conversation_prompt(messages, detected_mode=None):
    system_parts = [m['content'] for m in messages if m['role'] == 'system']
    user_parts = [m['content'] for m in messages if m['role'] == 'user']
    assistant_parts = [m['content'] for m in messages if m['role'] == 'assistant']
    
    prompt_parts = []
    
    if system_parts:
        prompt_parts.append(f"[SYSTEM] {' '.join(system_parts)}")
    
    if detected_mode:
        prompt_parts.append(f"[MODE:{detected_mode.upper()}]")
    
    history_length = min(3, len(user_parts), len(assistant_parts))
    for i in range(-history_length, 0):
        if i < len(user_parts):
            prompt_parts.append(f"[USER] {user_parts[i]}")
        if i < len(assistant_parts):
            prompt_parts.append(f"[ASSISTANT] {assistant_parts[i]}")
    
    if user_parts:
        prompt_parts.append(f"[USER] {user_parts[-1]}")
    
    prompt_parts.append("[ASSISTANT]")
    
    return ' '.join(prompt_parts)

@torch.inference_mode()
def generate_response(prompt, max_tokens=150, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2):
    if not model or not tokenizer:
        return "I'm currently unavailable. Please try again later."
    
    with generation_lock:
        try:
            encoded = encode(prompt, tokenizer)
            if not encoded:
                encoded = [tokenizer.special_tokens.get('[UNK]', 1)]
            
            context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
            
            generated = model.generate(
                context,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            raw_output = decode(generated[0].tolist(), tokenizer)
            
            if '[ASSISTANT]' in raw_output:
                response = raw_output.split('[ASSISTANT]')[-1].strip()
            else:
                response = raw_output.strip()
            
            response = clean_response(response)
            
            if len(response.split()) < 12:
                response = "I want to make sure I give you a thoughtful response. Could you tell me a bit more about what you mean?"
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "I'm having trouble processing that right now. Could you try rephrasing?"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'device': device,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/v1/chat/completions', methods=['POST'])
@limiter.limit("30 per minute")
def chat_completions():
    try:
        data = request.json
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 150)
        top_k = data.get('top_k', None)
        top_p = data.get('top_p', None)
        repetition_penalty = data.get('repetition_penalty', 1.2)
        mode = data.get('mode', None)
        stream = data.get('stream', False)
        
        request_id = str(uuid.uuid4())
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        user_content = ' '.join([m['content'] for m in messages if m['role'] == 'user'])
        
        if detect_crisis_mode(user_content):
            crisis_response = "I'm concerned about what you're sharing. Please reach out to a crisis hotline immediately. In the US, you can call or text 988 to connect with trained counselors. Other countries have similar services - please search for crisis support in your area. You don't have to go through this alone."
            
            response = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "medbrief-transformer",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": crisis_response
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_content.split()),
                    "completion_tokens": len(crisis_response.split()),
                    "total_tokens": len(user_content.split()) + len(crisis_response.split())
                }
            }
            
            return jsonify(response)
        
        if not mode:
            mode = detect_domain(user_content)
        
        mode_params = get_mode_params(mode)
        temperature = temperature if temperature != 0.8 else mode_params['temperature']
        top_k = top_k if top_k is not None else mode_params['top_k']
        top_p = top_p if top_p is not None else mode_params['top_p']
        repetition_penalty = repetition_penalty if repetition_penalty != 1.2 else mode_params['repetition_penalty']
        
        prompt = build_conversation_prompt(messages, mode)
        
        if stream:
            def generate_stream():
                response_text = generate_response(
                    prompt, max_tokens, temperature, top_k, top_p, repetition_penalty
                )
                
                chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "medbrief-transformer",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": response_text},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(generate_stream(), mimetype='text/plain')
        else:
            output = generate_response(
                prompt, max_tokens, temperature, top_k, top_p, repetition_penalty
            )
            
            response = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "medbrief-transformer",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(output.split()),
                    "total_tokens": len(prompt.split()) + len(output.split())
                }
            }
            
            return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
