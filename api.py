from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle
from model import TinyTransformer, block_size, embedding_dim
from utils import encode, decode, restore_text

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load vocab and model
with open("vocab.pkl", "rb") as f:
    stringtoindex, indextostring = pickle.load(f)

vocab_size = len(stringtoindex)
model = TinyTransformer(vocab_size, embedding_dim, block_size).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

@torch.inference_mode()
def generate(model, prompt, max_tokens=150, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2):
    encoded = encode(prompt, stringtoindex)
    if not encoded:
        # Default to index 0 if unknown tokens or empty
        encoded = [0]
    
    context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    generated = context

    for _ in range(max_tokens):
        idx_condensed = generated[:, -block_size:]
        logits = model(idx_condensed)
        logits = logits[:, -1, :] / temperature

        for token_id in set(generated[0].tolist()):
            # Prevent out of bounds error
            if token_id < vocab_size:
                logits[0, token_id] /= repetition_penalty

        probs = torch.softmax(logits, dim=-1)

        if top_k:
            top_probs, top_idx = torch.topk(probs, k=top_k)
            probs = torch.zeros_like(probs).scatter(1, top_idx, top_probs)
            probs /= probs.sum(dim=-1, keepdim=True)

        if top_p and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > top_p

            if torch.any(cutoff):
                cutoff_idx = cutoff.nonzero()[0][0].item() + 1
                sorted_probs = sorted_probs[:, :cutoff_idx]
                sorted_idx = sorted_idx[:, :cutoff_idx]
                probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
                probs /= probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    # Decode and restore original formatting exactly like generate.py
    raw_output = decode(generated[0].tolist(), indextostring)
    restored = restore_text(raw_output.split())
    return restored

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 150)
        
        # Get the latest user message
        prompt = ""
        for m in reversed(messages):
            if m['role'] == 'user':
                prompt = m['content']
                break
        
        if not prompt:
            prompt = messages[-1]['content'] if messages else ""

        # Run custom trained LLM inference
        output = generate(model, prompt, max_tokens=max_tokens, temperature=temperature, top_k=40, top_p=0.9, repetition_penalty=1.1)

        # Make it look like OpenAI JSON so engine.js seamlessly handles it
        response = {
            "id": "chatcmpl-custom",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "recall-custom-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output
                    },
                    "finish_reason": "stop"
                }
            ],
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
    app.run(host='127.0.0.1', port=5000)
