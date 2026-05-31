import torch
import argparse
from model import MedBriefTransformer, block_size, embedding_dim, n_head, n_layer, n_embd
from utils import load_tokenizer, encode, decode, detect_domain

def load_model_and_tokenizer(model_path="model.pth", vocab_path="vocab.json", merges_path="merges.pkl"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = load_tokenizer(vocab_path, merges_path)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        vocab_size = checkpoint.get('vocab_size', len(tokenizer.vocab))
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
            vocab_size=len(tokenizer.vocab),
            n_embd=embedding_dim,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size
        ).to(device)
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    return model, tokenizer, device

def get_mode_params(mode):
    params = {
        'psych': {'temperature': 0.75, 'top_k': 35, 'top_p': 0.9, 'repetition_penalty': 1.15},
        'health': {'temperature': 0.6, 'top_k': 25, 'top_p': 0.9, 'repetition_penalty': 1.2},
        'crisis': {'temperature': 0.5, 'top_k': 20, 'top_p': 0.8, 'repetition_penalty': 1.1},
        'portfolio': {'temperature': 0.7, 'top_k': 30, 'top_p': 0.9, 'repetition_penalty': 1.2},
        'general': {'temperature': 0.85, 'top_k': 45, 'top_p': 0.95, 'repetition_penalty': 1.1}
    }
    return params.get(mode, params['general'])

def generate_text(model, tokenizer, device, prompt, mode=None, max_tokens=150, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2):
    if mode:
        mode_params = get_mode_params(mode)
        temperature = temperature if temperature != 0.8 else mode_params['temperature']
        top_k = top_k if top_k != 40 else mode_params['top_k']
        top_p = top_p if top_p != 0.9 else mode_params['top_p']
        repetition_penalty = repetition_penalty if repetition_penalty != 1.2 else mode_params['repetition_penalty']
    
    if mode:
        prompt = f"[MODE:{mode.upper()}] {prompt}"
    
    encoded = encode(prompt, tokenizer)
    if not encoded:
        encoded = [tokenizer.special_tokens.get('[UNK]', 1)]
    
    context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
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
    
    response = response.replace('<eos>', '').replace('<eos>', '')
    response = ' '.join(response.split())
    
    return response

def interactive_mode():
    print("MedBrief AI - Interactive Generation Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 50)
    
    model, tokenizer, device = load_model_and_tokenizer()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'help':
            print("\nCommands:")
            print("  quit - Exit the program")
            print("  help - Show this help message")
            print("  mode:<mode> - Set generation mode (psych, health, crisis, portfolio, general)")
            print("  temp:<value> - Set temperature (0.1-2.0)")
            print("  tokens:<value> - Set max tokens (10-500)")
            print("Example: mode:psych I'm feeling anxious")
            continue
        
        mode = None
        prompt = user_input
        
        if user_input.startswith('mode:'):
            parts = user_input.split(' ', 1)
            if len(parts) > 1:
                mode = parts[0][5:].lower()
                prompt = parts[1] if len(parts) > 1 else ""
        
        if not prompt:
            print("Please provide a prompt.")
            continue
        
        if not mode:
            mode = detect_domain(prompt)
        
        print(f"\nMode: {mode}")
        print("Generating...")
        
        response = generate_text(model, tokenizer, device, prompt, mode=mode)
        
        print(f"\nMedBrief: {response}")

def batch_generation(prompts_file="prompts.txt", output_file="outputs.txt"):
    print(f"Loading model...")
    model, tokenizer, device = load_model_and_tokenizer()
    
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Prompts file '{prompts_file}' not found.")
        return
    
    outputs = []
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        mode = detect_domain(prompt)
        response = generate_text(model, tokenizer, device, prompt, mode=mode)
        
        outputs.append(f"Prompt: {prompt}\nMode: {mode}\nResponse: {response}\n{'-'*50}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(outputs))
    
    print(f"Generated responses saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate text using MedBrief AI model')
    parser.add_argument('--prompt', type=str, help='Single prompt to generate from')
    parser.add_argument('--mode', type=str, choices=['psych', 'health', 'crisis', 'portfolio', 'general'], help='Generation mode')
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Generation temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--batch', action='store_true', help='Run batch generation from prompts.txt')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='vocab.json', help='Path to vocabulary file')
    parser.add_argument('--merges', type=str, default='merges.pkl', help='Path to BPE merges file')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.batch:
        batch_generation()
    elif args.prompt:
        model, tokenizer, device = load_model_and_tokenizer(args.model, args.vocab, args.merges)
        response = generate_text(
            model, tokenizer, device, args.prompt, 
            mode=args.mode, max_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, repetition_penalty=args.repetition_penalty
        )
        print(f"Prompt: {args.prompt}")
        if args.mode:
            print(f"Mode: {args.mode}")
        print(f"Response: {response}")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
