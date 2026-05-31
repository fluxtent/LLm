import json
import re
from collections import defaultdict
import pickle

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[SYSTEM]': 4,
            '[USER]': 5,
            '[ASSISTANT]': 6,
            '[MODE:PSYCH]': 7,
            '[MODE:HEALTH]': 8,
            '[MODE:CRISIS]': 9,
            '[MODE:PORTFOLIO]': 10,
            '[MODE:GENERAL]': 11,
            '[PSYCH]': 12,
            '[HEALTH]': 13,
            '[NARRATIVE]': 14,
            '[QA]': 15,
            '[GENERAL]': 16
        }
        self.inverse_vocab = {v: k for k, v in self.special_tokens.items()}
        
    def train(self, text, vocab_size=8000, special_tokens=None):
        if special_tokens:
            self.special_tokens.update(special_tokens)
            self.inverse_vocab = {v: k for k, v in self.special_tokens.items()}
        
        text = self._preprocess_text(text)
        
        chars = list(text)
        vocab = set(chars)
        
        for token in self.special_tokens.keys():
            vocab.add(token)
        
        vocab = sorted(list(vocab))
        self.vocab = {token: i for i, token in enumerate(vocab)}
        
        merges = []
        current_vocab = vocab.copy()
        
        while len(current_vocab) < vocab_size:
            pairs = self._get_pairs(text)
            if not pairs:
                break
            
            pair = max(pairs, key=pairs.get)
            if pairs[pair] < 2:
                break
            
            new_token = ''.join(pair)
            if new_token in current_vocab:
                break
                
            merges.append(pair)
            current_vocab.append(new_token)
            
            text = text.replace(''.join(pair), ' '.join(pair))
        
        self.merges = merges
        self.vocab = {token: i for i, token in enumerate(current_vocab)}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
        
    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:\-\'"()[]{}]', ' ', text)
        return text.strip()
    
    def _get_pairs(self, text):
        pairs = defaultdict(int)
        tokens = text.split()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs
    
    def encode(self, text):
        text = self._preprocess_text(text)
        
        for special_token in self.special_tokens.keys():
            text = text.replace(special_token.lower(), f' {special_token} ')
        
        tokens = text.split()
        
        i = 0
        while i < len(tokens):
            if tokens[i] in self.special_tokens:
                i += 1
                continue
                
            j = 0
            while j < len(self.merges):
                pair = self.merges[j]
                if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i:i+2] = [''.join(pair)]
                    j = 0
                else:
                    j += 1
            i += 1
        
        encoded = []
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            else:
                encoded.append(self.special_tokens['[UNK]'])
        
        return encoded
    
    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        text_tokens = []
        for token_id in tokens:
            if token_id in self.inverse_vocab:
                text_tokens.append(self.inverse_vocab[token_id])
            else:
                text_tokens.append('[UNK]')
        
        text = ' '.join(text_tokens)
        
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def save(self, vocab_path, merges_path):
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
        
        with open(merges_path, 'wb') as f:
            pickle.dump(self.merges, f)
    
    def load(self, vocab_path, merges_path):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        with open(merges_path, 'rb') as f:
            self.merges = pickle.load(f)
        
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}

def train_bpe_from_file(input_file, vocab_size=8000, vocab_out='vocab.json', merges_out='merges.pkl'):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size)
    tokenizer.save(vocab_out, merges_out)
    
    print(f"Trained BPE tokenizer with vocab size: {len(tokenizer.vocab)}")
    print(f"Saved vocab to: {vocab_out}")
    print(f"Saved merges to: {merges_out}")
    
    return tokenizer

if __name__ == "__main__":
    import torch
    tokenizer = train_bpe_from_file("backup_data.txt", vocab_size=6000)
    
    test_text = "[SYSTEM] You are MedBrief AI. [USER] I'm feeling anxious."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
