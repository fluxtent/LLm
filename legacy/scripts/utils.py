import re
import json
import pickle
import hashlib
from collections import Counter
from bpe import BPETokenizer

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' <newline> ')
    text = re.sub(r'\.', ' <period> <eos> ', text)
    text = re.sub(r'\!', ' <exclamation> <eos> ', text)
    text = re.sub(r'\?', ' <question> <eos> ', text)
    text = re.sub(r',', ' <comma> ', text)
    text = re.sub(r'["“”]', ' <quote> ', text)
    text = re.sub(r'[-–—]', ' <dash> ', text)
    return text

def restore_text(tokens):
    text = ' '.join(tokens)
    text = re.sub(r'\s*<period>\s*', '.', text)
    text = re.sub(r'\s*<comma>\s*', ',', text)
    text = re.sub(r'\s*<question>\s*', '?', text)
    text = re.sub(r'\s*<exclamation>\s*', '!', text)
    text = re.sub(r'\s*<quote>\s*', '"', text)
    text = re.sub(r'\s*<dash>\s*', '-', text)
    text = re.sub(r'\s*<newline>\s*', '\n', text)
    text = re.sub(r'\s*<eos>\s*', ' <eos> ', text)
    sentences = []
    
    for segment in text.split('<eos>'):
        segment = segment.strip()
        if segment:
            segment = segment[0].upper() + segment[1:] if len(segment) > 1 else segment.upper()
            sentences.append(segment)
    return ' '.join(sentences)


def advanced_clean_text(text):
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # keep only ASCII
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s*([.?!,;:])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += '.'

    return text

def make_vocab(text):
    words = text.split()
    unique_words = sorted(set(words))
    stringtoindex = {word: i for i, word in enumerate(unique_words)}
    indextostring = {i: word for word, i in stringtoindex.items()}
    return stringtoindex, indextostring

def preview_vocab(token_map, count=10):
    print("Previewing first", count, "tokens:")
    for token, idx in list(token_map.items())[:count]:
        print(f"'{token}': {idx}")

def preprocess(text, mode="advanced"):
    if mode == "basic":
        return clean_text(text)
    return advanced_clean_text(text)

def count_char_freq(text):
    words = text.split()
    frequent = Counter(words)
    return frequent.most_common(10)

def add_domain_tag(text, domain):
    domain_tags = {
        'psych': '[PSYCH]',
        'health': '[HEALTH]',
        'narrative': '[NARRATIVE]',
        'qa': '[QA]',
        'general': '[GENERAL]'
    }
    tag = domain_tags.get(domain.lower(), '[GENERAL]')
    return f"{tag} {text}"

def detect_domain(text):
    psych_keywords = ['anxious', 'depressed', 'feeling', 'overwhelmed', 'struggling', 'therapist', 'panic', 'mental', 'emotional']
    health_keywords = ['symptom', 'medication', 'diagnosis', 'doctor', 'pain', 'condition', 'treatment', 'hospital', 'medical']
    narrative_keywords = ['story', 'character', 'felt', 'experienced', 'journey', 'tale', 'narrative', 'plot']
    qa_keywords = ['what', 'how', 'why', 'when', 'where', 'explain', 'define', 'describe']
    
    text_lower = text.lower()
    
    psych_score = sum(1 for kw in psych_keywords if kw in text_lower)
    health_score = sum(1 for kw in health_keywords if kw in text_lower)
    narrative_score = sum(1 for kw in narrative_keywords if kw in text_lower)
    qa_score = sum(1 for kw in qa_keywords if kw in text_lower)
    
    scores = {'psych': psych_score, 'health': health_score, 'narrative': narrative_score, 'qa': qa_score}
    max_domain = max(scores, key=scores.get)
    
    if scores[max_domain] == 0:
        return 'general'
    
    return max_domain

def encode(text, tokenizer):
    if hasattr(tokenizer, 'encode'):
        return tokenizer.encode(text)
    else:
        return [tokenizer[word] for word in text.split() if word in tokenizer]

def decode(indices, tokenizer):
    if hasattr(tokenizer, 'decode'):
        return tokenizer.decode(indices)
    else:
        return ' '.join([tokenizer[i] for i in indices])

def load_tokenizer(vocab_path='vocab.json', merges_path='merges.pkl'):
    tokenizer = BPETokenizer()
    tokenizer.load(vocab_path, merges_path)
    return tokenizer

def deduplicate_documents(documents, similarity_threshold=0.85):
    seen_hashes = set()
    unique_docs = []
    
    for doc in documents:
        doc_hash = hashlib.sha256(doc.encode()).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)
    
    return unique_docs

def filter_quality(documents):
    filtered = []
    for doc in documents:
        words = doc.split()
        if len(words) < 100:
            continue
        
        non_alpha = sum(1 for c in doc if not c.isalpha() and not c.isspace())
        if non_alpha / len(doc) > 0.3:
            continue
        
        sentences = re.split(r'[.!?]+', doc)
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_len < 5 or avg_sentence_len > 60:
            continue
        
        filtered.append(doc)
    
    return filtered

def create_train_val_split(documents, val_ratio=0.05):
    import random
    random.shuffle(documents)
    split_idx = int(len(documents) * (1 - val_ratio))
    return documents[:split_idx], documents[split_idx:]

def save_tokenized_data(data, filename):
    import torch
    torch.save(torch.tensor(data, dtype=torch.long), filename)

def load_tokenized_data(filename):
    import torch
    return torch.load(filename)
