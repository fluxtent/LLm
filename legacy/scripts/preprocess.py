import os
import re
import json
import hashlib
from collections import Counter
from utils import load_data, advanced_clean_text, detect_domain, add_domain_tag, deduplicate_documents, filter_quality, create_train_val_split
from bpe import BPETokenizer

def load_corpus_files(corpus_dir="corpus"):
    documents = []
    if os.path.exists(corpus_dir):
        for filename in os.listdir(corpus_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(corpus_dir, filename)
                print(f"Loading {filename}...")
                text = load_data(filepath)
                documents.extend([doc.strip() for doc in text.split('\n\n') if doc.strip()])
    return documents

def generate_synthetic_pairs(output_file="synthetic_pairs.txt", num_pairs=1000):
    pairs = []
    
    psych_templates = [
        ("I've been feeling {emotion} lately", "That sounds really difficult. {emotion} can be overwhelming. What specific situations seem to trigger these feelings?"),
        ("I think I might have {condition}", "I understand why you're concerned about {condition}. It's important to talk to a healthcare professional who can give you proper guidance."),
        ("How do I cope with {challenge}?", "Coping with {challenge} is different for everyone. Some people find it helpful to try {strategy}, while others prefer {alternative}.")
    ]
    
    health_templates = [
        ("What are the symptoms of {condition}?", "{condition} typically involves symptoms like {symptom1}, {symptom2}, and {symptom3}. However, everyone's experience can be different."),
        ("Is {treatment} effective for {condition}?", "{treatment} can be effective for some people with {condition}, but it's best to discuss this with your doctor to see if it's right for you."),
        ("Should I see a doctor about {symptom}?", "If you're experiencing {symptom}, it's always a good idea to consult with a healthcare professional for proper evaluation.")
    ]
    
    general_templates = [
        ("Tell me about {topic}", "{topic} is an interesting subject. Here's what you should know about it..."),
        ("How does {concept} work?", "{concept} works by {mechanism}. It's quite fascinating when you break it down."),
        ("What do you think about {idea}?", "{idea} raises some important considerations. On one hand, {pro}, but on the other hand, {con}.")
    ]
    
    emotions = ["anxious", "depressed", "overwhelmed", "stressed", "confused"]
    conditions = ["anxiety", "depression", "insomnia", "panic attacks"]
    challenges = ["stress", "anxiety", "negative thoughts", "sleep problems"]
    strategies = ["deep breathing", "meditation", "exercise", "journaling"]
    alternatives = ["talking to friends", "professional help", "mindfulness practices"]
    
    symptoms = ["headaches", "fatigue", "chest pain", "dizziness"]
    treatments = ["medication", "therapy", "lifestyle changes", "exercise"]
    
    topics = ["artificial intelligence", "mental health", "nutrition", "exercise"]
    concepts = ["neural networks", "cognitive behavioral therapy", "metabolism", "muscle growth"]
    ideas = ["universal basic income", "remote work", "online education", "social media"]
    
    import random
    
    for i in range(num_pairs):
        if i < num_pairs // 3:
            template = random.choice(psych_templates)
            user = template[0].format(
                emotion=random.choice(emotions),
                condition=random.choice(conditions),
                challenge=random.choice(challenges)
            )
            assistant = template[1].format(
                emotion=random.choice(emotions),
                condition=random.choice(conditions),
                challenge=random.choice(challenges),
                strategy=random.choice(strategies),
                alternative=random.choice(alternatives)
            )
            domain = "psych"
        elif i < 2 * num_pairs // 3:
            template = random.choice(health_templates)
            user = template[0].format(
                condition=random.choice(conditions),
                treatment=random.choice(treatments),
                symptom=random.choice(symptoms)
            )
            assistant = template[1].format(
                condition=random.choice(conditions),
                treatment=random.choice(treatments),
                symptom=random.choice(symptoms),
                symptom1=random.choice(symptoms),
                symptom2=random.choice(symptoms),
                symptom3=random.choice(symptoms)
            )
            domain = "health"
        else:
            template = random.choice(general_templates)
            user = template[0].format(
                topic=random.choice(topics),
                concept=random.choice(concepts),
                idea=random.choice(ideas)
            )
            assistant = template[1].format(
                topic=random.choice(topics),
                concept=random.choice(concepts),
                idea=random.choice(ideas),
                pro="it offers flexibility and convenience",
                con="it may lead to isolation"
            )
            domain = "general"
        
        pair = f"[USER] {user} [ASSISTANT] {assistant}"
        tagged_pair = add_domain_tag(pair, domain)
        pairs.append(tagged_pair)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(pairs))
    
    print(f"Generated {len(pairs)} synthetic pairs in {output_file}")

def preprocess_corpus(input_files=None, output_dir="processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    all_documents = []
    
    if input_files is None:
        input_files = ["backup_data.txt"]
    
    for file_path in input_files:
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            raw_text = load_data(file_path)
            clean_text = advanced_clean_text(raw_text)
            documents = [doc.strip() for doc in clean_text.split('\n\n') if doc.strip()]
            all_documents.extend(documents)
    
    corpus_docs = load_corpus_files()
    all_documents.extend(corpus_docs)
    
    print(f"Total documents before filtering: {len(all_documents)}")
    
    all_documents = deduplicate_documents(all_documents)
    print(f"After deduplication: {len(all_documents)}")
    
    all_documents = filter_quality(all_documents)
    print(f"After quality filtering: {len(all_documents)}")
    
    tagged_documents = []
    for doc in all_documents:
        domain = detect_domain(doc)
        tagged_doc = add_domain_tag(doc, domain)
        tagged_documents.append(tagged_doc)
    
    train_docs, val_docs = create_train_val_split(tagged_documents, val_ratio=0.05)
    
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_docs))
    
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(val_docs))
    
    print(f"Saved {len(train_docs)} training documents and {len(val_docs)} validation documents")
    
    return train_docs, val_docs

def train_tokenizer_from_corpus(train_file="processed/train.txt", vocab_size=6000):
    print("Training BPE tokenizer from corpus...")
    text = load_data(train_file)
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=vocab_size)
    tokenizer.save('vocab.json', 'merges.pkl')
    print(f"Tokenizer trained with vocab size: {len(tokenizer.vocab)}")
    return tokenizer

def tokenize_and_save_data(tokenizer, train_file="processed/train.txt", val_file="processed/val.txt"):
    import torch
    
    print("Tokenizing training data...")
    train_text = load_data(train_file)
    train_tokens = tokenizer.encode(train_text)
    save_tokenized_data(train_tokens, 'train_data.bin')
    
    print("Tokenizing validation data...")
    val_text = load_data(val_file)
    val_tokens = tokenizer.encode(val_text)
    save_tokenized_data(val_tokens, 'val_data.bin')
    
    print(f"Training data: {len(train_tokens)} tokens")
    print(f"Validation data: {len(val_tokens)} tokens")

def save_tokenized_data(data, filename):
    import torch
    torch.save(torch.tensor(data, dtype=torch.long), filename)

def main():
    print("Starting corpus preprocessing pipeline...")
    
    generate_synthetic_pairs(num_pairs=500)
    
    preprocess_corpus()
    
    tokenizer = train_tokenizer_from_corpus()
    
    tokenize_and_save_data(tokenizer)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
