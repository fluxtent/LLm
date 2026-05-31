from utils import load_data, load_tokenizer, encode, decode, advanced_clean_text, detect_domain, add_domain_tag, save_tokenized_data, load_tokenized_data, create_train_val_split, deduplicate_documents, filter_quality
from model import MedBriefTransformer, block_size, embedding_dim, n_head, n_layer, dropout
import os
import torch
import math
import time
import json
from torch.cuda.amp import autocast, GradScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def load_and_preprocess_data(data_file):
    raw_text = load_data(data_file)
    text = advanced_clean_text(raw_text)
    
    documents = [doc.strip() for doc in text.split('\n\n') if doc.strip()]
    documents = deduplicate_documents(documents)
    documents = filter_quality(documents)
    
    tagged_docs = []
    for doc in documents:
        domain = detect_domain(doc)
        tagged_doc = add_domain_tag(doc, domain)
        tagged_docs.append(tagged_doc)
    
    train_docs, val_docs = create_train_val_split(tagged_docs, val_ratio=0.05)
    
    return train_docs, val_docs

def get_batch(data, batch_size=16, block_size=block_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def evaluate(model, data, batch_size=16):
    model.eval()
    total_loss = 0
    total_batches = min(100, len(data) // (batch_size * block_size))
    
    with torch.no_grad():
        for _ in range(total_batches):
            xb, yb = get_batch(data, batch_size)
            logits, loss = model(xb, yb)
            total_loss += loss.item()
    
    model.train()
    return total_loss / total_batches

def train_model():
    print("Loading and preprocessing data...")
    train_docs, val_docs = load_and_preprocess_data("backup_data.txt")
    
    print("Training BPE tokenizer...")
    train_text = '\n\n'.join(train_docs)
    from bpe import BPETokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocab_size=6000)
    tokenizer.save('vocab.json', 'merges.pkl')
    
    print("Tokenizing data...")
    all_text = '\n\n'.join(train_docs + val_docs)
    encoded_data = tokenizer.encode(all_text)
    data_tensor = torch.tensor(encoded_data, dtype=torch.long)
    
    train_size = int(len(data_tensor) * 0.95)
    train_data = data_tensor[:train_size]
    val_data = data_tensor[train_size:]
    
    save_tokenized_data(train_data.tolist(), 'train_data.bin')
    save_tokenized_data(val_data.tolist(), 'val_data.bin')
    
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    model = MedBriefTransformer(
        vocab_size=vocab_size,
        n_embd=embedding_dim,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-5)
    scaler = GradScaler()
    
    grad_accum_steps = 8
    batch_size = 16
    eval_interval = 1000
    save_interval = 10000
    max_steps = 100000
    
    step = 0
    best_val_loss = float('inf')
    
    print("Starting training...")
    start_time = time.time()
    
    while step < max_steps:
        optimizer.zero_grad()
        
        for _ in range(grad_accum_steps):
            xb, yb = get_batch(train_data, batch_size // grad_accum_steps)
            
            with autocast():
                logits, loss = model(xb, yb)
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            perplexity = math.exp(loss.item())
            print(f"Step {step} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        if step % eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_data)
            val_perplexity = math.exp(val_loss)
            print(f"Validation | Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': step,
                    'val_loss': val_loss,
                    'vocab_size': vocab_size
                }, 'best_model.pth')
                print("New best model saved!")
        
        if step % save_interval == 0 and step > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': step,
                'val_loss': val_loss if step % eval_interval == 0 else None,
                'vocab_size': vocab_size
            }, f'model_step{step}.pth')
            print(f"Checkpoint saved: model_step{step}.pth")
        
        step += 1
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'val_loss': best_val_loss,
        'vocab_size': vocab_size
    }
    torch.save(final_checkpoint, 'model.pth')
    print("Training completed. Final model saved as model.pth")

if __name__ == "__main__":
    train_model()
