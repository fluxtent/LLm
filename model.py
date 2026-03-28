import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 128
embedding_dim = 256
vocab_size = None 

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

