from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    ff_mult: int = 4
    dropout: float = 0.1


class TinyTransformer(nn.Module):
    """Legacy compatibility path for older checkpoints."""

    def __init__(self, vocab_size: int, embedding_dim: int = 256, block_size: int = 256):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        positions = torch.arange(t, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(positions)
        return self.lm_head(tok_emb + pos_emb)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cache(self, seq_len: int, device: torch.device, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        stacked = torch.stack((-x2, x1), dim=-1)
        return stacked.flatten(-2)

    def _apply_rope(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        cos, sin = self._rope_cache(seq_len, x.device, offset=offset)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            q = self._apply_rope(q, seq_len, offset=past_k.size(2))
            k = self._apply_rope(k, seq_len, offset=past_k.size(2))
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            past_len = past_k.size(2)
        else:
            past_len = 0
            q = self._apply_rope(q, seq_len, offset=0)
            k = self._apply_rope(k, seq_len, offset=0)

        total_len = k.size(2)
        mask = self.mask[:, :, past_len : past_len + seq_len, :total_len]
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.config.dropout if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.resid_dropout(self.proj(out))
        new_cache = (k, v) if use_cache else None
        return out, new_cache


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.n_embd * config.ff_mult
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, new_cache = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ff(self.ln_2(x))
        return x, new_cache


class MedBriefTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor] | None] | None]:
        batch, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.block_size}")

        past_len = 0
        if past_key_values and past_key_values[0] is not None:
            past_len = past_key_values[0][0].size(2)
        if past_len + seq_len > self.block_size:
            raise ValueError("past context plus current sequence exceeds block size")

        x = self.wte(idx)
        x = self.drop(x)

        new_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for block_index, block in enumerate(self.blocks):
            block_past = past_key_values[block_index] if past_key_values else None
            x, block_cache = block(x, past_kv=block_past, use_cache=use_cache)
            if use_cache:
                new_cache.append(block_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss, (new_cache if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        generated = idx
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None

        for _ in range(max_new_tokens):
            if use_kv_cache and past_key_values is not None:
                idx_cond = generated[:, -1:]
            else:
                idx_cond = generated[:, -self.block_size :]

            logits, _, new_cache = self(
                idx_cond,
                past_key_values=past_key_values,
                use_cache=use_kv_cache,
            )
            if use_kv_cache:
                past_key_values = new_cache

            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                unique_ids = torch.unique(generated)
                for token_id in unique_ids:
                    logits[:, token_id] /= repetition_penalty

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(1, sorted_indices, remove)
                logits = logits.masked_fill(mask, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
            if generated.size(1) >= self.block_size:
                past_key_values = None

        return generated

    def to_checkpoint_config(self) -> dict[str, int | float]:
        return asdict(self.config)


block_size = 256
embedding_dim = 384
n_embd = embedding_dim
n_head = 6
n_layer = 6
ff_mult = 4
dropout = 0.1


def default_config(vocab_size: int, dropout_override: float | None = None) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        ff_mult=ff_mult,
        dropout=dropout if dropout_override is None else dropout_override,
    )
