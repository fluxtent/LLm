from __future__ import annotations

import json
import pickle
import re
import unicodedata
from collections import Counter, OrderedDict
from pathlib import Path


DEFAULT_SPECIAL_TOKENS = OrderedDict(
    [
        ("[PAD]", 0),
        ("[UNK]", 1),
        ("[BOS]", 2),
        ("[EOS]", 3),
        ("[SYSTEM]", 4),
        ("[USER]", 5),
        ("[ASSISTANT]", 6),
        ("[MODE:PSYCH]", 7),
        ("[MODE:HEALTH]", 8),
        ("[MODE:CRISIS]", 9),
        ("[MODE:PORTFOLIO]", 10),
        ("[MODE:GENERAL]", 11),
        ("[PSYCH]", 12),
        ("[HEALTH]", 13),
        ("[NARRATIVE]", 14),
        ("[QA]", 15),
        ("[GENERAL]", 16),
    ]
)

WORD_END = "</w>"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class BPETokenizer:
    def __init__(self, special_tokens: OrderedDict[str, int] | None = None):
        self.special_tokens = OrderedDict(special_tokens or DEFAULT_SPECIAL_TOKENS)
        self.vocab: dict[str, int] = dict(self.special_tokens)
        self.inverse_vocab: dict[int, str] = {value: key for key, value in self.vocab.items()}
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self.pattern = self._compile_special_pattern()

    def _compile_special_pattern(self) -> re.Pattern[str]:
        escaped = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
        return re.compile(f"({'|'.join(escaped)})") if escaped else re.compile("$^")

    def _split_preserving_special_tokens(self, text: str) -> list[str]:
        parts: list[str] = []
        for segment in re.split(self.pattern, text):
            if segment:
                parts.append(segment)
        return parts

    def _word_to_symbols(self, word: str) -> tuple[str, ...]:
        if not word:
            return tuple()
        chars = list(word)
        chars[-1] = f"{chars[-1]}{WORD_END}"
        return tuple(chars)

    def _symbol_pairs(self, word: tuple[str, ...]) -> set[tuple[str, str]]:
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _get_pair_stats(self, vocab_words: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
        stats: Counter[tuple[str, str]] = Counter()
        for word, freq in vocab_words.items():
            for pair in self._symbol_pairs(word):
                stats[pair] += freq
        return stats

    def _merge_pair(
        self,
        pair: tuple[str, str],
        vocab_words: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, ...], int]:
        merged: dict[tuple[str, ...], int] = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        pattern = re.compile(rf"(?<!\S){re.escape(bigram)}(?!\S)")

        for word, freq in vocab_words.items():
            rendered = " ".join(word)
            updated = tuple(pattern.sub(replacement, rendered).split())
            merged[updated] = merged.get(updated, 0) + freq
        return merged

    def _build_word_frequencies(self, texts: list[str]) -> dict[tuple[str, ...], int]:
        frequencies: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            normalized = normalize_text(text)
            for segment in self._split_preserving_special_tokens(normalized):
                if segment in self.special_tokens:
                    continue
                for word in segment.split():
                    frequencies[self._word_to_symbols(word)] += 1
        return dict(frequencies)

    def train(self, texts: str | list[str], vocab_size: int = 6000) -> None:
        corpus = [texts] if isinstance(texts, str) else list(texts)
        vocab_words = self._build_word_frequencies(corpus)
        if not vocab_words:
            raise ValueError("cannot train tokenizer on empty corpus")

        base_symbols: set[str] = set()
        for word in vocab_words:
            base_symbols.update(word)

        self.vocab = dict(self.special_tokens)
        next_id = max(self.vocab.values()) + 1
        for symbol in sorted(base_symbols):
            if symbol not in self.vocab:
                self.vocab[symbol] = next_id
                next_id += 1

        self.merges = []
        while len(self.vocab) < vocab_size:
            stats = self._get_pair_stats(vocab_words)
            if not stats:
                break
            best_pair, count = stats.most_common(1)[0]
            if count < 2:
                break
            merged_symbol = "".join(best_pair)
            if merged_symbol in self.vocab:
                break
            vocab_words = self._merge_pair(best_pair, vocab_words)
            self.merges.append(best_pair)
            self.vocab[merged_symbol] = next_id
            next_id += 1

        self.merge_ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}

    def encode_word(self, word: str) -> list[int]:
        if not word:
            return []
        symbols = list(self._word_to_symbols(word))
        for pair in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    symbols[i : i + 2] = ["".join(pair)]
                else:
                    i += 1

        token_ids: list[int] = []
        for symbol in symbols:
            if symbol in self.vocab:
                token_ids.append(self.vocab[symbol])
                continue
            remainder = symbol.replace(WORD_END, "")
            for char in remainder:
                token_ids.append(self.vocab.get(char, self.special_tokens["[UNK]"]))
            if symbol.endswith(WORD_END):
                token_ids.append(self.vocab.get(WORD_END, self.special_tokens["[UNK]"]))
        return token_ids or [self.special_tokens["[UNK]"]]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        normalized = normalize_text(text)
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.special_tokens["[BOS]"])
        for segment in self._split_preserving_special_tokens(normalized):
            if segment in self.special_tokens:
                token_ids.append(self.special_tokens[segment])
                continue
            for word in segment.split():
                token_ids.extend(self.encode_word(word))
        if add_eos:
            token_ids.append(self.special_tokens["[EOS]"])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        words: list[str] = []
        current = ""
        specials_to_skip = {"[PAD]", "[BOS]"}

        for token_id in token_ids:
            symbol = self.inverse_vocab.get(int(token_id), "[UNK]")
            if symbol in specials_to_skip:
                continue
            if symbol in self.special_tokens:
                if current:
                    words.append(current)
                    current = ""
                if symbol == "[EOS]":
                    break
                words.append(symbol)
                continue
            if symbol == WORD_END:
                if current:
                    words.append(current)
                    current = ""
                continue
            if symbol.endswith(WORD_END):
                current += symbol[: -len(WORD_END)]
                words.append(current)
                current = ""
            else:
                current += symbol

        if current:
            words.append(current)

        text = " ".join(words)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\s+'", "'", text)
        text = re.sub(r"'\s+", "'", text)
        return text.strip()

    def save(self, vocab_path: str | Path = "vocab.json", merges_path: str | Path = "merges.pkl") -> None:
        Path(vocab_path).write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")
        with Path(merges_path).open("wb") as handle:
            pickle.dump(self.merges, handle)

    def load(self, vocab_path: str | Path = "vocab.json", merges_path: str | Path = "merges.pkl") -> "BPETokenizer":
        self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        with Path(merges_path).open("rb") as handle:
            self.merges = pickle.load(handle)
        self.merge_ranks = {tuple(pair): index for index, pair in enumerate(self.merges)}
        self.inverse_vocab = {int(value): key for key, value in self.vocab.items()}
        self.special_tokens = OrderedDict(
            (token, self.vocab[token]) for token in DEFAULT_SPECIAL_TOKENS if token in self.vocab
        )
        self.pattern = self._compile_special_pattern()
        return self


def train_bpe_from_file(
    input_file: str | Path,
    vocab_size: int = 6000,
    vocab_out: str | Path = "vocab.json",
    merges_out: str | Path = "merges.pkl",
) -> BPETokenizer:
    text = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=vocab_size)
    tokenizer.save(vocab_out, merges_out)
    return tokenizer


if __name__ == "__main__":
    tokenizer = train_bpe_from_file("backup_data.txt", vocab_size=6000)
    sample = "[SYSTEM] You are MedBrief AI. [USER] I'm feeling anxious about work."
    encoded = tokenizer.encode(sample, add_bos=True, add_eos=True)
    print(f"Encoded {len(encoded)} tokens")
    print(tokenizer.decode(encoded))
