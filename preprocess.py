from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from bpe import BPETokenizer
from utils import (
    add_domain_tag,
    advanced_clean_text,
    create_train_val_split,
    deduplicate_documents,
    detect_domain,
    encode,
    filter_quality,
    load_data,
    save_tokenized_data,
    split_documents,
    write_json,
)


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
DEFAULT_CORPUS_DIR = ROOT / "corpus"


def load_local_corpus(corpus_dir: Path) -> list[str]:
    documents: list[str] = []
    if not corpus_dir.exists():
        return documents
    for path in sorted(corpus_dir.rglob("*.txt")):
        raw_text = load_data(path)
        documents.extend(split_documents(advanced_clean_text(raw_text)))
    return documents


def generate_synthetic_instruction_pairs(count: int = 500) -> list[str]:
    emotions = ["anxious", "drained", "overwhelmed", "stuck", "ashamed", "restless"]
    triggers = ["before work", "during social events", "at bedtime", "after conflict", "around deadlines"]
    body_signals = ["tight chest", "racing thoughts", "nausea", "shaky hands", "brain fog"]
    health_topics = ["insomnia", "burnout", "headaches", "brain fog", "stress hormones"]
    portfolio_features = ["mode-aware routing", "memory continuity", "safety-first design", "OpenAI-compatible delivery"]
    utility_tasks = ["organize a decision", "compare tradeoffs", "turn a vague goal into steps", "summarize a complex idea"]

    pairs: list[str] = []
    for emotion in emotions:
        for trigger in triggers:
            for signal in body_signals:
                pairs.append(
                    "[QA] [USER] I've been feeling "
                    f"{emotion} {trigger}, and it usually shows up as {signal}. "
                    "[ASSISTANT] That sounds exhausting. When anxiety keeps showing up in a repeatable setting, it often helps to separate the trigger, the body response, and the story your mind starts telling about it. "
                    f"In your case, the combination of {emotion}, {trigger}, and {signal} suggests there may be a loop worth slowing down and naming. "
                    "A useful first step is to notice what happens in the minute before the feeling spikes. What part feels most intense right now?"
                )

    for topic in health_topics:
        for lens in ["timing", "severity", "associated symptoms", "daily function", "duration"]:
            pairs.append(
                "[HEALTH] [USER] Can you explain "
                f"{topic} in a careful, non-diagnostic way? "
                "[ASSISTANT] Generally speaking, that topic can have more than one explanation, so I do not want to collapse it into a diagnosis. "
                f"A careful way to think about {topic} is to pay attention to {lens}, because that often changes how concerning it is. "
                "If it is persistent, worsening, severe, or affecting daily function, a clinician is the right person to evaluate it."
            )

    for feature in portfolio_features:
        for angle in ["product trust", "user continuity", "safety boundaries", "multi-mode coherence", "deployment credibility"]:
            pairs.append(
                "[GENERAL] [USER] What makes MedBrief AI different as a product? "
                "[ASSISTANT] MedBrief AI is designed to feel like a serious product rather than a generic chatbot. "
                f"One important differentiator is {feature}, especially when you care about {angle}. "
                "That helps the system stay coherent across emotional support, healthcare education, and portfolio questions."
            )

    for task in utility_tasks:
        for framing in ["known vs unknowns", "decision criteria", "sequencing", "risk awareness", "clarity of next step"]:
            pairs.append(
                "[GENERAL] [USER] Can you help me "
                f"{task}? "
                "[ASSISTANT] Yes. Start with the part of the situation the user actually named, then turn that into one usable next move. "
                f"For this request, focus first on {framing}, because that keeps the answer tied to the user's real constraint instead of a generic framework. "
                "Then give the next concrete action and only add structure where it helps."
            )

    if len(pairs) >= count:
        return pairs[:count]

    while len(pairs) < count:
        seed = len(pairs)
        emotion = emotions[seed % len(emotions)]
        topic = health_topics[seed % len(health_topics)]
        task = utility_tasks[seed % len(utility_tasks)]
        pairs.append(
            "[GENERAL] [USER] I need both emotional clarity and practical structure. "
            f"[ASSISTANT] A useful place to begin is to name the emotional layer ({emotion}), the health uncertainty ({topic}), and the practical task ({task}) separately. "
            "That prevents one kind of overwhelm from swallowing the whole problem and usually leads to a more intelligent next step."
        )
    return pairs[:count]


def collect_documents(corpus_dir: Path, include_backup: bool = True, synthetic_pairs: int = 500) -> list[str]:
    documents: list[str] = []
    if include_backup and (ROOT / "backup_data.txt").exists():
        backup_text = load_data(ROOT / "backup_data.txt")
        documents.extend(split_documents(advanced_clean_text(backup_text)))
    documents.extend(load_local_corpus(corpus_dir))
    documents.extend(generate_synthetic_instruction_pairs(synthetic_pairs))
    return documents


def preprocess_documents(documents: list[str]) -> tuple[list[str], list[str], dict]:
    deduped = deduplicate_documents(documents, near_dup_threshold=0.85)
    filtered = filter_quality(deduped)
    tagged = [add_domain_tag(document, detect_domain(document)) for document in filtered]
    train_docs, val_docs = create_train_val_split(tagged, val_ratio=0.05, seed=17)
    manifest = {
        "raw_documents": len(documents),
        "deduplicated_documents": len(deduped),
        "filtered_documents": len(filtered),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "domain_counts": Counter(detect_domain(document) for document in filtered),
    }
    return train_docs, val_docs, manifest


def tokenize_splits(
    train_docs: list[str],
    val_docs: list[str],
    vocab_size: int,
    vocab_path: Path,
    merges_path: Path,
) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train(train_docs, vocab_size=vocab_size)
    tokenizer.save(vocab_path, merges_path)

    train_tokens: list[int] = []
    for document in train_docs:
        train_tokens.extend(encode(document, tokenizer, add_bos=True, add_eos=True))

    val_tokens: list[int] = []
    for document in val_docs:
        val_tokens.extend(encode(document, tokenizer, add_bos=True, add_eos=True))

    save_tokenized_data(train_tokens, ROOT / "train_data.bin")
    save_tokenized_data(val_tokens, ROOT / "val_data.bin")
    return tokenizer


def write_processed_text(train_docs: list[str], val_docs: list[str]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "train.txt").write_text("\n\n".join(train_docs), encoding="utf-8")
    (PROCESSED_DIR / "val.txt").write_text("\n\n".join(val_docs), encoding="utf-8")


def json_summary(manifest: dict) -> str:
    lines = [
        "Preprocessing complete:",
        f"  raw_documents={manifest['raw_documents']}",
        f"  deduplicated_documents={manifest['deduplicated_documents']}",
        f"  filtered_documents={manifest['filtered_documents']}",
        f"  train_documents={manifest['train_documents']}",
        f"  val_documents={manifest['val_documents']}",
        f"  vocab_size={manifest['vocab_size']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MedBrief training corpora and build BPE artifacts")
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--synthetic-pairs", type=int, default=500)
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--skip-backup", action="store_true")
    args = parser.parse_args()

    documents = collect_documents(
        Path(args.corpus_dir),
        include_backup=not args.skip_backup,
        synthetic_pairs=args.synthetic_pairs,
    )
    train_docs, val_docs, manifest = preprocess_documents(documents)
    write_processed_text(train_docs, val_docs)
    tokenizer = tokenize_splits(
        train_docs,
        val_docs,
        vocab_size=args.vocab_size,
        vocab_path=ROOT / "vocab.json",
        merges_path=ROOT / "merges.pkl",
    )
    manifest["vocab_size"] = len(tokenizer.vocab)
    write_json(ROOT / "preprocess_manifest.json", manifest)
    print(json_summary(manifest))


if __name__ == "__main__":
    main()
