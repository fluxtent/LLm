"""Validate built MedBrief training assets."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from training.scripts.common import BUILT_DIR, normalize_text, read_jsonl, record_fingerprint
from training.scripts.schemas import EvalPrompt, PreferencePair, SFTConversation


EXPECTED_MODE_COUNTS = {
    "psych": 3300,
    "health": 1500,
    "portfolio": 600,
    "general": 600,
}


def _validate_sft_file(filename: str) -> list[dict]:
    rows = read_jsonl(BUILT_DIR / filename)
    for row in rows:
        SFTConversation.model_validate(row)
    return rows


def _assert_no_split_leakage(*splits: list[dict]) -> None:
    seen: dict[str, int] = {}
    for split_index, split in enumerate(splits):
        for row in split:
            fingerprint = record_fingerprint(row["messages"])
            previous_split = seen.get(fingerprint)
            if previous_split is not None and previous_split != split_index:
                raise AssertionError("split leakage detected")
            seen[fingerprint] = split_index


def _assert_no_medication_dosing(rows: list[dict]) -> None:
    banned_patterns = ("mg", "milligrams", "you should take", "safe dose", "dosage should be")
    for row in rows:
        assistant_text = row["messages"][-1]["content"].lower()
        if row["mode"] == "health" and any(pattern in assistant_text for pattern in banned_patterns):
            if "can't help with medication dosing" not in assistant_text:
                raise AssertionError("prohibited medication advice detected")


def main() -> None:
    train_rows = _validate_sft_file("sft_train.jsonl")
    val_rows = _validate_sft_file("sft_val.jsonl")
    test_rows = _validate_sft_file("sft_test.jsonl")
    golden_rows = [EvalPrompt.model_validate(row).model_dump() for row in read_jsonl(BUILT_DIR / "golden_eval_prompts.jsonl")]
    safety_rows = [EvalPrompt.model_validate(row).model_dump() for row in read_jsonl(BUILT_DIR / "safety_eval_prompts.jsonl")]
    preference_rows = [PreferencePair.model_validate(row).model_dump() for row in read_jsonl(BUILT_DIR / "preference_pairs.jsonl")]

    _assert_no_split_leakage(train_rows, val_rows, test_rows)
    _assert_no_medication_dosing(train_rows + val_rows + test_rows)

    all_rows = train_rows + val_rows + test_rows
    mode_counts = Counter(row["mode"] for row in all_rows)
    if mode_counts != EXPECTED_MODE_COUNTS:
        raise AssertionError(f"unexpected mode mix: {mode_counts}")

    manifest = json.loads((BUILT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["minimum_gates"]["sft_conversations"] > len(all_rows):
        raise AssertionError("SFT minimum gate not met")
    if len(golden_rows) < 150:
        raise AssertionError("golden prompts gate not met")
    if len(safety_rows) < 300:
        raise AssertionError("safety prompts gate not met")
    if len(preference_rows) < 500:
        raise AssertionError("preference pair gate not met")

    normalized_messages = {normalize_text(json.dumps(row["messages"], ensure_ascii=False)) for row in all_rows}
    if len(normalized_messages) != len(all_rows):
        raise AssertionError("dedupe effectiveness check failed")

    print("Training assets validated successfully.")


if __name__ == "__main__":
    main()
