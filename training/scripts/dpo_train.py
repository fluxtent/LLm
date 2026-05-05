"""DPO training entrypoint for MedBrief AI.

DPO only runs when the release eval report says SFT quality is below the gate
or a red-line safety failure exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from training.scripts.common import BUILT_DIR


BACKBONE = "microsoft/Phi-3-mini-4k-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO for MedBrief AI when the release gate fails")
    parser.add_argument("--preference-file", default=str(BUILT_DIR / "preference_pairs.jsonl"))
    parser.add_argument("--eval-report", default="training/data/built/release_eval_report.json")
    parser.add_argument("--output-dir", default="training/outputs/dpo")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def should_run_dpo(eval_report_path: str, force: bool) -> bool:
    if force:
        return True
    report_file = Path(eval_report_path)
    if not report_file.exists():
        return False
    report = json.loads(report_file.read_text(encoding="utf-8"))
    return report.get("average_rubric_score", 5.0) < 4.2 or report.get("red_line_failures", 0) > 0


def main() -> None:
    args = parse_args()
    if not should_run_dpo(args.eval_report, args.force):
        print("Skipping DPO because the SFT release gate passed.")
        return

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BACKBONE, trust_remote_code=True, device_map="auto")
    dataset = load_dataset("json", data_files={"train": args.preference_file})["train"]

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        logging_steps=10,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(Path(args.output_dir) / "adapter")
    tokenizer.save_pretrained(Path(args.output_dir) / "adapter")


if __name__ == "__main__":
    main()
