"""QLoRA SFT training entrypoint for MedBrief AI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from training.scripts.common import BUILT_DIR


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]
BACKBONE = "microsoft/Phi-3-mini-4k-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MedBrief AI with QLoRA SFT")
    parser.add_argument("--train-file", default=str(BUILT_DIR / "sft_train.jsonl"))
    parser.add_argument("--eval-file", default=str(BUILT_DIR / "sft_val.jsonl"))
    parser.add_argument("--output-dir", default="training/outputs/sft")
    parser.add_argument("--smoke", action="store_true", help="Run a 100-sample smoke finetune")
    return parser.parse_args()


def _format_example(example: dict, tokenizer) -> dict:
    rendered = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": rendered}


def main() -> None:
    args = parse_args()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )

    dataset = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})
    if args.smoke:
        dataset["train"] = dataset["train"].select(range(min(100, len(dataset["train"]))))
        dataset["eval"] = dataset["eval"].select(range(min(20, len(dataset["eval"]))))

    dataset = dataset.map(lambda example: _format_example(example, tokenizer))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=2048,
        num_train_epochs=2,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25 if args.smoke else 100,
        save_steps=25 if args.smoke else 100,
        save_total_limit=2,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
        dataset_text_field="text",
    )
    trainer.train()
    trainer.model.save_pretrained(Path(args.output_dir) / "adapter")
    tokenizer.save_pretrained(Path(args.output_dir) / "adapter")


if __name__ == "__main__":
    main()
