# Training

The production training path uses `microsoft/Phi-3-mini-4k-instruct` as the fixed backbone and prepares MedBrief-specific conversational behavior with QLoRA/LoRA.

## Key scripts

- `training/scripts/build_dataset.py`: builds the SFT dataset, golden prompts, safety prompts, preference pairs, and manifest.
- `training/scripts/validate_dataset.py`: validates schema, composition, split leakage, and safety rules.
- `training/scripts/sft_train.py`: runs SFT or smoke-finetune with TRL + PEFT.
- `training/scripts/dpo_train.py`: only runs if the eval report says DPO is needed.
- `training/scripts/evaluate_release.py`: scores an OpenAI-compatible endpoint against the generated prompt sets.

## Typical flow

```bash
python training/scripts/build_dataset.py
python training/scripts/validate_dataset.py
python training/scripts/sft_train.py --smoke
python training/scripts/evaluate_release.py --api-base-url http://127.0.0.1:8001
```

The backend can also capture local chat interactions as reviewable training candidates:

```bash
curl http://127.0.0.1:8001/v1/training/export ^
  -H "X-MedBrief-Admin-Token: <admin-token>" > training/data/local_interactions.jsonl
```

Review and filter exported examples before mixing them into supervised fine-tuning or preference training. The chat service records candidates; it does not update model weights live during medical or mental-health conversations.
