from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from model import MedBriefTransformer, ModelConfig, default_config
from utils import calculate_perplexity, encode, load_tokenized_data, load_tokenizer


ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MedBrief custom transformer")
    parser.add_argument("--stage", choices=["pretrain", "finetune", "reinforce"], default="pretrain")
    parser.add_argument("--train-file", default="train_data.bin")
    parser.add_argument("--val-file", default="val_data.bin")
    parser.add_argument("--train-text", default="")
    parser.add_argument("--val-text", default="")
    parser.add_argument("--model-out", default="model.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=150000)
    parser.add_argument("--resume", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def stage_defaults(args: argparse.Namespace) -> None:
    if args.stage == "finetune":
        args.learning_rate = 2e-5 if args.learning_rate == 3e-4 else args.learning_rate
        args.max_steps = 10000 if args.max_steps == 150000 else args.max_steps
    elif args.stage == "reinforce":
        args.learning_rate = 2e-5 if args.learning_rate == 3e-4 else args.learning_rate
        args.max_steps = 3000 if args.max_steps == 150000 else args.max_steps

    if args.smoke:
        args.max_steps = 20
        args.eval_interval = 10
        args.save_interval = 20
        args.batch_size = 1
        args.grad_accum = 1


def load_sequence(path_or_text: str, tokenizer=None) -> list[int]:
    path = Path(path_or_text)
    if path.suffix == ".bin" and path.exists():
        return load_tokenized_data(path)
    if path.exists():
        if tokenizer is None:
            raise ValueError("tokenizer required to encode text inputs")
        text = path.read_text(encoding="utf-8", errors="ignore")
        return encode(text, tokenizer, add_bos=True, add_eos=True)
    if tokenizer is not None and path_or_text:
        return encode(path_or_text, tokenizer, add_bos=True, add_eos=True)
    raise FileNotFoundError(f"could not find training source: {path_or_text}")


def get_batch(data: list[int], batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("training data is too short for the configured block size")
    indices = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i : i + block_size], dtype=torch.long) for i in indices])
    y = torch.stack([torch.tensor(data[i + 1 : i + block_size + 1], dtype=torch.long) for i in indices])
    return x.to(DEVICE), y.to(DEVICE)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step + 1, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        min_factor = min_lr / base_lr
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def evaluate(model: MedBriefTransformer, data: list[int], block_size: int, batch_size: int) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(min(24, max(1, len(data) // (block_size * max(batch_size, 1))))):
            xb, yb = get_batch(data, max(batch_size, 1), block_size)
            _, loss, _ = model(xb, yb)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def create_model(vocab_size: int, block_size: int) -> MedBriefTransformer:
    config = default_config(vocab_size=vocab_size)
    config = ModelConfig(**{**config.__dict__, "block_size": block_size})
    return MedBriefTransformer(config).to(DEVICE)


def save_checkpoint(
    path: str | Path,
    model: MedBriefTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    val_loss: float,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "config": model.to_checkpoint_config(),
        "vocab_size": model.config.vocab_size,
    }
    torch.save(checkpoint, path)


def maybe_resume(
    model: MedBriefTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    checkpoint_path: str,
) -> tuple[int, float]:
    if not checkpoint_path:
        return 0, float("inf")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("step", 0)), float(checkpoint.get("val_loss", float("inf")))


def main() -> None:
    args = parse_args()
    stage_defaults(args)
    tokenizer = load_tokenizer()

    train_source = args.train_text or args.train_file
    val_source = args.val_text or args.val_file
    train_data = load_sequence(train_source, tokenizer=tokenizer)
    val_data = load_sequence(val_source, tokenizer=tokenizer)

    model = create_model(vocab_size=len(tokenizer.vocab), block_size=args.block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        base_lr=args.learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    start_step, best_val_loss = maybe_resume(model, optimizer, scheduler, args.resume)
    start_time = time.time()

    print(f"Using device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []

        for _ in range(args.grad_accum):
            xb, yb = get_batch(train_data, args.batch_size, args.block_size)
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                _, loss, _ = model(xb, yb)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            micro_losses.append(float(loss.item() * args.grad_accum))

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss = sum(micro_losses) / max(len(micro_losses), 1)
        if step % 100 == 0 or step == args.max_steps - 1:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"step={step} train_loss={train_loss:.4f} perplexity={calculate_perplexity(train_loss):.2f} "
                f"lr={current_lr:.6f} elapsed={elapsed:.1f}s"
            )

        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.block_size, max(1, args.batch_size))
            print(f"validation_loss={val_loss:.4f} validation_perplexity={calculate_perplexity(val_loss):.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ROOT / args.model_out, model, optimizer, scheduler, step, val_loss)
                print("saved new best model")
            if val_loss > train_loss * 1.3:
                print("warning: validation loss suggests overfitting")

        if step > 0 and step % args.save_interval == 0:
            checkpoint_name = ROOT / f"model_step{step}.pth"
            save_checkpoint(checkpoint_name, model, optimizer, scheduler, step, best_val_loss)
            print(f"saved checkpoint: {checkpoint_name.name}")

    save_checkpoint(ROOT / args.model_out, model, optimizer, scheduler, args.max_steps, best_val_loss)
    print(f"training complete, saved {args.model_out}")


if __name__ == "__main__":
    main()
