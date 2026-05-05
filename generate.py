from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from model import MedBriefTransformer, ModelConfig
from utils import (
    CRISIS_KEYWORDS,
    add_mode_tag,
    clean_response,
    decode,
    detect_mode,
    encode,
    is_low_quality_response,
    load_tokenizer,
    looks_like_medication_dosing,
)


MODE_PARAMETERS = {
    "psych": {"temperature": 0.75, "top_k": 35, "top_p": 0.9, "repetition_penalty": 1.15},
    "health": {"temperature": 0.6, "top_k": 25, "top_p": 0.9, "repetition_penalty": 1.2},
    "crisis": {"temperature": 0.5, "top_k": 20, "top_p": 0.8, "repetition_penalty": 1.1},
    "general": {"temperature": 0.85, "top_k": 45, "top_p": 0.95, "repetition_penalty": 1.1},
    "portfolio": {"temperature": 0.7, "top_k": 30, "top_p": 0.9, "repetition_penalty": 1.2},
}

SYSTEM_PROMPT = (
    "You are MedBrief AI, a calm, intelligent, emotionally perceptive assistant built for emotional support, "
    "healthcare education, portfolio explanation, and high-quality general reasoning. "
    "In psychology mode, reflect before advising, notice patterns gently, and ask one useful follow-up. "
    "In health mode, never diagnose or prescribe, separate what is known from what is uncertain, and encourage clinicians for evaluation. "
    "In portfolio mode, sound polished, credible, and specific about the product. "
    "In general mode, answer clearly, thoughtfully, and without filler. "
    "In crisis mode, stop normal conversation and prioritize immediate safety resources. "
    "Never fake certainty, never give medication dosing, and never continue casually if someone may be in danger."
)


def load_runtime(
    model_path: str = "model.pth",
    vocab_path: str = "vocab.json",
    merges_path: str = "merges.pkl",
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime: dict[str, Any] = {
        "device": device,
        "model": None,
        "tokenizer": None,
        "model_loaded": False,
        "serve_model": False,
        "serve_strategy": "fallback",
        "fallback_available": True,
        "revision": "fallback",
    }

    vocab_file = Path(vocab_path)
    merges_file = Path(merges_path)
    model_file = Path(model_path)
    if not (vocab_file.exists() and merges_file.exists()):
        return runtime

    tokenizer = load_tokenizer(vocab_file, merges_file)
    runtime["tokenizer"] = tokenizer
    if not model_file.exists():
        return runtime

    checkpoint = torch.load(model_file, map_location=device)
    config_dict = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if config_dict is None:
        raise ValueError("model checkpoint missing config; retrain with the new stack")

    config = ModelConfig(**config_dict)
    model = MedBriefTransformer(config).to(device)
    checkpoint_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model.load_state_dict(compatible_state, strict=False)
    model.eval()
    if device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass
    runtime["model"] = model
    runtime["model_loaded"] = True
    val_loss = float(checkpoint.get("val_loss", float("inf"))) if isinstance(checkpoint, dict) else float("inf")
    step = int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0
    runtime["release_ready"] = val_loss <= 2.5 and step >= 1000
    runtime["validation_loss"] = val_loss
    runtime["training_step"] = step
    runtime["serve_model"] = device == "cuda" and runtime["release_ready"]
    runtime["serve_strategy"] = "model" if runtime["serve_model"] else "fallback"
    runtime["revision"] = str(model_file.stat().st_mtime)
    return runtime


def build_prompt(
    messages: list[dict[str, str]],
    mode: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    memory_summary: str | None = None,
) -> tuple[str, str]:
    system_parts = [system_prompt]
    system_parts.extend(message["content"] for message in messages if message["role"] == "system")

    conversational = [message for message in messages if message["role"] in {"user", "assistant"}]
    conversational = conversational[-6:]
    latest_user = next((message["content"] for message in reversed(conversational) if message["role"] == "user"), "")

    if mode and latest_user and not latest_user.startswith("[MODE:"):
        for message in reversed(conversational):
            if message["role"] == "user":
                message["content"] = add_mode_tag(message["content"], mode)
                latest_user = message["content"]
                break

    parts = [f"[SYSTEM] {' '.join(system_parts)}"]
    if memory_summary:
        parts.append(f"[SYSTEM] Memory summary: {memory_summary.strip()}")
    for message in conversational:
        parts.append(f"[{message['role'].upper()}] {message['content']}")
    parts.append("[ASSISTANT]")
    return "\n".join(parts), latest_user


def heuristic_response(latest_user: str, mode: str) -> str:
    lowered = latest_user.lower()

    def reflective_prefix() -> str:
        if "i feel" in lowered or "i'm" in lowered or "im " in lowered:
            return "What you're describing sounds personally heavy, not abstract. "
        if "why" in lowered:
            return "The useful way to approach that is to separate the mechanism from the meaning. "
        return ""

    def psych_response() -> str:
        if any(
            phrase in lowered
            for phrase in (
                "failure",
                "failed",
                "it's all over",
                "its all over",
                "can't think",
                "cant think",
                "dont know what to do",
                "don't know what to do",
            )
        ):
            return (
                "I’m going to take this seriously without treating the panic as a final verdict. "
                "What you wrote sounds like an overload state: your brain is trying to summarize a whole year as one irreversible failure, which makes it almost impossible to think clearly. "
                "The first move is not to solve your life tonight; it is to lower the intensity enough that your thinking comes back online. "
                "For the next ten minutes, do only three things: sit somewhere safe, drink water or breathe slowly, and write one sentence that starts with “the immediate problem is...” "
                "If “it’s all over” means you might hurt yourself or you cannot stay safe, call or text 988 now or tell someone nearby directly. "
                "If you are physically safe, tell me the one concrete thing that happened today that made the whole year feel like failure."
            )
        if any(word in lowered for word in ("anxious", "anxiety", "panic", "worried", "worry")):
            return (
                f"{reflective_prefix()}Anxiety tends to make the future feel both urgent and slippery at the same time. "
                "A useful first step is to slow the cycle down into trigger, body response, and interpretation. "
                "What seems to hit first for you: the thought, the body sensation, or the urge to escape it?"
            )
        if any(word in lowered for word in ("burnout", "exhausted", "drained", "numb")):
            return (
                f"{reflective_prefix()}That sounds closer to depletion than ordinary stress. "
                "Burnout usually narrows life into obligation, then makes even recovery feel like work. "
                "What is taking the most energy right now: pressure, uncertainty, people, or the feeling that you never really get to stop?"
            )
        if any(word in lowered for word in ("stuck", "cycle", "pattern", "same thing", "loop")):
            return (
                f"{reflective_prefix()}This sounds less like one isolated moment and more like a repeating loop. "
                "Those loops usually have a trigger, an emotional shift, a coping move, and a consequence that resets the cycle. "
                "If we map yours clearly, the point of leverage usually becomes much easier to see."
            )
        if any(word in lowered for word in ("lonely", "alone", "isolated")):
            return (
                f"{reflective_prefix()}Loneliness is heavy partly because it affects both emotion and interpretation. "
                "After a while it can make neutral situations feel like evidence that connection is out of reach. "
                "Does this feel more like lack of people around you, or lack of feeling understood even when people are present?"
            )
        return (
            f"{reflective_prefix()}That sounds emotionally significant, and it makes sense that it would keep pulling at your attention. "
            "When something keeps returning, it usually helps to name the pattern instead of treating every moment as separate. "
            "What feels most mentally loud right now: the feeling itself, what it means, or what you think it says about you?"
        )

    def health_response() -> str:
        if any(word in lowered for word in ("headache", "migraine", "head pain")):
            return (
                "Generally speaking, headaches can be associated with several things, including stress, dehydration, poor sleep, illness, eye strain, or tension. "
                "What matters most is severity, frequency, suddenness, and what shows up with them. "
                "If a headache is severe, new, persistent, worsening, or paired with neurological symptoms, a clinician should evaluate it."
            )
        if any(word in lowered for word in ("sleep", "insomnia", "can't sleep", "cant sleep")):
            return (
                "Generally speaking, sleep problems can come from stress, irregular schedules, stimulants, mood changes, pain, or medical factors. "
                "The careful way to think about it is pattern first: trouble falling asleep, staying asleep, waking too early, or feeling unrefreshed despite enough time in bed. "
                "If it keeps happening or is affecting function, a clinician is the right next step."
            )
        if any(word in lowered for word in ("stress", "cortisol", "nervous system", "brain fog")):
            return (
                "One careful way to frame that is that chronic stress can influence sleep, attention, muscle tension, digestion, and how activated the nervous system feels. "
                "That does not mean every symptom automatically comes from stress, but stress can absolutely shape how the body feels. "
                "The safest next step is to look at duration, severity, and what other symptoms are happening alongside it."
            )
        return (
            "Generally speaking, that can have more than one explanation, and I do not want to collapse it into a diagnosis. "
            "The safest way to approach it is to look at timing, severity, what else is happening with it, and whether it is affecting daily function. "
            "If it is severe, persistent, worsening, or just feels medically off, a clinician should evaluate it."
        )

    def portfolio_response() -> str:
        if "different" in lowered or "why" in lowered:
            return (
                "MedBrief AI is meant to feel like a focused product, not a generic chatbot pasted into a portfolio. "
                "Its differentiators are mode-aware responses, safety-first handling for crisis and medical boundaries, continuity through lightweight memory, and a custom-model path designed around tone and behavioral coherence. "
                "The goal is to make the interaction feel intentional, credible, and product-grade."
            )
        return (
            "MedBrief AI sits at the intersection of healthcare, AI product design, and emotionally intelligent interaction. "
            "The core idea is to deliver one polished conversational layer that can shift cleanly between emotional support, careful health education, portfolio explanation, and general utility without losing tone or safety."
        )

    def philosophical_response() -> str:
        if "meaning of life" in lowered or "purpose of life" in lowered:
            return (
                "A grounded answer is that meaning usually is not something we discover once and keep forever. "
                "It is something we build through commitment: who we love, what we take responsibility for, what we create, what we repair, and what we refuse to betray. "
                "If you want, I can answer that philosophically, psychologically, or spiritually."
            )
        if lowered.strip() in {"death", "life and death", "life or death"} or "death" in lowered:
            return (
                "Death matters because it puts a boundary around life. "
                "It is frightening because it involves loss, uncertainty, and the end of time with the people and possibilities we care about, but that same finitude is part of what gives love, choice, and time their weight. "
                "If you're asking from a personal place rather than a philosophical one, say that directly and I'll respond with that seriousness."
            )
        if "worth more" in lowered:
            return (
                "Usually the deeper question is not whether one abstract concept is worth more, but what gives value in the first place. "
                "Life carries the possibility of love, repair, meaning, and change; death carries finality and perspective. "
                "So philosophically, death gives life urgency, but life is where value can still be lived."
            )
        return (
            "The cleanest way to approach that is to separate what is factual, what is interpretive, and what kind of answer would actually help right now. "
            "Once those are separated, the question usually becomes much more tractable."
        )

    if any(keyword in lowered for keyword in CRISIS_KEYWORDS) or mode == "crisis":
        return (
            "I'm really glad you said something. Your safety matters most right now. "
            "Please call or text 988 right now if you're in the US, or contact local emergency services if you might act on these thoughts. "
            "If you can, reach out to someone nearby and tell them you need help staying safe."
        )
    if looks_like_medication_dosing(lowered):
        return (
            "I can't tell you how much medication to take. That decision needs a licensed clinician, pharmacist, or poison control professional "
            "who can consider your exact medication, history, and safety risks."
        )
    if mode == "psych":
        return psych_response()
    if mode == "health":
        return health_response()
    if mode == "portfolio":
        return portfolio_response()
    if any(token in lowered for token in ("meaning of life", "purpose", "death", "mortality", "life or death", "worth more")):
        return philosophical_response()
    if any(token in lowered for token in ("plan", "organize", "decision", "compare", "tradeoff")):
        return (
            "A useful way to handle that is to split it into three parts: what is known, what is uncertain, and what would count as a good outcome. "
            "Most hard decisions become easier once those pieces are visible. "
            "If you want, I can turn your situation into a short framework and walk through it with you."
        )
    if len(lowered.split()) <= 3:
        return philosophical_response()
    return philosophical_response()


def generate_response(
    runtime: dict[str, Any],
    messages: list[dict[str, str]],
    mode: str | None = None,
    memory_summary: str | None = None,
    max_tokens: int = 180,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    allow_heuristic: bool = False,
) -> str:
    active_mode = mode or detect_mode(next((m["content"] for m in reversed(messages) if m["role"] == "user"), ""))
    params = {**MODE_PARAMETERS["general"], **MODE_PARAMETERS.get(active_mode, {})}
    if temperature is not None:
        params["temperature"] = temperature
    if top_k is not None:
        params["top_k"] = top_k
    if top_p is not None:
        params["top_p"] = top_p
    if repetition_penalty is not None:
        params["repetition_penalty"] = repetition_penalty

    prompt, latest_user = build_prompt(messages, mode=active_mode, memory_summary=memory_summary)
    if not runtime["serve_model"] or runtime["tokenizer"] is None:
        return heuristic_response(latest_user, active_mode) if allow_heuristic else ""

    tokenizer = runtime["tokenizer"]
    model: MedBriefTransformer = runtime["model"]
    prompt_ids = encode(prompt, tokenizer, add_bos=True)
    context = torch.tensor(prompt_ids, dtype=torch.long, device=runtime["device"]).unsqueeze(0)
    eos_id = tokenizer.special_tokens.get("[EOS]")

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            eos_token_id=eos_id,
            use_kv_cache=True,
        )

    response_ids = output[0][context.shape[1] :].tolist()
    response = clean_response(decode(response_ids, tokenizer))
    if not response or is_low_quality_response(response):
        return heuristic_response(latest_user, active_mode) if allow_heuristic else ""
    return response


def interactive_mode() -> None:
    runtime = load_runtime()
    print("MedBrief AI interactive mode. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break
        mode = detect_mode(user_input)
        messages = [{"role": "user", "content": user_input}]
        print(f"\nMedBrief ({mode}): {generate_response(runtime, messages, mode=mode)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the MedBrief custom stack")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--mode", choices=list(MODE_PARAMETERS), default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--vocab", default="vocab.json")
    parser.add_argument("--merges", default="merges.pkl")
    args = parser.parse_args()

    if args.interactive or not args.prompt:
        interactive_mode()
        return

    runtime = load_runtime(args.model, args.vocab, args.merges)
    mode = args.mode or detect_mode(args.prompt)
    messages = [{"role": "user", "content": args.prompt}]
    print(generate_response(runtime, messages, mode=mode, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
