"""Inference engine adapters for the custom MedBrief model and optional provider backends."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .medical_ontology import MedicalContext, detect_medical_context
from .schemas import UserProfile
from .settings import Settings
from .template_engine import TemplateEngine, TemplateRequest


@dataclass(frozen=True)
class InferenceResult:
    text: str
    finish_reason: str = "stop"
    latency_ms: int = 0
    upstream_model: str | None = None


class BaseInferenceEngine:
    name = "base"

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        raise NotImplementedError

    async def health(self) -> bool:
        return True

    async def warmup(self) -> None:
        return None


class MockInferenceEngine(BaseInferenceEngine):
    name = "mock"

    def __init__(self) -> None:
        self._templates = TemplateEngine()

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        del max_tokens, temperature, top_p
        started = time.perf_counter()
        latest_user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        recent_assistant = tuple(
            message["content"] for message in messages if message["role"] == "assistant"
        )
        text = self._templates.render(
            TemplateRequest(
                latest_user=latest_user,
                mode=mode,
                request_id=request_id,
                conversation_id=conversation_id,
                profile=profile,
                medical_context=detect_medical_context(latest_user),
                recent_assistant_messages=recent_assistant,
            )
        )
        return InferenceResult(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            upstream_model=model,
        )


class LocalResponderEngine(BaseInferenceEngine):
    name = "local-responder"

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        del max_tokens, temperature, top_p, request_id, conversation_id
        started = time.perf_counter()
        latest_user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        previous_user = [
            message["content"] for message in messages if message["role"] == "user"
        ][:-1]
        text = self._respond(
            latest_user=latest_user,
            previous_user=previous_user[-3:],
            mode=mode,
            profile=profile,
            medical_context=detect_medical_context(latest_user),
        )
        return InferenceResult(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            upstream_model=model,
        )

    async def health(self) -> bool:
        return True

    def _respond(
        self,
        *,
        latest_user: str,
        previous_user: list[str],
        mode: str,
        profile: UserProfile | None,
        medical_context: MedicalContext,
    ) -> str:
        text = _strip_mode_tag(latest_user)
        lowered = _normalize(text)
        if lowered.strip(" .!?") in {"hi", "hey", "hello", "yo"}:
            return "Hey. I am here and ready. Send the question, feeling, symptom, or project you want to work through."
        if mode == "health" or medical_context.has_medical_signal or _looks_medical_question(lowered):
            return self._health_response(text, lowered, medical_context, profile)
        if mode == "psych" or _distress_score(lowered) > 0:
            return self._support_response(text, lowered, previous_user, profile)
        if mode == "portfolio" or "medbrief" in lowered or "api key" in lowered:
            return self._product_response(text)
        return self._general_response(text, lowered, previous_user)

    def _health_response(
        self,
        text: str,
        lowered: str,
        medical_context: MedicalContext,
        profile: UserProfile | None,
    ) -> str:
        subject = _definition_subject(text)
        known = _known_medical_definition(subject or lowered)
        if known:
            return known
        if "nephritis" in lowered:
            return (
                "Nephritis is inflammation of the kidneys. It can affect the kidney tissue or filtering units, and it may happen after an infection, from an autoimmune condition, from some medications, or with other kidney diseases. "
                "Common clues can include blood or protein in the urine, swelling, high blood pressure, fever, flank pain, or changes in urination, but some cases are found only on urine or blood tests. "
                "It is something a clinician should evaluate because the cause matters; urgent care is important if there is severe pain, very little urine, major swelling, trouble breathing, or rapidly worsening symptoms."
            )
        if subject:
            specialty = _medical_specialty_definition(subject)
            if specialty:
                return specialty
            return (
                f"{subject.capitalize()} is not a term I can define confidently from the local medical knowledge base alone. "
                "If it is from a lab report, diagnosis list, or class note, send the surrounding sentence and I can explain it in context without pretending certainty. "
                "For health terms, context matters: the body system, symptoms, timeline, and exact wording often change the meaning."
            )
        if medical_context.symptoms:
            symptom = medical_context.symptoms[0]
            return (
                f"{symptom.capitalize()} can come from several different causes, so I would not treat one message as a diagnosis. "
                "What matters is when it started, how severe it is, whether it is worsening, and what else is happening with it. "
                "If it is sudden, severe, persistent, or paired with alarming symptoms, getting medical care is the right move."
            )
        return (
            "I can answer this as general health education, not a diagnosis. "
            "The useful details are what changed, when it started, how severe it is, what else is happening, and whether it is affecting daily function. "
            "If you give me the specific term, symptom, or lab result, I can explain what it usually means and what details a clinician would care about."
        )

    def _support_response(
        self,
        text: str,
        lowered: str,
        previous_user: list[str],
        profile: UserProfile | None,
    ) -> str:
        del profile
        continuity = ""
        if previous_user and any(token in lowered for token in ("this", "that", "again", "either", "same")):
            continuity = f" I am reading this in context of what you just said: {previous_user[-1][:120].strip()}."
        safety = ""
        if any(token in lowered for token in ("giving up", "give up", "fuck life", "hate my life", "can't do this", "cant do this")):
            safety = " Are you safe right now, meaning not about to hurt yourself?"
        if "pressure" in lowered:
            return (
                f"That sounds like pressure is crowding out your ability to think, not like you are weak.{continuity} "
                f"For the next few minutes, shrink the problem: sit somewhere steady, breathe slower than feels natural, and name the single pressure point that is loudest.{safety}"
            ).strip()
        if "hate my life" in lowered or "fuck life" in lowered:
            return (
                f"{text.strip().capitalize()} sounds like pain talking at full volume, not a final verdict on your life.{continuity} "
                f"Right now the first job is safety and getting your thinking back online: feet on the floor, unclench your hands, and tell me what happened right before this spiked.{safety}"
            ).strip()
        if "dont know what to do" in lowered or "don't know what to do" in lowered:
            return (
                f"You do not need to solve your whole life from inside this state.{continuity} "
                "Give me the nearest concrete fact: what happened today, who is involved, and what has to be handled first."
            )
        return (
            f"What you wrote sounds personal, not abstract.{continuity} "
            "The useful move is to slow it down into: what happened, what it made you feel, and what your mind concluded from it. "
            "Start with the first part only."
        )

    def _product_response(self, text: str) -> str:
        lowered = _normalize(text)
        if "api" in lowered or "key" in lowered:
            return (
                "MedBrief API keys are generated by this app and are used to call its own OpenAI-compatible `/v1/chat/completions` endpoint. "
                "They are not OpenAI keys; they authenticate requests into the MedBrief gateway, which then routes to the configured MedBrief model backend and applies memory, safety, and response-quality checks."
            )
        return (
            "MedBrief AI is meant to be its own assistant surface: chat UI, memory, generated API keys, safety handling, and a model gateway behind one product. "
            "The important architecture point is that the website should call MedBrief's backend, not require users to bring their own key."
        )

    def _general_response(self, text: str, lowered: str, previous_user: list[str]) -> str:
        subject = _definition_subject(text)
        if subject:
            known = _known_general_definition(subject)
            if known:
                return known
            if subject in {"anything", "everything"}:
                return (
                    f"{subject.capitalize()} is a broad word, so the direct answer depends on the frame. "
                    "In plain terms, it means the set of possible things being talked about; philosophically, it points at existence as a whole rather than one object. "
                    "If you mean purpose or meaning, that is a different question: purpose is the role something serves or the reason it matters."
                )
            return (
                f"I do not want to fake a definition for {subject}. "
                "Give me the context where you saw it, or ask for a specific angle, and I can answer that directly instead of swapping the word into a canned sentence."
            )
        if previous_user and any(token in lowered for token in ("this", "that", "it", "again")):
            return (
                f"You are referring back to the last point: {previous_user[-1][:140].strip()}. "
                "The direct next step is to name which part you want resolved: the meaning, the decision, the cause, or the action."
            )
        return (
            "I can help with that. Give me the exact thing you want answered, built, compared, or explained, and I will respond directly to that instead of wrapping it in a generic framework."
        )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _strip_mode_tag(text: str) -> str:
    return re.sub(r"^\s*\[mode:[a-z]+\]\s*", "", text, flags=re.I).strip()


def _definition_subject(text: str) -> str:
    cleaned = _strip_mode_tag(text).strip()
    match = re.match(r"^(what is|what's|define|explain)\s+(.+?)[?.!]*$", cleaned, flags=re.I)
    if not match:
        return ""
    subject = re.sub(r"^(a|an|the)\s+", "", match.group(2).strip(), flags=re.I)
    return subject[:80].strip()


def _known_medical_definition(subject: str) -> str:
    normalized = _normalize(subject).strip(" ?.!").replace("-", " ")
    definitions = {
        "nephrology": (
            "Nephrology is the branch of medicine that focuses on the kidneys. "
            "Nephrologists deal with kidney function, chronic kidney disease, kidney inflammation, electrolyte problems, blood-pressure problems related to the kidneys, dialysis, and transplant-related kidney care. "
            "So the simple version is: nephrology is kidney medicine."
        ),
        "nephritis": (
            "Nephritis is inflammation of the kidneys. It can affect the kidney tissue or filtering units, and it may happen after an infection, from an autoimmune condition, from some medications, or with other kidney diseases. "
            "Common clues can include blood or protein in the urine, swelling, high blood pressure, fever, flank pain, or changes in urination, but some cases are found only on urine or blood tests. "
            "It is something a clinician should evaluate because the cause matters."
        ),
        "urology": (
            "Urology is the medical specialty focused on the urinary tract and the male reproductive system. "
            "Urologists handle problems involving the kidneys' drainage system, ureters, bladder, urethra, prostate, kidney stones, urinary obstruction, and many surgical urinary conditions. "
            "The quick distinction: nephrology is kidney function; urology is the urinary tract and surgical plumbing side."
        ),
        "cardiology": (
            "Cardiology is the branch of medicine focused on the heart and blood vessels. "
            "Cardiologists evaluate things like chest pain, heart rhythm problems, heart failure, valve disease, high blood pressure complications, and coronary artery disease."
        ),
        "neurology": (
            "Neurology is the branch of medicine focused on the brain, spinal cord, nerves, and muscles. "
            "Neurologists evaluate problems like seizures, migraines, stroke symptoms, weakness, numbness, tremor, multiple sclerosis, and nerve pain."
        ),
        "oncology": (
            "Oncology is the branch of medicine focused on cancer. "
            "Oncologists diagnose, stage, and treat cancers using tools such as chemotherapy, immunotherapy, targeted therapy, radiation coordination, and long-term surveillance."
        ),
        "endocrinology": (
            "Endocrinology is the branch of medicine focused on hormones and hormone-producing glands. "
            "It covers conditions like diabetes, thyroid disease, adrenal problems, pituitary disorders, and some reproductive hormone issues."
        ),
        "dermatology": (
            "Dermatology is the branch of medicine focused on skin, hair, and nails. "
            "Dermatologists handle rashes, acne, eczema, psoriasis, skin infections, hair loss, and skin cancer screening."
        ),
        "gastroenterology": (
            "Gastroenterology is the branch of medicine focused on the digestive system. "
            "It covers the esophagus, stomach, intestines, liver, gallbladder, bile ducts, and pancreas."
        ),
        "rheumatology": (
            "Rheumatology is the branch of medicine focused on autoimmune and inflammatory diseases that often affect joints, muscles, blood vessels, and connective tissue. "
            "Rheumatologists treat conditions like rheumatoid arthritis, lupus, vasculitis, and inflammatory joint disease."
        ),
    }
    return definitions.get(normalized, "")


def _medical_specialty_definition(subject: str) -> str:
    normalized = _normalize(subject).strip(" ?.!").replace("-", " ")
    roots = {
        "nephr": ("kidneys", "kidney medicine"),
        "cardi": ("the heart", "heart medicine"),
        "neur": ("the nervous system", "brain and nerve medicine"),
        "onc": ("cancer", "cancer medicine"),
        "derm": ("skin", "skin medicine"),
        "endocrin": ("hormones and glands", "hormone medicine"),
        "gastro": ("the digestive system", "digestive-system medicine"),
        "rheumat": ("autoimmune and inflammatory joint/connective-tissue disease", "rheumatic disease medicine"),
        "hemat": ("blood", "blood medicine"),
        "pulmon": ("the lungs", "lung medicine"),
    }
    if normalized.endswith("ology"):
        for root, (body_system, plain_label) in roots.items():
            if normalized.startswith(root):
                return (
                    f"{subject.capitalize()} is the medical specialty that focuses on {body_system}. "
                    f"In plain English, it means {plain_label}. "
                    "The specialist in that field evaluates related diseases, test results, symptoms, and long-term treatment plans."
                )
        return (
            f"{subject.capitalize()} is likely the study or medical specialty of a specific body system or disease area, because “-ology” means “the study of.” "
            "I do not have enough local knowledge to name the exact field confidently, so send the context where you saw it and I can narrow it down."
        )
    if normalized.endswith("itis"):
        base = normalized[:-4]
        return (
            f"{subject.capitalize()} usually means inflammation of something; the suffix “-itis” means inflammation. "
            f"The exact meaning depends on what “{base}” refers to, so the surrounding medical context matters."
        )
    return ""


def _known_general_definition(subject: str) -> str:
    normalized = _normalize(subject).strip(" ?.!").replace("-", " ")
    if normalized in {"purpose of anything", "purpose of everything"}:
        return (
            "The purpose of anything depends on what kind of thing it is. "
            "A heart exists to move blood, a bridge exists to carry people across a gap, and a promise exists to hold trust between people. "
            "If you mean life or existence as a whole, there may not be one assigned purpose waiting to be discovered; purpose is often made through what reduces suffering, creates connection, builds something real, or protects what matters."
        )
    if normalized.startswith("purpose of "):
        target = normalized.removeprefix("purpose of ").strip()
        if target:
            return (
                f"The purpose of {target} is the role it serves, not just what it is called. "
                "A useful answer asks what it changes, who it helps or affects, and what would be missing if it did not exist."
            )
    definitions = {
        "purpose": (
            "Purpose is the reason something matters or the role it serves. "
            "For a tool, purpose is what it is used for; for a choice, it is what the choice moves you toward; for a person, purpose is usually built through what they care about and keep choosing."
        ),
        "meaning": (
            "Meaning is the significance something has to a person, situation, or life. "
            "It is not only a dictionary definition; it is why something feels worth attention, effort, grief, love, or responsibility."
        ),
    }
    return definitions.get(normalized, "")


def _looks_medical_question(lowered: str) -> bool:
    return any(
        token in lowered
        for token in (
            "itis",
            "ology",
            "kidney",
            "disease",
            "symptom",
            "diagnosis",
            "treatment",
            "blood test",
            "urine",
            "infection",
            "inflammation",
        )
    )


def _distress_score(lowered: str) -> int:
    return sum(
        1
        for token in (
            "fuck life",
            "hate my life",
            "giving up",
            "give up",
            "pressure",
            "overwhelmed",
            "can't think",
            "cant think",
            "dont know what to do",
            "don't know what to do",
            "alone",
            "failure",
        )
        if token in lowered
    )


class CustomMedBriefEngine(BaseInferenceEngine):
    name = "custom"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._runtime: dict[str, Any] | None = None
        self._runtime_error: Exception | None = None
        self._lock = threading.Lock()

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_error is not None:
            raise RuntimeError("custom MedBrief runtime failed to load") from self._runtime_error

        with self._lock:
            if self._runtime is not None:
                return self._runtime
            try:
                from generate import load_runtime

                runtime = load_runtime(
                    model_path=self._settings.custom_model_path,
                    vocab_path=self._settings.custom_vocab_path,
                    merges_path=self._settings.custom_merges_path,
                )
                if self._settings.custom_allow_cpu and runtime.get("release_ready") and runtime.get("model_loaded"):
                    runtime["serve_model"] = True
                    runtime["serve_strategy"] = "model"
                self._runtime = runtime
                return runtime
            except Exception as exc:
                self._runtime_error = exc
                raise

    def _complete_sync(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None,
        profile: UserProfile | None,
    ) -> InferenceResult:
        del request_id, conversation_id, profile
        from generate import MODE_PARAMETERS, generate_response

        runtime = self._load_runtime()
        params = {**MODE_PARAMETERS["general"], **MODE_PARAMETERS.get(mode, {})}
        started = time.perf_counter()
        with self._lock:
            text = generate_response(
                runtime,
                messages,
                mode=mode,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=params["top_k"],
                top_p=top_p,
                repetition_penalty=params["repetition_penalty"],
                allow_heuristic=False,
            )
        return InferenceResult(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            upstream_model=self._settings.public_model_id,
        )

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        del model
        return await asyncio.to_thread(
            self._complete_sync,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            request_id=request_id,
            mode=mode,
            conversation_id=conversation_id,
            profile=profile,
        )

    async def health(self) -> bool:
        try:
            runtime = await asyncio.to_thread(self._load_runtime)
            return bool(
                runtime.get("model_loaded")
                and runtime.get("tokenizer") is not None
                and runtime.get("release_ready")
            )
        except Exception:
            return False

    async def warmup(self) -> None:
        try:
            await asyncio.to_thread(self._load_runtime)
        except Exception:
            return None


class OllamaChatEngine(BaseInferenceEngine):
    name = "ollama"

    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._default_model = settings.ollama_model
        self._public_model_id = settings.public_model_id
        self._timeout = settings.ollama_timeout_seconds
        self._keep_alive = settings.ollama_keep_alive
        self._num_ctx = settings.ollama_num_ctx
        self._num_thread = settings.ollama_num_thread

    def _resolve_model(self, requested_model: str | None) -> str:
        if not requested_model or requested_model == self._public_model_id:
            return self._default_model
        return requested_model

    def _payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, object]:
        return {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": self._num_ctx,
                "num_thread": self._num_thread,
            },
        }

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        del request_id, mode, conversation_id, profile
        started = time.perf_counter()
        requested_model = self._resolve_model(model)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=self._payload(
                    model=requested_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
            )
            response.raise_for_status()
            data = response.json()
        content = data["message"]["content"]
        return InferenceResult(
            text=content,
            finish_reason=data.get("done_reason", "stop"),
            latency_ms=int((time.perf_counter() - started) * 1000),
            upstream_model=data.get("model", requested_model),
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
            models = {model.get("name") for model in data.get("models", []) if isinstance(model, dict)}
            return self._default_model in models
        except Exception:
            return False

    async def warmup(self) -> None:
        try:
            await self.complete(
                messages=[
                    {
                        "role": "system",
                        "content": "Reply with ok.",
                    },
                    {
                        "role": "user",
                        "content": "ok",
                    },
                ],
                model=self._default_model,
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                request_id="warmup",
                mode="general",
            )
        except Exception:
            return None


class OpenAIChatEngine(BaseInferenceEngine):
    name = "openai"

    def __init__(self, settings: Settings):
        self._base_url = settings.openai_base_url.rstrip("/")
        self._api_key = settings.openai_api_key
        self._default_model = settings.openai_model
        self._public_model_id = settings.public_model_id
        self._timeout = settings.request_timeout_seconds

    def _resolve_model(self, requested_model: str | None) -> str:
        if not requested_model or requested_model == self._public_model_id:
            return self._default_model
        return requested_model

    def _headers(self, request_id: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "X-Request-ID": request_id,
        }

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        del mode, conversation_id, profile
        started = time.perf_counter()
        requested_model = self._resolve_model(model)
        payload = {
            "model": requested_model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(request_id),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        return InferenceResult(
            text=content,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            latency_ms=int((time.perf_counter() - started) * 1000),
            upstream_model=data.get("model", requested_model),
        )

    async def health(self) -> bool:
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self._base_url}/v1/models",
                    headers=self._headers("health-check"),
                )
            return response.is_success
        except httpx.HTTPError:
            return False


class VLLMChatEngine(BaseInferenceEngine):
    name = "vllm"

    def __init__(self, settings: Settings):
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._is_vercel_ai_gateway = "ai-gateway.vercel.sh" in self._base_url
        self._api_key = self._resolve_api_key(settings)
        self._timeout = settings.request_timeout_seconds
        self._signing_secret = settings.vllm_signing_secret
        self._default_model = settings.vllm_model or settings.public_model_id
        self._public_model_id = settings.public_model_id
        self._health_checked_at = 0.0
        self._health_ok = False

    def _resolve_api_key(self, settings: Settings) -> str:
        if self._is_vercel_ai_gateway:
            return settings.ai_gateway_api_key or settings.vercel_oidc_token or settings.vllm_api_key
        return settings.vllm_api_key

    def _resolve_model(self, requested_model: str | None) -> str:
        if not requested_model or requested_model == self._public_model_id:
            return self._default_model
        return requested_model

    def _openai_path(self, suffix: str) -> str:
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/{suffix.lstrip('/')}"
        return f"{self._base_url}/v1/{suffix.lstrip('/')}"

    def _headers(self, request_id: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if self._signing_secret:
            signature = hmac.new(
                self._signing_secret.encode("utf-8"),
                request_id.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-MedBrief-Signature"] = signature
        return headers

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        started = time.perf_counter()
        requested_model = self._resolve_model(model)
        payload = {
            "model": requested_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if not self._is_vercel_ai_gateway:
            payload["extra_body"] = {
                "mode": mode,
                "request_id": request_id,
                "conversation_id": conversation_id,
                "preferences": profile.preferences.model_dump() if profile else None,
            }

        delay = 0.5
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self._openai_path("chat/completions"),
                        headers=self._headers(request_id),
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                content = data["choices"][0]["message"]["content"]
                return InferenceResult(
                    text=content,
                    finish_reason=data["choices"][0].get("finish_reason", "stop"),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    upstream_model=data.get("model", requested_model),
                )
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network path
                detail = re.sub(r"\s+", " ", exc.response.text).strip()[:500]
                status = exc.response.status_code
                last_error = RuntimeError(f"vLLM completion returned HTTP {status}: {detail}")
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"vLLM completion failed after retries: {last_error}") from last_error

    async def health(self) -> bool:
        now = time.time()
        if now - self._health_checked_at < 60:
            return self._health_ok
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                response = await client.post(
                    self._openai_path("chat/completions"),
                    headers=self._headers("health-check"),
                    json={
                        "model": self._default_model,
                        "messages": [{"role": "user", "content": "ok"}],
                        "max_tokens": 1,
                        "temperature": 0,
                        "stream": False,
                    },
                )
            self._health_ok = response.is_success
        except httpx.HTTPError:
            self._health_ok = False
        self._health_checked_at = now
        return self._health_ok


def create_inference_engine(settings: Settings) -> BaseInferenceEngine:
    if settings.active_engine == "custom":
        return CustomMedBriefEngine(settings)
    if settings.active_engine == "openai":
        return OpenAIChatEngine(settings)
    if settings.active_engine == "ollama":
        return OllamaChatEngine(settings)
    if settings.active_engine == "vllm":
        return VLLMChatEngine(settings)
    return MockInferenceEngine()
