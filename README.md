# MedBrief AI

MedBrief AI now follows the guide-aligned FastAPI gateway in `backend/app/` together with the `recall-app/` frontend. The older root-level custom-model and Flask files are still in the repository for experimentation, but the production-facing runtime is the backend gateway plus the frontend served against port `8001`.

## Runtime Paths

- `backend/app/`: authoritative FastAPI gateway, safety layer, prompt assembly, persistent profile memory, API keys, and inference engine selection
- `recall-app/`: public frontend with runtime config, memory continuity, and crisis support UI
- Root training files (`bpe.py`, `preprocess.py`, `model.py`, `train.py`, `generate.py`, `eval.py`): the custom MedBrief model, tokenizer, training, evaluation, and local inference stack
- `legacy/`: archived older checkpoints and scripts

## Local Setup

Install dependencies:

```bash
py -m pip install -r backend/requirements.txt
```

Start the FastAPI backend:

```bash
py -m uvicorn backend.app.main:app --reload --port 8001
```

The frontend can be opened in either of two ways:

```bash
http://127.0.0.1:8001/
```

or:

```bash
py -m http.server 3004 --directory recall-app
```

Then open [http://127.0.0.1:3004/index.html](http://127.0.0.1:3004/index.html). The frontend runtime config already targets `http://127.0.0.1:8001`.

## API Surface

- `GET /health`
- `GET /api/config`
- `GET /runtime-config.json`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/profile`
- `POST /v1/memory/summarize`
- `POST /v1/session/init`
- `POST /v1/feedback`
- `POST /api/keys`
- `GET /api/keys`
- `DELETE /api/keys/{key_id}`

Streaming uses SSE and is enabled by default in the frontend.

## Deployment Direction

- Frontend: Vercel static hosting
- Backend: FastAPI gateway on Railway, Render, or Modal
- Real-model path: the local MedBrief transformer checkpoint in `model.pth`
- Optional deployment path: host the MedBrief model behind vLLM/Ollama if you package it that way
- Fallback path: profile-aware template engine for demo mode and runtime failure resilience

Suggested local environment variables for the custom local model:

```bash
MEDBRIEF_RUNTIME_API_BASE=http://127.0.0.1:8001
MEDBRIEF_INFERENCE_ENGINE=custom
MEDBRIEF_CUSTOM_MODEL_PATH=model.pth
MEDBRIEF_CUSTOM_VOCAB_PATH=vocab.json
MEDBRIEF_CUSTOM_MERGES_PATH=merges.pkl
```

Developer API keys can be generated from the Settings panel or with:

```bash
curl -X POST http://127.0.0.1:8001/api/keys ^
  -H "Content-Type: application/json" ^
  -d "{\"label\":\"local dev\"}"
```

Then call MedBrief's OpenAI-compatible chat endpoint with `Authorization: Bearer <generated-key>`. These are MedBrief API keys for your own backend; they are not OpenAI keys.
