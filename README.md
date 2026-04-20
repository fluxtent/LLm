# MedBrief AI

MedBrief AI now follows the guide-aligned FastAPI gateway in `backend/app/` together with the `recall-app/` frontend. The older root-level custom-model and Flask files are still in the repository for experimentation, but the production-facing runtime is the backend gateway plus the frontend served against port `8001`.

## Runtime Paths

- `backend/app/`: authoritative FastAPI gateway, safety layer, prompt assembly, profile memory, and inference engine selection
- `recall-app/`: public frontend with runtime config, memory continuity, and crisis support UI
- Root training files (`bpe.py`, `preprocess.py`, `model.py`, `train.py`, `generate.py`, `eval.py`): custom-model experimentation and legacy local inference work
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

Streaming uses SSE and is enabled by default in the frontend.

## Deployment Direction

- Frontend: Vercel static hosting
- Backend: FastAPI gateway on Railway, Render, or Modal
- Real-model path: vLLM-backed Phi-3 inference when configured
- Fallback path: profile-aware template engine for demo mode and upstream failure resilience

Suggested local environment variables:

```bash
MEDBRIEF_RUNTIME_API_BASE=http://127.0.0.1:8001
MEDBRIEF_VLLM_BASE_URL=http://127.0.0.1:8000
MEDBRIEF_VLLM_API_KEY=medbrief-local
```
