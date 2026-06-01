# MedBrief AI

MedBrief AI now follows the guide-aligned FastAPI gateway in `backend/app/` together with the `recall-app/` frontend. The older root-level custom-model and Flask files are still in the repository for experimentation, but the production-facing runtime is the backend gateway plus the frontend served against port `8001`.

## Runtime Paths

- `backend/app/`: authoritative FastAPI gateway, safety layer, prompt assembly, persistent profile memory, local learning export, API keys, and inference engine selection
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
- `GET /v1/training/export`
- `POST /api/keys`
- `GET /api/keys`
- `DELETE /api/keys/{key_id}`

Streaming uses SSE and is enabled by default in the frontend.

## Deployment Direction

- Frontend: Vercel static hosting
- Global app/API entrypoint: Vercel
- Self-run model backend: FastAPI gateway on Modal, Railway, Render, or a GPU server
- Real-model path: the local MedBrief transformer checkpoint in `model.pth`
- Optional self-hosted deployment path: host an open-weight model behind vLLM or Ollama
- Fallback path: disabled by default; MedBrief should not fake personalized intelligence with scripted responses

## Universal Deployment

For a worldwide product, do not point Vercel at `localhost` or a laptop Ollama process. Deploy one self-run MedBrief backend with a public HTTPS URL, then set this on Vercel:

```bash
MEDBRIEF_REMOTE_BACKEND_URL=https://your-self-hosted-medbrief-backend.example
```

With that variable set, Vercel becomes the global public edge for the website and OpenAI-compatible API, while all model, memory, profile, feedback, API-key, and training-export calls are proxied to the self-hosted backend. This keeps the product usable from anywhere without requiring users to bring OpenAI keys.

Suggested local environment variables for the custom local model:

```bash
MEDBRIEF_RUNTIME_API_BASE=http://127.0.0.1:8001
MEDBRIEF_INFERENCE_ENGINE=custom
MEDBRIEF_CUSTOM_MODEL_PATH=model.pth
MEDBRIEF_CUSTOM_VOCAB_PATH=vocab.json
MEDBRIEF_CUSTOM_MERGES_PATH=merges.pkl
MEDBRIEF_ALLOW_LOCAL_RESPONDER_FALLBACK=false
MEDBRIEF_STRICT_MODEL_BACKEND=true
MEDBRIEF_LEARNING_CAPTURE_ENABLED=true
```

Developer API keys can be generated from the Settings panel or with:

```bash
curl -X POST http://127.0.0.1:8001/api/keys ^
  -H "Content-Type: application/json" ^
  -d "{\"label\":\"local dev\"}"
```

Then call MedBrief's OpenAI-compatible chat endpoint with `Authorization: Bearer <generated-key>`. These are MedBrief API keys for your own backend; they are not OpenAI keys.

To export locally captured prompt/response pairs for review and future fine-tuning:

```bash
curl http://127.0.0.1:8001/v1/training/export ^
  -H "X-MedBrief-Admin-Token: <admin-token>"
```

The app learns immediately through local memory/profile state and captures interactions for later local training. It does not perform unsafe live weight updates during a medical or mental-health chat.
