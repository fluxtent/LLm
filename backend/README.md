# Backend

Production serving is handled by a FastAPI gateway in `backend/app/` and a Modal deployment entrypoint in `backend/modal/`.

## Local development

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8001
```

The gateway supports `custom`, `ollama`, `vllm`, `openai`, and `mock` engines. The default is `custom`, which loads MedBrief's own checkpoint from `model.pth` with `vocab.json` and `merges.pkl`. The self-run defaults do not use external AI providers, and scripted local-response fallback is disabled unless you explicitly opt in for demos.

If you set `MEDBRIEF_ENV=production`, startup now fails loudly when the gateway is still pointed at `mock` or is using the default insecure API key.

## Expected environment variables

- `MEDBRIEF_INFERENCE_ENGINE`: `custom`, `ollama`, `vllm`, `openai`, or `mock`
- `MEDBRIEF_CUSTOM_MODEL_PATH`: path to the MedBrief checkpoint, default `model.pth`
- `MEDBRIEF_CUSTOM_VOCAB_PATH`: path to the MedBrief tokenizer vocabulary, default `vocab.json`
- `MEDBRIEF_CUSTOM_MERGES_PATH`: path to the MedBrief tokenizer merges, default `merges.pkl`
- `MEDBRIEF_CUSTOM_ALLOW_CPU`: allow local model inference on CPU when CUDA is unavailable, default `true`
- `MEDBRIEF_VLLM_BASE_URL`: base URL for the internal or remote vLLM server
- `MEDBRIEF_VLLM_API_KEY`: bearer token used when proxying to vLLM
- `MEDBRIEF_OPENAI_API_KEY` or `OPENAI_API_KEY`: optional external adapter key, only used when `MEDBRIEF_INFERENCE_ENGINE=openai`
- `MEDBRIEF_ALLOW_EXTERNAL_OPENAI`: explicitly allow the external OpenAI adapter, default `false`
- `MEDBRIEF_STORE_PATH`: persistent JSON store for profiles, summaries, feedback, and API key hashes
- `MEDBRIEF_API_KEYS_ENABLED`: enable `/api/keys` key generation, default `true`
- `MEDBRIEF_REQUIRE_API_KEY`: require generated API keys for API routes, default `false`
- `MEDBRIEF_ADMIN_TOKEN`: admin token for API-key management in production
- `MEDBRIEF_ALLOW_LOCAL_RESPONDER_FALLBACK`: allow scripted local fallback responses, default `false`
- `MEDBRIEF_STRICT_MODEL_BACKEND`: fail instead of silently substituting a mock engine, default `true`
- `MEDBRIEF_LEARNING_CAPTURE_ENABLED`: store local prompt/response pairs for later training export, default `true`
- `MEDBRIEF_ENV`: `development` or `production`
- `MEDBRIEF_MODEL_ID`: public model name exposed through the OpenAI-compatible API
- `MEDBRIEF_BASE_MODEL_ID`: backbone model ID, fixed to Phi-3 Mini for v1
- `MEDBRIEF_ADAPTER_ID`: deployed adapter or LoRA identifier
- `MEDBRIEF_BUILD_SHA`: release/build identifier

## Public routes

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`
- `GET /api/config`
- `POST /v1/profile`
- `POST /v1/memory/summarize`
- `POST /v1/session/init`
- `POST /v1/feedback`
- `GET /v1/training/export`
- `POST /api/keys`
- `GET /api/keys`
- `DELETE /api/keys/{key_id}`

`GET /v1/training/export` returns local JSONL by default and is protected by the same admin controls as API-key management. Captured interactions are training candidates for offline review, SFT, or preference-data creation; chat requests do not mutate model weights live.
