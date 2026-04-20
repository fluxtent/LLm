# Backend

Production serving is handled by a FastAPI gateway in `backend/app/` and a Modal deployment entrypoint in `backend/modal/`.

## Local development

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8001
```

By default, the gateway runs in `mock` mode unless `MEDBRIEF_VLLM_BASE_URL` or `MEDBRIEF_INFERENCE_ENGINE=vllm` is set. In mock mode, the backend now serves the richer profile-aware template engine instead of a static canned fallback.

If you set `MEDBRIEF_ENV=production`, startup now fails loudly when the gateway is still pointed at `mock` or is using the default insecure API key.

## Expected environment variables

- `MEDBRIEF_INFERENCE_ENGINE`: `mock` or `vllm`
- `MEDBRIEF_VLLM_BASE_URL`: base URL for the internal or remote vLLM server
- `MEDBRIEF_VLLM_API_KEY`: bearer token used when proxying to vLLM
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
