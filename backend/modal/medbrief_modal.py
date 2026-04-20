"""Modal deployment entrypoint for the MedBrief gateway.

This module launches a local vLLM process inside a Modal GPU container and
serves the FastAPI gateway as the public HTTP interface.
"""

from __future__ import annotations

import os
import subprocess
import time

import modal

from backend.app.main import create_app


APP_NAME = os.getenv("MEDBRIEF_MODAL_APP_NAME", "medbrief-ai")
BASE_MODEL_ID = os.getenv("MEDBRIEF_BASE_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
PUBLIC_MODEL_ID = os.getenv("MEDBRIEF_MODEL_ID", "medbrief-phi3-mini")
GPU_TYPE = os.getenv("MEDBRIEF_GPU_TYPE", "L4")
VLLM_API_KEY = os.getenv("MEDBRIEF_VLLM_API_KEY", "medbrief-local")
VLLM_PORT = int(os.getenv("MEDBRIEF_VLLM_PORT", "8000"))
LORA_ALIAS = os.getenv("MEDBRIEF_ADAPTER_ALIAS", "medbrief")
LORA_PATH = os.getenv("MEDBRIEF_ADAPTER_PATH", "")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("backend/requirements.txt")
    .pip_install("vllm>=0.6.0", "modal>=0.73.0")
)

app = modal.App(APP_NAME, image=image)


@app.cls(gpu=GPU_TYPE, timeout=60 * 60)
class MedBriefGateway:
    @modal.enter()
    def start_vllm(self) -> None:
        command = [
            "vllm",
            "serve",
            BASE_MODEL_ID,
            "--host",
            "127.0.0.1",
            "--port",
            str(VLLM_PORT),
            "--api-key",
            VLLM_API_KEY,
            "--served-model-name",
            PUBLIC_MODEL_ID,
        ]
        if LORA_PATH:
            command.extend(["--enable-lora", "--lora-modules", f"{LORA_ALIAS}={LORA_PATH}"])

        self._process = subprocess.Popen(command)
        os.environ["MEDBRIEF_INFERENCE_ENGINE"] = "vllm"
        os.environ["MEDBRIEF_VLLM_BASE_URL"] = f"http://127.0.0.1:{VLLM_PORT}"
        os.environ["MEDBRIEF_VLLM_API_KEY"] = VLLM_API_KEY
        os.environ["MEDBRIEF_MODEL_ID"] = PUBLIC_MODEL_ID

        time.sleep(12)

    @modal.exit()
    def stop_vllm(self) -> None:
        if getattr(self, "_process", None):
            self._process.terminate()

    @modal.asgi_app()
    def fastapi_app(self):
        return create_app()
