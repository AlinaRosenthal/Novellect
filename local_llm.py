from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import requests


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 120) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    @property
    def tags_url(self) -> str:
        return f"{self.base_url}/api/tags"

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    def list_models(self) -> List[str]:
        try:
            response = requests.get(self.tags_url, timeout=min(self.timeout, 15))
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", []) or []
            return [model.get("name") or model.get("model") for model in models if model.get("name") or model.get("model")]
        except Exception:
            return []

    def ping(self) -> Tuple[bool, str]:
        models = self.list_models()
        if models:
            return True, f"Ollama доступен. Найдено моделей: {len(models)}"
        return False, "Ollama недоступен или не найдено локальных моделей."

    def has_model(self, model_name: str) -> bool:
        models = self.list_models()
        if not models:
            return False
        normalized = model_name.strip().lower()
        return any(model.lower() == normalized for model in models)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        num_predict: int = 400,
    ) -> Dict[str, str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system
        response = requests.post(self.generate_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return {
            "text": data.get("response", "").strip(),
            "done_reason": str(data.get("done_reason", "")),
        }
