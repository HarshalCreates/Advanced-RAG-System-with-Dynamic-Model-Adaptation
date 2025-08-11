from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import os


class GenerationClient:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    async def astream(self, system: str, user: str):  # yields tokens
        yield self.complete(system, user)


class OpenAIGeneration(GenerationClient):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return res.choices[0].message.content or ""


class AnthropicGeneration(GenerationClient):
    def __init__(self, model: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=1024,
            messages=[{"role": "user", "content": user}],
        )
        return "".join([b.text for b in msg.content if getattr(b, "text", None)])


class OllamaGeneration(GenerationClient):
    def __init__(self, model: str) -> None:
        import httpx

        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.client = httpx.Client(base_url=base_url)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"
        r = self.client.post("/api/generate", json={"model": self.model, "prompt": prompt})
        r.raise_for_status()
        return r.json().get("response", "")


class EchoGeneration(GenerationClient):
    def complete(self, system: str, user: str) -> str:
        return f"[SYSTEM]\n{system}\n\n[USER]\n{user[:4000]}"


@dataclass
class GenerationFactory:
    backend: str
    model: str

    def build(self) -> GenerationClient:
        b = self.backend.lower()
        try:
            if b == "openai":
                return OpenAIGeneration(self.model)
            if b == "anthropic":
                return AnthropicGeneration(self.model)
            if b == "ollama":
                return OllamaGeneration(self.model)
        except Exception:
            return EchoGeneration()
        return EchoGeneration()


