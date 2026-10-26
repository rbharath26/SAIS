"""Embedding encoder interfaces and implementations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TextEncoder(Protocol):
    """Produces one vector per input string."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEncoder:
    """A deterministic, dependency-free encoder for tests and examples.

    This is not a semantic embedding model and should not be used for research
    results. Install the ``embeddings`` extra for the default transformer model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


class SentenceTransformerEncoder:
    """Lazy adapter for sentence-transformers.

    The model is downloaded by sentence-transformers on its first use if it is
    not already cached locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "SentenceTransformerEncoder requires `pip install "
                    "sais-score[embeddings]`. Use HashingEncoder for an "
                    "offline demonstration."
                ) from error
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist()


class OllamaEncoder:
    """Embedding adapter for a locally running Ollama server.

    The legacy ``/api/embeddings`` endpoint is used for compatibility with
    Ollama 0.40. Each text is requested separately because that endpoint does
    not support batching. No text leaves the local Ollama server.
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:0.6b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, list[float]] = {}

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if text not in self._cache:
                self._cache[text] = self._embed(text)
            vectors.append(self._cache[text])
        return vectors

    def _embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model_name, "prompt": text}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama embedding request failed ({error.code}): {details}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`."
            ) from error
        embedding = body.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(
                f"Ollama model {self.model_name!r} returned no embedding: {body!r}"
            )
        return [float(value) for value in embedding]
