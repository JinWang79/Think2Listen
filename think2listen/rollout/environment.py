"""Audio-model environment for targeted evidence requests."""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache

from openai import AsyncOpenAI

from think2listen.prompts import AUDIO_OBSERVATION_SYSTEM_PROMPT
from think2listen.rollout.cache import AudioObservationCache


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


@lru_cache(maxsize=1)
def _cache() -> AudioObservationCache:
    return AudioObservationCache(os.environ.get("AUDIO_CACHE_PATH", "artifacts/cache/audio_observations.sqlite"))


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.environ.get("AUDIO_MODEL_BASE_URL", "http://127.0.0.1:8901/v1"),
        api_key=os.environ.get("AUDIO_MODEL_API_KEY", "EMPTY"),
        timeout=float(os.environ.get("AUDIO_MODEL_TIMEOUT_SECONDS", "30")),
        max_retries=0,
    )


async def get_observation(*, sample_id: str, audio_path: str, query: str) -> str:
    """Return one actor-visible observation, using a persistent local cache."""

    model = _required("AUDIO_MODEL_NAME_OR_PATH")
    prompt_version = os.environ.get("AUDIO_PROMPT_VERSION", "targeted-audio-v1")
    cached = _cache().get(
        audio_id=sample_id,
        query=query,
        omni_model=model,
        prompt_version=prompt_version,
    )
    if cached is not None:
        return cached

    attempts = int(os.environ.get("AUDIO_MODEL_MAX_ATTEMPTS", "2"))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await _client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": AUDIO_OBSERVATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}},
                            {"type": "text", "text": query},
                        ],
                    },
                ],
                temperature=float(os.environ.get("AUDIO_MODEL_TEMPERATURE", "0.3")),
                top_p=float(os.environ.get("AUDIO_MODEL_TOP_P", "0.95")),
                max_tokens=int(os.environ.get("AUDIO_MODEL_MAX_TOKENS", "256")),
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            observation = (response.choices[0].message.content or "").strip()
            if not observation:
                raise RuntimeError("Audio model returned an empty observation")
            _cache().put_if_absent(
                audio_id=sample_id,
                query=query,
                omni_model=model,
                prompt_version=prompt_version,
                observation=observation,
            )
            return observation
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                await asyncio.sleep(0.5)
    raise RuntimeError(
        f"Audio request failed after {attempts} attempts for sample_id={sample_id!r}: {last_error!r}"
    ) from last_error
