"""LLM routing via litellm Router.

How it works
------------
litellm.Router maps short **aliases** (e.g. "claude-sonnet") to the real model
string + provider credentials. Callers only ever use the alias — switching
providers is a config change, not a code change.

API keys are loaded from a .env file at repo root (never committed).
Create one with:
    OPENROUTER_API_KEY=sk-or-...
    ANTHROPIC_API_KEY=sk-ant-...
"""

import json
import os
import time

from dotenv import load_dotenv
from litellm.router import Router

load_dotenv()

# ---------------------------------------------------------------------------
# Model list
# Each entry needs:
#   "model_name"     — the alias callers use
#   "litellm_params" — the real model string + credentials
#
# Ollama models run locally, no API key needed.
# OpenRouter models need OPENROUTER_API_KEY.
# Direct Anthropic needs ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------

_OLLAMA_BASE = "http://localhost:11434"

router = Router(
    model_list=[
        # --- Local (Ollama) ---
        {
            "model_name": "llama3.1",
            "litellm_params": {
                "model": "ollama/llama3.1:latest",
                "api_base": _OLLAMA_BASE,
            },
        },
        {
            "model_name": "qwen2.5:32b",
            "litellm_params": {
                "model": "ollama/qwen2.5:32b",
                "api_base": _OLLAMA_BASE,
            },
        },
        {
            "model_name": "qwen3:32b",
            "litellm_params": {
                "model": "ollama/qwen3:32b",
                "api_base": _OLLAMA_BASE,
            },
        },
        # --- OpenRouter (remote, one API key, many providers) ---
        {
            "model_name": "claude-sonnet",
            "litellm_params": {
                "model": "openrouter/anthropic/claude-sonnet-4.6",
                "api_key": os.getenv("MY_ANTHROPIC_KEY"),
            },
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "openrouter/openai/gpt-4o",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
            },
        },
        {
            "model_name": "deepseek-r1",
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-r1-0528",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "gemini-2.5-pro",
            "litellm_params": {
                "model": "openrouter/google/gemini-2.5-pro",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
            },
        },
        {
            "model_name": "gemini-2.5-flash",   # ~10x cheaper than pro, similar reasoning
            "litellm_params": {
                "model": "openrouter/google/gemini-2.5-flash-preview-05-20",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
            },
        },
        {
            "model_name": "deepseek-v3",         # cheap, strong, not a thinking model
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-chat-v3-0324",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
            },
        },
        {
            # Second model arm for the reproducibility study.  The obvious
            # reading of an unstable result is "the model is weak", so the
            # comparison has to exist for any claim about the framework to hold:
            # failure modes that reproduce across models are structural, ones
            # that do not are model capability.
            "model_name": "deepseek-v4-pro",
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-v4-pro",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
            },
        },
        {
            "model_name": "deepseek-v3.2",
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-v3.2",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
            },
        },
        {
            "model_name": "qwen3-235b",
            "litellm_params": {
                "model": "openrouter/qwen/qwen3-235b-a22b",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
            },
        },
        {
            "model_name": "kimi-k2",
            "litellm_params": {
                "model": "openrouter/moonshotai/kimi-k2",
                "api_key": os.getenv("MY_OPENROUTER_API_KEY"),
            },
        },
        {
            "model_name": "tencent-hy3-preview",
            "litellm_params": {
                "model": "openrouter/tencent/hy3-preview:free",
                "api_key": os.getenv("TENCENT_API_KEY"),
            },
        },
    ],
    num_retries=3,
    retry_after=10,
)


def call_llm(system: str, user: str, model: str, raw: bool = False) -> dict | list | str:
    """Single LLM call via the router.

    Args:
        system: system prompt
        user: user message
        model: alias from the model_list above, e.g. "claude-sonnet"
        raw: if True, return the plain text response instead of parsing JSON

    Returns:
        Parsed JSON (dict or list) when raw=False, plain string when raw=True.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    kwargs: dict = {"model": model, "messages": messages}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Streaming is required for thinking models (e.g. tencent-hy3-preview, deepseek-r1)
            # that return content=None in non-streaming mode.
            stream = router.completion(**kwargs, stream=True)
            content = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    content += delta
            content = content.strip()

            if raw:
                return content

            # Extract JSON — handles objects {...} or arrays [...], code fences, preamble text.
            # raw_decode stops exactly where the JSON ends, ignoring any trailing text.
            decoder = json.JSONDecoder()
            obj_start = next((i for i, c in enumerate(content) if c in "{["), None)
            if obj_start is None:
                raise ValueError(f"No JSON found in model response:\n{content}")
            result, _ = decoder.raw_decode(content, obj_start)
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}/{max_retries}] {e} — retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
