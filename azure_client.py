"""Shared Azure OpenAI client.

Both agent.py and tools.py need to talk to the model now: agent.py runs the tool
loop, and search_courses asks the model for courses. This module exists so
neither has to import the other.
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from openai import AzureOpenAI

_client: AzureOpenAI | None = None
_deployment: str | None = None


def build_client() -> tuple[AzureOpenAI, str]:
    """Create the Azure client from environment variables, cached after the first call.

    Note this is AzureOpenAI, not OpenAI: Azure authenticates with an `api-key`
    header against your own resource endpoint, and the "model" you pass to the
    API is your *deployment* name, not a public model name.
    """
    global _client, _deployment

    if _client is not None and _deployment is not None:
        return _client, _deployment

    load_dotenv()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

    missing = [
        name for name, value in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_API_VERSION", api_version),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
        ] if not value
    ]
    if missing:
        raise SystemExit(
            "Missing environment variables: " + ", ".join(missing) +
            "\nCopy .env.example to .env and fill it in. Never paste keys into chat or code."
            "\nIf the variable clearly IS in .env, re-save the file as UTF-8 without BOM."
        )

    _client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    _deployment = deployment
    return _client, _deployment


def extract_json(raw: str) -> dict | None:
    """Pull a JSON object out of a model message, tolerating markdown fences and prose."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def ask_for_json(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> dict | None:
    """One plain completion (no tools) whose answer is expected to be a JSON object."""
    client, deployment = build_client()

    kwargs = dict(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_tokens,
    )
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Older, non-reasoning deployments want max_tokens instead.
        if "max_completion_tokens" in str(exc):
            kwargs.pop("max_completion_tokens")
            kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    return extract_json(response.choices[0].message.content or "")
