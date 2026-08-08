"""Shared Azure OpenAI client.

Both agent.py and tools.py need to talk to the model now: agent.py runs the tool
loop, and search_courses asks the model for courses. This module exists so
neither has to import the other.
"""

from __future__ import annotations

import json
import os
import random
import re
import time

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI, APIConnectionError, APITimeoutError

import dns_cache

_client: AzureOpenAI | None = None
_deployment: str | None = None

# The agent loop makes many model calls with slow tool work in between. httpx
# keeps an idle connection for only 5s by default, so every call after a slow
# tool step reopens the socket - a fresh DNS lookup and TLS handshake each time.
# A home router's DNS forwarder drops queries under that kind of burst, which
# surfaces as `getaddrinfo failed` -> APIConnectionError. Holding the connection
# open for the whole session means we resolve the host roughly once.
_LIMITS = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=10,
    keepalive_expiry=300.0,
)
_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=15.0)
_MAX_RETRIES = 3

# Belt-and-braces retry around the SDK's own, for when a DNS/network blip
# outlasts the built-in attempts. DNS outages here last ~10-20s, so back off
# far enough to actually outlive one.
_RETRY_ATTEMPTS = 4
_RETRY_BASE_DELAY = 3.0


def _diagnose(exc: BaseException) -> str:
    """Turn the underlying transport failure into advice worth acting on."""
    cause = exc.__cause__ or exc
    text = f"{type(cause).__name__}: {cause}".lower()

    if isinstance(exc, APITimeoutError):
        return ("The request timed out waiting for a reply. The deployment may be "
                "overloaded - retry, or raise the read timeout in azure_client.py.")
    if "getaddrinfo" in text or "11001" in text or "name or service not known" in text:
        return ("DNS could not resolve the endpoint host. This is usually the local "
                "resolver dropping queries, not Azure. Try a public DNS server "
                "(1.1.1.1 or 8.8.8.8) on your active network adapter, or switch "
                "network - `nslookup <your endpoint host>` will confirm.")
    if "certificate" in text or "ssl" in text:
        return ("TLS verification failed - typically a proxy re-signing traffic. "
                "Point SSL_CERT_FILE at your corporate root CA bundle.")
    if "refused" in text or "unreachable" in text or "timed out" in text:
        return ("The host refused or dropped the connection - check VPN, firewall, "
                "or whether the Azure resource still exists.")
    return ("Check that you are online and that a proxy, VPN, or firewall is not "
            "blocking HTTPS egress for Python.")


def call_with_retry(make_call):
    """Run `make_call()`, retrying connection failures with exponential backoff.

    Only APIConnectionError is retried - that is the "never got a response"
    family (DNS, TLS, reset socket, timeout). Real API errors (401, 404, 429,
    content policy) are raised straight away, because retrying them is useless.
    """
    last: APIConnectionError | None = None

    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return make_call()
        except APIConnectionError as exc:
            last = exc
            if attempt == _RETRY_ATTEMPTS - 1:
                break
            delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            cause = exc.__cause__ or exc
            print(
                f"[azure_client] {type(cause).__name__}: {cause} - retrying in "
                f"{delay:.1f}s ({attempt + 1}/{_RETRY_ATTEMPTS - 1})",
                flush=True,
            )
            time.sleep(delay)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "(unset)").strip()
    cause = last.__cause__ if last is not None else None
    detail = f"{type(cause).__name__}: {cause}" if cause else "no further detail"
    raise RuntimeError(
        f"Could not reach Azure OpenAI at {endpoint} after {_RETRY_ATTEMPTS} "
        f"attempts.\n  Underlying error: {detail}\n  {_diagnose(last)}"
    ) from last


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

    # Survive the local resolver dropping queries mid-run - see dns_cache.py.
    dns_cache.install()

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
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
        http_client=httpx.Client(limits=_LIMITS, timeout=_TIMEOUT),
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
        response = call_with_retry(lambda: client.chat.completions.create(**kwargs))
    except Exception as exc:
        # Older, non-reasoning deployments want max_tokens instead.
        if "max_completion_tokens" in str(exc):
            kwargs.pop("max_completion_tokens")
            kwargs["max_tokens"] = max_tokens
            response = call_with_retry(lambda: client.chat.completions.create(**kwargs))
        else:
            raise

    return extract_json(response.choices[0].message.content or "")
