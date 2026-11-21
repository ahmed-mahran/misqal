"""Chat-model factory. Keys and URLs come from the environment or ``.env``; never from code,
never logged."""

from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

# Public OpenAI-compatible endpoints with a known default URL: "<prefix>": (url var, url, key var).
HOSTED = {
    "cerebras": ("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "groq": ("GROQ_BASE_URL", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("TOGETHER_BASE_URL", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "baseten": ("BASETEN_BASE_URL", "https://inference.baseten.co/v1", "BASETEN_API_KEY"),
}


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines); the existing environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip())


def _endpoint(prefix: str) -> tuple[str, str] | None:
    """(base_url, api_key) for an OpenAI-compatible prefix: a known host, or any private gateway
    configured as ``<PREFIX>_BASE_URL`` + ``<PREFIX>_API_KEY``."""
    if prefix in HOSTED:
        url_var, default_url, key_var = HOSTED[prefix]
        key = os.environ.get(key_var)
        if not key:
            raise SystemExit(f"{prefix}: needs {key_var} in the environment or .env")
        return os.environ.get(url_var, default_url), key
    env = prefix.upper().replace("-", "_")
    base_url, key = os.environ.get(f"{env}_BASE_URL"), os.environ.get(f"{env}_API_KEY")
    return (base_url, key) if base_url and key else None


# Default cap (USD) on a gateway key's spend per budget window, for every script that calls models.
DAILY_BUDGET = 25.0
# Jobs stop starting this many dollars below the cap, so jobs already running can finish under it.
BUDGET_MARGIN = 2.0

# prefix -> (checked at, (spend, end of the budget window) or None)
_spend_cache: dict[str, tuple[float, tuple[float, str] | None]] = {}
_spend_lock = threading.Lock()


def _spend_window(spec: str, max_age: float = 30.0) -> tuple[float, str] | None:
    prefix = spec.partition(":")[0]
    if prefix in HOSTED or not (endpoint := _endpoint(prefix)):
        return None
    with _spend_lock:
        checked, window = _spend_cache.get(prefix, (0.0, None))
        if time.monotonic() - checked < max_age:
            return window
        base_url, key = endpoint
        root = base_url.rstrip("/").removesuffix("/v1")
        req = urllib.request.Request(f"{root}/key/info", headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                info = json.load(resp)
            info = info.get("info", info)
            window = (float(info["spend"]), str(info.get("budget_reset_at")))
        except Exception:  # no spend reporting: no budget check
            window = None
        _spend_cache[prefix] = (time.monotonic(), window)
        return window


def daily_spend(spec: str, max_age: float = 30.0) -> float | None:
    """Spend (USD) in the current budget window of the key behind a private gateway prefix, as
    the gateway reports it at ``/key/info`` (LiteLLM proxies). None for hosted providers and for
    gateways that do not report spend. Cached for ``max_age`` seconds."""
    window = _spend_window(spec, max_age)
    return None if window is None else window[0]


def spend_start(spec: str, path: Path) -> float:
    """The key's spend when a run first checked it in the current budget window. Kept in ``path``
    (JSON, per gateway prefix) so every run in the same window counts from the same point, and
    marked anew in the next window. 0 for keys that report no spend."""
    window = _spend_window(spec)
    if window is None:
        return 0.0
    prefix = spec.partition(":")[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Several run processes share the file: lock across processes, replace atomically.
    with _spend_lock, open(path.with_suffix(path.suffix + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marks = json.loads(path.read_text()) if path.exists() else {}
        mark = marks.get(prefix)
        if not mark or mark["window"] != window[1]:
            mark = marks[prefix] = {"window": window[1], "spend": window[0]}
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(marks, indent=1) + "\n")
            tmp.replace(path)
        return mark["spend"]


def over_budget(spec: str, budget: float = DAILY_BUDGET, start_file: Path | None = None) -> bool:
    """Whether the key behind ``spec`` has spent ``budget`` (less ``BUDGET_MARGIN``) in its budget
    window, counted from the mark in ``start_file`` when given, else from the window's start."""
    spent = daily_spend(spec)
    if spent is None:
        return False
    return spent - (spend_start(spec, start_file) if start_file else 0.0) >= budget - BUDGET_MARGIN


def _reasoning_chat_openai():
    """ChatOpenAI that keeps the provider's reasoning text (``reasoning`` or ``reasoning_content``)
    in ``additional_kwargs["reasoning_content"]`` for traces; it is not sent back to the model.
    Provider data attached to tool calls (``extra_content``, e.g. Gemini's thought signatures,
    which Gemini requires back with the call) is kept and sent back with the same tool call."""
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI

    class ReasoningChatOpenAI(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            raw = response if isinstance(response, dict) else response.model_dump()
            for gen, choice in zip(result.generations, raw.get("choices") or []):
                msg = choice.get("message") or {}
                if text := msg.get("reasoning_content") or msg.get("reasoning"):
                    gen.message.additional_kwargs["reasoning_content"] = text
                content = gen.message.content
                if isinstance(content, str) and "<thought>" in content:  # Gemini thought summaries
                    thoughts = re.findall(r"<thought>(.*?)</thought>", content, flags=re.S)
                    gen.message.content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.S).strip()
                    gen.message.additional_kwargs["reasoning_content"] = "\n\n".join(thoughts)
                extra = {tc["id"]: tc["extra_content"] for tc in msg.get("tool_calls") or []
                         if tc.get("extra_content")}
                if extra:
                    gen.message.additional_kwargs["tool_call_extra"] = extra
            return result

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            sent = self._convert_input(input_).to_messages()
            for msg, out in zip(sent, payload.get("messages") or []):
                extra = isinstance(msg, AIMessage) and msg.additional_kwargs.get("tool_call_extra")
                for tc in (out.get("tool_calls") or []) if extra else []:
                    if tc.get("id") in extra:
                        tc["extra_content"] = extra[tc["id"]]
            return payload

    return ReasoningChatOpenAI


def make_model(spec: str, *, max_tokens: int = 4096, reasoning: bool = False,
               timeout: float = 180.0, temperature: float | None = 0.0) -> BaseChatModel:
    """``cerebras:gpt-oss-120b``, ``mygw:some-model`` (with ``MYGW_BASE_URL``/``MYGW_API_KEY``
    set), or any ``init_chat_model`` id such as ``openai:gpt-4.1-mini``. ``reasoning`` = medium
    effort for gpt-oss models. ``temperature=None`` leaves the provider default (some reasoning
    models accept no other value)."""
    from langchain.chat_models import init_chat_model

    extra: dict = {} if temperature is None else {"temperature": temperature}
    prefix, _, name = spec.partition(":")
    if name and (endpoint := _endpoint(prefix)):
        base_url, key = endpoint
        if "gpt-oss" in name:
            extra["reasoning_effort"] = "medium" if reasoning else "low"
        if prefix == "google":  # Gemini thinks either way; this returns its thought summaries
            extra["extra_body"] = {"extra_body": {"google": {"thinking_config": {"include_thoughts": True}}}}
        return _reasoning_chat_openai()(model=name, base_url=base_url, api_key=key,
                                        max_tokens=max_tokens, timeout=timeout, max_retries=6,
                                        **extra)
    return init_chat_model(spec, max_tokens=max_tokens, timeout=timeout, **extra)
