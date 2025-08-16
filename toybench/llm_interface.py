# llm_interface.py
"""LLM interface abstractions for ToyBench.

* Google Gemini models via `google-genai`
* OpenAI models via both
      * `/v1/chat/completions` – used for true conversational history, and
      * `/v1/responses`         – kept for single-shot prompts and the legacy flattened-string path.
* xAI Grok models via `openai` with custom base URL.
* Quality Compute Simulator – added for integration with the custom backend simulator.
* Anthropic Claude models via `anthropic`

The conversational path now sends an array of role-tagged messages
(`user` / `assistant`) instead of concatenating everything into one big string,
fully matching the OpenAI chat-completion spec.

MODIFIED TO EXTRACT AND RETURN TOKEN USAGE DATA AND RAW API RESPONSE OBJECT.
CORRECTED TYPE HINTS FOR STATIC ANALYSIS USING TYPE_CHECKING.
MODIFIED GeminiInterface to improve text extraction and logging on failure, handling `RepeatedComposite` types, and to support thinkingBudget extraction.
Improved OpenAIInterface methods (preferring chat, better multimodal).
ADDED: xAI GrokInterface for Grok 3 Mini with reasoning support.
UPDATED: Added support in GrokInterface for models that may not support reasoning_effort.
ADDED: QualityComputeInterface for calling the Quality Compute simulator backend.
ADDED: AnthropicInterface for calling Anthropic Claude models.
UPDATED: Added support for Anthropic extended thinking with streaming.
FIXED: Removed erroneous 'stream': True from Anthropic stream call to resolve TypeError.
UPDATED: QualityComputeInterface to support additional parameters like max_tokens via **kwargs.
UPDATED: GeminiInterface to support thinkingBudget for Gemini 2.5 models and extract thoughts_token_count for better logging and serialization.
FIXED: GeminiInterface now passes thinking_config using google.genai.types objects as required by the client library.
UPDATED: Fixed image part handling in Gemini multimodal calls with added error handling and logging for Pydantic validation.
ADDED: Enhanced logging for image validation errors in Gemini API calls.
UPDATED: OpenRouterInterface to support reasoning_effort for compatibility with models like gpt-oss-120b.
ADDED: OpenAIInterface now strips None-valued kwargs (e.g., max_tokens=None) before API calls to avoid 400 errors.
UPDATED: Added configurable reasoning effort for OpenAI models.
UPDATED: Added support for OpenAI background mode (async) with polling.
FIXED: Decoupled OpenRouterInterface from OpenAIInterface to prevent metadata leakage.
UPDATED: Added provider_name to all interfaces for better scoping and metadata management.
FIXED: Ensured time.sleep() usage is correct by importing time.
UPDATED: Modified OpenAI background polling logging to reduce verbosity: log only start, errors, and completion with timers.
"""

from __future__ import annotations

import logging
import time  # used for polling / sleep
import json
import base64
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Import heavy dependencies only for type checking
if TYPE_CHECKING:
    from google.genai.types import (
        Candidate,
        GenerateContentResponse,
        PromptFeedback,
        UsageMetadata,
        HttpOptions,
    )
    from google.genai import types as genai_types
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice as OpenAIChoice
    from openai.types.completion_usage import CompletionUsage
    from anthropic.types import Message

# Runtime imports (gracefully handle missing libs)
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Also import the openai module object (used for RateLimitError checks)
try:
    import openai
except Exception:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

import requests
import os

logger = logging.getLogger(__name__)

# Type aliases
TokenUsage = Optional[Dict[str, int | None]]
RawAPIResponse = Optional[Any]
LLMResponse = Tuple[Optional[str], TokenUsage, RawAPIResponse]

# ---------------------------------------------------------------------------
#  Base abstract interface
# ---------------------------------------------------------------------------
class LLMInterface(ABC):
    def __init__(self, api_key: str, model_name: str, provider_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.provider_name = provider_name

    @abstractmethod
    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        pass

# ---------------------------------------------------------------------------
#  Gemini implementation
# ---------------------------------------------------------------------------
class GeminiInterface(LLMInterface):
    def __init__(self, api_key: str, model_name: str, provider_name: str, thinking_enabled: bool = True, thinking_budget: Optional[int] = None):
        super().__init__(api_key, model_name, provider_name)
        if not api_key:
            raise ValueError("Gemini API Key is required.")
        if genai is None or genai_types is None:
            raise ImportError("google-genai library is not installed or types module is missing.")

        try:
            http_options = genai_types.HttpOptions(timeout=1000000)
            self.client = genai.Client(api_key=api_key, http_options=http_options)
            self.thinking_enabled = thinking_enabled
            self.thinking_budget = thinking_budget
            self._validate_model_and_thinking_budget()
            logger.info("GeminiInterface initialised with model: %s, thinking_enabled: %s, thinking_budget: %s", model_name, thinking_enabled, thinking_budget if thinking_budget is not None else "auto")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"Gemini initialization failed: {e}") from e

    def _validate_model_and_thinking_budget(self):
        model_lower = self.model_name.lower()
        is_flash_2_5 = "2.5-flash" in model_lower
        is_pro_2_5 = "2.5-pro" in model_lower or "2.5-pro-preview" in model_lower

        if is_pro_2_5:
            if self.thinking_budget is not None:
                if not (128 <= self.thinking_budget <= 32768):
                    logger.warning("Invalid thinking_budget for Gemini 2.5 Pro. Resetting to auto.")
                    self.thinking_budget = None
                    self.thinking_enabled = True
            else:
                self.thinking_enabled = True
        elif is_flash_2_5:
            if self.thinking_budget is not None:
                if not (0 <= self.thinking_budget <= 24576):
                    logger.warning("Invalid thinking_budget for Gemini 2.5 Flash. Resetting to auto.")
                    self.thinking_budget = None
                    self.thinking_enabled = True
                elif self.thinking_budget == 0:
                    self.thinking_enabled = False
                else:
                    self.thinking_enabled = True
            else:
                self.thinking_enabled = True
        else:
            if self.thinking_budget is not None or self.thinking_enabled:
                logger.warning("Model does not support thinkingBudget. Disabling thinking.")
            self.thinking_enabled = False
            self.thinking_budget = None

    def _call_api(self, prompt_or_contents: Any, retries: int = 3, delay: int = 10) -> LLMResponse:
        resp: Optional["GenerateContentResponse"] = None
        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        raw_api_response_for_return = None
        contents_to_use = []

        if isinstance(prompt_or_contents, str):
            contents_to_use.append(genai_types.Content(role="user", parts=[genai_types.Part(text=prompt_or_contents)]))
        elif isinstance(prompt_or_contents, list):
            for item in prompt_or_contents:
                if isinstance(item, dict) and 'role' in item and 'parts' in item:
                    parts_list = []
                    for part in item.get('parts', []):
                        if isinstance(part, str):
                            parts_list.append(genai_types.Part(text=part))
                        elif isinstance(part, dict):
                            if "text" in part:
                                parts_list.append(genai_types.Part(text=part["text"]))
                            elif "inline_data" in part:
                                inline_data_dict = part["inline_data"]
                                mime_type = inline_data_dict.get("mime_type")
                                data = inline_data_dict.get("data")
                                if mime_type and data:
                                    try:
                                        data_bytes = base64.b64decode(data) if isinstance(data, str) else data
                                        parts_list.append(genai_types.Part.from_bytes(mime_type=mime_type, data=data_bytes))
                                    except Exception as e:
                                        logger.error(f"Failed to create Part from data: {e}", exc_info=True)
                            else:
                                logger.warning(f"Unsupported part dict structure: {part}.")
                        else:
                            logger.warning(f"Unsupported part type in history: {type(part)} / {part}.")
                    if parts_list:
                        contents_to_use.append(genai_types.Content(role=item['role'], parts=parts_list))
                else:
                    logger.warning(f"Skipping invalid item in contents: {item}.")

        for attempt in range(retries):
            try:
                config_obj = None
                if self.thinking_enabled and genai_types is not None:
                    thinking_params = {}
                    if self.thinking_budget is not None:
                        thinking_params["thinking_budget"] = self.thinking_budget
                    config_obj = genai_types.GenerateContentConfig(thinking_config=genai_types.ThinkingConfig(**thinking_params))

                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents_to_use,
                    config=config_obj,
                )
                raw_api_response_for_return = self._serialize_response(resp)

                # Extract text
                text_response = None
                prompt_feedback: Optional["PromptFeedback"] = getattr(resp, 'prompt_feedback', None)
                candidates_list = getattr(resp, 'candidates', None)
                if candidates_list and len(candidates_list) > 0:
                    cand0: Optional["Candidate"] = candidates_list[0]
                    if cand0 and getattr(cand0, 'content', None) and hasattr(cand0.content, 'parts'):
                        for part in getattr(cand0.content, 'parts', []) or []:
                            if hasattr(part, 'text') and isinstance(part.text, str) and part.text:
                                text_response = part.text
                                break

                # Token usage
                usage_meta: Optional["UsageMetadata"] = getattr(resp, 'usage_metadata', None)
                if usage_meta:
                    token_usage = {
                        k: v
                        for k, v in {
                            "input_tokens": getattr(usage_meta, 'prompt_token_count', None),
                            "output_tokens": getattr(usage_meta, 'candidates_token_count', None),
                            "reasoning_tokens": getattr(usage_meta, 'thoughts_token_count', None),
                            "total_tokens": getattr(usage_meta, 'total_token_count', None),
                        }.items()
                        if v is not None
                    }

                return text_response, token_usage, raw_api_response_for_return
            except Exception as e:
                logger.error(f"Gemini API error (Attempt {attempt+1}/{retries}): {e}", exc_info=True)
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    return None, None, None
        return None, None, None

    def _serialize_response(self, response: Any) -> Dict:
        try:
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            if hasattr(response, 'to_dict'):
                return response.to_dict()

            if isinstance(response, (dict, list)):
                return response

            try:
                return json.loads(json.dumps(response.__dict__))
            except Exception:
                serializable_dict = {}
                for key in dir(response):
                    if not key.startswith('_') and not callable(getattr(response, key)):
                        value = getattr(response, key)
                        if hasattr(value, 'model_dump'):
                            serializable_dict[key] = value.model_dump()
                        elif hasattr(value, 'to_dict'):
                            serializable_dict[key] = value.to_dict()
                        elif isinstance(value, (int, float, str, bool, type(None), list, dict)):
                            serializable_dict[key] = value
                        else:
                            serializable_dict[key] = str(value)
                return serializable_dict
        except Exception as e:
            logger.warning(f"Failed to serialize Gemini response: {e}", exc_info=True)
            return {}

    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        return self._call_api(prompt)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        return self._call_api(history)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        return self._call_api(contents)

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        return self._call_api(prompt)

# ---------------------------------------------------------------------------
#  OpenAI implementation - UPDATED FOR TIMEOUT, RETRY, AND BACKGROUND MODE HANDLING
# ---------------------------------------------------------------------------
class OpenAIInterface(LLMInterface):
    """Uses /v1/responses for all interactions."""
    def __init__(
        self,
        api_key: str,
        model_name: str,
        provider_name: str,
        reasoning_effort: str = "high",
        timeout: Optional[float] = 1200,  # 20 minutes default
        max_retries: int = 1,             # minimize silent SDK retries
        background_enabled: bool = False,
        background_poll_interval: float = 2.0,
    ):
        super().__init__(api_key, model_name, provider_name)
        if not api_key:
            raise ValueError("OpenAI API Key is required.")
        if OpenAI is None:
            raise ImportError("openai library is not installed. Please install it to use OpenAIInterface.")

        try:
            self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
            self.model = model_name
            self.reasoning_effort = reasoning_effort
            self.timeout = timeout
            self.max_retries = max_retries
            self.background_enabled = background_enabled
            self.background_poll_interval = background_poll_interval
            logger.info(
                "OpenAIInterface initialised with model: %s, reasoning_effort: %s, timeout=%ss, max_retries=%s, background_enabled=%s, background_poll_interval=%ss",
                model_name, self.reasoning_effort, timeout, max_retries, background_enabled, background_poll_interval
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"OpenAI initialization failed: {e}") from e

    def _should_add_reasoning(self) -> bool:
        """Return True if we should include OpenAI 'reasoning' block for GPT‑5 family."""
        name = (self.model or "").lower()
        return name.startswith("gpt-5")

    @staticmethod
    def _as_dict(obj: Any) -> Any:
        if obj is None:
            return obj
        if hasattr(obj, "model_dump"):
            try:
                d = obj.model_dump()
                return d if d is not None else obj
            except Exception:
                pass
        if hasattr(obj, "dict"):
            try:
                d = obj.dict()
                return d if d is not None else obj
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            d = getattr(obj, "__dict__")
            return d if d is not None else obj
        return obj

    def _build_responses_input(self, input_data):
        if isinstance(input_data, str):
            return input_data  # Direct string input for single-shot
        elif isinstance(input_data, list):
            messages = []
            for item in input_data:
                role = "assistant" if item.get("role") == "model" else "user"
                content_list = []
                parts = item.get("parts", [])
                for part in parts:
                    # Per /v1/responses: assistant turns must use output_text,
                    # user turns use input_text (and input_image if present).
                    text_type = "output_text" if role == "assistant" else "input_text"

                    if isinstance(part, str):
                        content_list.append({"type": text_type, "text": part})
                    elif isinstance(part, dict):
                        if "text" in part:
                            content_list.append({"type": text_type, "text": part["text"]})
                        # Only allow images on user turns for input; skip on assistant turns.
                        elif role == "user" and "source" in part and "inline_data" in part["source"]:
                            inline_data = part["source"]["inline_data"]
                            mime_type = inline_data.get("mime_type")
                            data = inline_data.get("data")
                            if mime_type and data:
                                image_url = f"data:{mime_type};base64,{data}"
                                content_list.append({"type": "input_image", "image_url": {"url": image_url}})

                if content_list:
                    messages.append({"role": role, "content": content_list})
            return messages
        else:
            raise ValueError("Invalid input data type for Responses API")

    def _call_responses_api(self, input_param, **kwargs) -> LLMResponse:
        text_response = None
        token_usage = None
        resp = None

        try:
            # Build body
            body = {
                "model": self.model,
                "input": input_param,  # Can be str or list of messages
                "tool_choice": "none",
            }

            # Add other parameters from kwargs if not None
            for key in ["instructions", "max_output_tokens", "temperature", "top_p", "stream"]:
                if key in kwargs and kwargs[key] is not None:
                    body[key] = kwargs[key]

            # Handle max_tokens input, map to max_output_tokens
            if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
                body["max_output_tokens"] = kwargs["max_tokens"]

            # Add reasoning for GPT-5 models
            if self._should_add_reasoning():
                body["reasoning"] = {"effort": self.reasoning_effort}

            # Background mode handling
            if self.background_enabled:
                body["background"] = True
                body["store"] = True  # Required for background mode
                if "stream" in body and body["stream"]:
                    logger.warning("Streaming is not supported in background mode; disabling stream.")
                    body["stream"] = False

            # Strip None-valued keys to avoid errors
            body = {k: v for k, v in body.items() if v is not None}

            # Idempotency key per call (prevents duplicate charges if client times out)
            idempotency_key = f"toybench-{uuid.uuid4().hex}"
            extra_headers = {"Idempotency-Key": idempotency_key}

            start_ts = time.time()  # Start timing the request
            if self.background_enabled:
                # Create background response and poll until completion
                resp = self.client.responses.create(**body, extra_headers=extra_headers, timeout=self.timeout)
                resp_id = getattr(resp, "id", None)
                status = getattr(resp, "status", None)
                if resp_id:
                    if status in {"queued", "in_progress"}:
                        # Polling start log (once)
                        poll_count = 0
                        logger.info(
                            "OpenAI background polling started for response_id=%s initial_status=%s interval=%.2fs",
                            resp_id, status, self.background_poll_interval
                        )
                        # Poll loop (no per-iteration logs)
                        while status in {"queued", "in_progress"}:
                            time.sleep(self.background_poll_interval)  # Use time.sleep for correctness
                            poll_count += 1
                            try:
                                polled_resp = self.client.responses.retrieve(resp_id, extra_headers=extra_headers, timeout=self.timeout)
                                if polled_resp:
                                    resp = polled_resp
                                    status = getattr(resp, "status", None)
                            except Exception as pe:
                                # Error during polling (log and continue)
                                logger.error(
                                    "OpenAI background polling error for response_id=%s (poll #%d): %s",
                                    resp_id, poll_count, pe
                                )
                                continue
                        # Polling completion log (once)
                        elapsed = time.time() - start_ts
                        logger.info(
                            "OpenAI background polling complete for response_id=%s terminal_status=%s polls=%d elapsed=%.2fs",
                            resp_id, status, poll_count, elapsed
                        )
                    else:
                        # Terminal immediately without polling
                        elapsed = time.time() - start_ts
                        logger.info("OpenAI background response %s returned terminal status '%s' without polling in %.2fs",
                                    resp_id, status, elapsed)
                else:
                    logger.error(f"Background response creation failed or missing ID/status: {resp}")
                    return None, None, None
            else:
                # Synchronous call
                resp = self.client.responses.create(**body, extra_headers=extra_headers, timeout=self.timeout)
                elapsed = time.time() - start_ts
                logger.debug(f"OpenAI responses.create completed in {elapsed:.2f}s (idempotency_key={idempotency_key})")

            # Extract text from response: find the 'message' item and join its output_text parts or use output_text directly
            if resp and getattr(resp, "output", None):
                for item in resp.output:
                    if getattr(item, "type", None) == "message" and getattr(item, "content", None):
                        text_parts = [
                            getattr(part, "text", "")
                            for part in item.content
                            if hasattr(part, "text") and getattr(part, "text", "")
                        ]
                        candidate = " ".join([t for t in text_parts if t]).strip()
                        if candidate:
                            text_response = candidate
                            break
            # Prefer output_text if available (esp. for background responses)
            if text_response is None and resp is not None and hasattr(resp, "output_text"):
                text_candidate = getattr(resp, "output_text", None)
                if isinstance(text_candidate, str) and text_candidate.strip():
                    text_response = text_candidate.strip()
            if text_response is None:
                logger.warning("No 'message' item or 'output_text' found in response output.")

            # Extract token usage
            if resp and getattr(resp, "usage", None):
                usage_data = self._as_dict(resp.usage)
                if isinstance(usage_data, dict):
                    token_usage = {
                        "input_tokens": usage_data.get("input_tokens"),
                        "output_tokens": usage_data.get("output_tokens"),
                        "total_tokens": usage_data.get("total_tokens"),
                        "cached_input_tokens": usage_data.get("input_tokens_details", {}).get("cached_tokens") if isinstance(usage_data.get("input_tokens_details"), dict) else None,
                        "reasoning_output_tokens": usage_data.get("output_tokens_details", {}).get("reasoning_tokens") if isinstance(usage_data.get("output_tokens_details"), dict) else None,
                        # Background mode annotations for observability (scoped to OpenAI)
                        "background": True if self.background_enabled else None,
                        "status": getattr(resp, "status", None) if self.background_enabled else None,
                    }
                    token_usage = {k: v for k, v in token_usage.items() if v is not None}

            raw_api_response_serialized = self._serialize_response(resp)
            return text_response, token_usage, raw_api_response_serialized

        except Exception as e:
            logger.error("OpenAI /responses API error: %s", e, exc_info=True)
            return None, None, None

    def _serialize_response(self, response: Any) -> Dict:
        try:
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            if hasattr(response, 'to_dict'):
                return response.to_dict()

            serializable_dict = {}
            for key in dir(response):
                if not key.startswith('_') and not callable(getattr(response, key)):
                    value = getattr(response, key)
                    if hasattr(value, 'model_dump'):
                        serializable_dict[key] = value.model_dump()
                    elif hasattr(value, 'to_dict'):
                        serializable_dict[key] = value.to_dict()
                    elif isinstance(value, (int, float, str, bool, type(None), list, dict)):
                        serializable_dict[key] = value
                    else:
                        serializable_dict[key] = str(value)
            return serializable_dict
        except Exception as e:
            logger.warning(f"Failed to serialize OpenAI response: {e}", exc_info=True)
            return {}

    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        return self._call_responses_api(prompt, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        responses_input = self._build_responses_input(history)
        return self._call_responses_api(responses_input, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        responses_input = self._build_responses_input(contents)
        return self._call_responses_api(responses_input, **kwargs)

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        return self._call_responses_api(prompt, **kwargs)

# ---------------------------------------------------------------------------
#  xAI Grok implementation
# ---------------------------------------------------------------------------
class GrokInterface(LLMInterface):
    def __init__(self, api_key: str, model_name: str, provider_name: str, reasoning_effort: str = "low"):
        super().__init__(api_key, model_name, provider_name)
        if not api_key:
            raise ValueError("xAI API Key is required for GrokInterface.")
        if OpenAI is None:
            raise ImportError("openai library is not installed. Please install it to use GrokInterface.")

        self.reasoning_effort = (reasoning_effort or "low").lower()
        if self.reasoning_effort not in ["low", "high"]:
            logger.warning(f"Invalid reasoning_effort '{reasoning_effort}'. Defaulting to 'low'.")
            self.reasoning_effort = "low"

        self.supports_reasoning = model_name in ["grok-3-mini-beta", "grok-3-mini-fast-beta"]
        if not self.supports_reasoning:
             logger.warning(f"Model '{model_name}' may not support reasoning_effort. It will be omitted if unsupported.")

        try:
            self.client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)
            self.model = model_name
            logger.info(f"GrokInterface initialised with model: {model_name}, reasoning_effort: {self.reasoning_effort if self.supports_reasoning else 'N/A'}")
        except Exception as e:
            logger.error(f"Failed to initialize Grok client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"Grok initialization failed: {e}") from e

    def _call_grok_chat_api(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        resp: Optional[Any] = None
        retries = 20
        base_delay = 1
        max_delay = 60

        for attempt in range(retries):
            try:
                # Build kwargs safely (strip None)
                safe_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                safe_kwargs['model'] = self.model
                safe_kwargs['messages'] = messages
                if self.supports_reasoning:
                    safe_kwargs["reasoning_effort"] = self.reasoning_effort

                resp = self.client.chat.completions.create(**safe_kwargs)

                # Serialize raw
                raw_api_response = self._serialize_response(resp)

                # Text
                choice_list = getattr(resp, 'choices', None)
                if choice_list and len(choice_list) > 0:
                    first_choice = choice_list[0]
                    text_response = first_choice.message.content

                # Usage
                usage = getattr(resp, 'usage', None)
                if usage:
                    completion_tokens_details = getattr(usage, 'completion_tokens_details', None)
                    reasoning_tokens = getattr(completion_tokens_details, 'reasoning_tokens', None) if completion_tokens_details else None
                    token_usage = {
                        "input_tokens": getattr(usage, 'prompt_tokens', None),
                        "output_tokens": getattr(usage, 'completion_tokens', None),
                        "total_tokens": getattr(usage, 'total_tokens', None),
                        "reasoning_tokens": reasoning_tokens if self.supports_reasoning else None,
                    }

                return text_response, token_usage, raw_api_response
            except openai.RateLimitError as e:
                if attempt < retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Grok rate limit 429. Retrying in {delay}s... ({attempt+1}/{retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Grok API rate limit exceeded after {retries} attempts: {e}", exc_info=True)
                    return None, None, None
            except Exception as e:
                logger.error(f"Grok API error: {e}", exc_info=True)
                return None, None, None
        return None, None, None

    def _serialize_response(self, response: Any) -> Dict:
        try:
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            if hasattr(response, 'to_dict'):
                return response.to_dict()

            serializable_dict = {}
            for key in dir(response):
                if not key.startswith('_') and not callable(getattr(response, key)):
                    value = getattr(response, key)
                    if hasattr(value, 'model_dump'):
                        serializable_dict[key] = value.model_dump()
                    elif hasattr(value, 'to_dict'):
                        serializable_dict[key] = value.to_dict()
                    elif isinstance(value, (int, float, str, bool, type(None), list, dict)):
                        serializable_dict[key] = value
                    else:
                        serializable_dict[key] = str(value)
            return serializable_dict
        except Exception as e:
            logger.warning(f"Failed to serialize Grok response: {e}", exc_info=True)
            return {}

    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_grok_chat_api(messages, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        # Convert Gemini style history to OpenAI style
        openai_messages: List[Dict[str, Any]] = []
        last_role = None
        for entry in history:
            role = "assistant" if entry.get("role") == "model" else "user"
            parts = entry.get("parts", [])
            text_parts = []
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, str):
                        text_parts.append(p)
                    elif isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                        text_parts.append(p["text"])
            elif isinstance(parts, str):
                text_parts.append(parts)

            content = "\n".join(text_parts).strip()
            if not content:
                continue

            if openai_messages and last_role == role:
                openai_messages[-1]["content"] += "\n" + content
            else:
                openai_messages.append({"role": role, "content": content})
                last_role = role

        if not openai_messages:
            return None, None, None

        return self._call_grok_chat_api(openai_messages, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        logger.warning("Multimodal generation is not supported by xAI Grok models.")
        return None, None, None

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_grok_chat_api(messages, **kwargs)

# ---------------------------------------------------------------------------
#  OpenRouter implementation - UPDATED TO EXTEND LLMINTERFACE DIRECTLY
# ---------------------------------------------------------------------------
class OpenRouterInterface(LLMInterface):
    """OpenRouter via OpenAI-compatible API, with provider options and reasoning effort."""
    def __init__(self, api_key: str, model_name: str, provider_name: str, allow_fallbacks: bool = True, sort: str = "price", reasoning_effort: str = "low", provider: Optional[str] = None):
        super().__init__(api_key, model_name, provider_name)
        try:
            self.client = requests.Session()  # Use requests directly for OpenRouter
            self.allow_fallbacks = allow_fallbacks
            self.sort = sort
            self.provider = provider
            self.reasoning_effort = (reasoning_effort or "low").lower()
            logger.info(
                "OpenRouterInterface initialised with model: %s, allow_fallbacks=%s, sort=%s, reasoning_effort=%s, provider=%s",
                model_name, allow_fallbacks, sort, self.reasoning_effort, provider
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"OpenRouter initialization failed: {e}") from e

    def _call_chat_api(self, messages: list[dict], **kwargs) -> LLMResponse:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        provider_config = {}
        if self.provider:
            # Force a specific provider and disable fallbacks.
            provider_config["order"] = [self.provider]
            provider_config["allow_fallbacks"] = False
        else:
            # Use default routing options.
            provider_config["allow_fallbacks"] = self.allow_fallbacks
            provider_config["sort"] = self.sort

        body = {
            "model": self.model_name,
            "messages": messages,
            "provider": provider_config,
            "reasoning": {"effort": self.reasoning_effort},
        }

        # Merge optional params if not None
        for k in ("max_tokens", "temperature", "top_p", "stream"):
            if k in kwargs and kwargs[k] is not None:
                body[k] = kwargs[k]

        try:
            resp = self.client.post(url, headers=headers, json=body, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage")
            token_usage = None
            if usage:
                token_usage = {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            return text, token_usage, data
        except Exception as e:
            logger.error("OpenRouter HTTP error: %s", e, exc_info=True)
            return None, None, None

    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_chat_api(messages, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        # Convert to OpenAI-style messages
        openai_messages = []
        last_role = None
        for entry in history:
            role = "assistant" if entry.get("role") == "model" else "user"
            parts = entry.get("parts", [])
            text_parts = []
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, str):
                        text_parts.append(p)
                    elif isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                        text_parts.append(p["text"])
            elif isinstance(parts, str):
                text_parts.append(parts)
            content = "\n".join(text_parts).strip()
            if not content:
                continue
            if openai_messages and last_role == role:
                openai_messages[-1]["content"] += "\n" + content
            else:
                openai_messages.append({"role": role, "content": content})
                last_role = role
        if not openai_messages:
            return None, None, None
        return self._call_chat_api(openai_messages, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        logger.warning("Multimodal generation is not supported by OpenRouterInterface. Returning None.")
        return None, None, None

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_chat_api(messages, **kwargs)

# ---------------------------------------------------------------------------
#  Quality Compute Simulator implementation
# ---------------------------------------------------------------------------
class QualityComputeInterface(LLMInterface):
    def __init__(self, api_key: str, model_name: str, provider_name: str, **kwargs):
        super().__init__(api_key, model_name, provider_name)
        self.base_url = os.environ.get("QUALITY_COMPUTE_URL", "https://qualitycompute.henosis.us")
        if not api_key:
            raise ValueError("Quality Compute API Key is required.")
        self.config = kwargs
        self.is_collaborative = self.config.get('use_collaborative_agent', False)
        logger.info(f"QualityComputeInterface initialised. Collaborative Mode: {self.is_collaborative}")

    def _make_api_call(self, endpoint: str, payload: dict) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        full_url = f"{self.base_url}{endpoint}"

        try:
            response = requests.post(full_url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            return data.get("selected_text"), data.get("usage"), data
        except requests.RequestException as e:
            logger.error(f"API call to {full_url} failed: {e}", exc_info=True)
            error_response = {"error": str(e)}
            try:
                error_response = e.response.json()
            except Exception:
                pass
            return None, None, error_response

    def _prepare_and_call(self, prompt_or_history: str | list) -> LLMResponse:
        if self.is_collaborative:
            endpoint = "/api/generate_collaborative"
            payload = {
                "input": prompt_or_history,
                "team_leader_model": self.config.get("team_leader_model"),
                "student_model": self.config.get("student_model"),
                "judge_model": self.config.get("judge_model"),
                "num_students": self.config.get("num_students"),
                "num_turns": self.config.get("num_turns"),
            }
        else:
            endpoint = "/api/generate"
            payload = {
                "input": prompt_or_history,
                "model": self.model_name,
                "ensemble_mode": self.config.get("ensemble_mode"),
                "judge_model": self.config.get("judge_model"),
            }

        return self._make_api_call(endpoint, payload)

    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        return self._prepare_and_call(prompt)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        messages = []
        for entry in history:
            role = "assistant" if entry.get("role") == "model" else "user"
            parts = entry.get("parts", [])
            content = " ".join([p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]) if isinstance(parts, list) else str(parts)
            if content:
                messages.append({"role": role, "content": content})

        if self.is_collaborative:
            last_user_prompt = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else "No user prompt."
            return self._prepare_and_call(last_user_prompt)

        return self._prepare_and_call(messages)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        logger.warning("Multimodal is not supported. Extracting text for generation.")
        text_prompt = " ".join([part.get("text", "") for item in contents for part in item.get("parts", []) if isinstance(part, dict) and "text" in part])
        return self.generate_action(text_prompt, **kwargs)

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        payload = {"input": prompt, "model": self.model_name}
        return self._make_api_call("/api/generate", payload)

# ---------------------------------------------------------------------------
#  Anthropic implementation
# ---------------------------------------------------------------------------
class AnthropicInterface(LLMInterface):
    def __init__(self, api_key: str, model_name: str, provider_name: str, thinking_enabled: bool = False, thinking_budget: int = 16000):
        super().__init__(api_key, model_name, provider_name)
        if not api_key:
            raise ValueError("Anthropic API Key is required.")
        if anthropic is None:
            raise ImportError("anthropic library is not installed. Please install it to use AnthropicInterface.")

        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget

        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model_name
            logger.info(f"AnthropicInterface initialised with model: {model_name}, thinking_enabled: {thinking_enabled}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"Anthropic initialization failed: {e}") from e

    def _call_anthropic_api(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        retries = 3
        delay = 10

        for attempt in range(retries):
            try:
                anthropic_kwargs = {
                    "model": self.model,
                    "max_tokens": 64000,  # default; caller can override
                    "messages": messages,
                }
                if system_prompt:
                    anthropic_kwargs["system"] = system_prompt

                # Only include non-None extra kwargs
                for key, value in kwargs.items():
                    if key not in ["model", "messages", "system"] and value is not None:
                        anthropic_kwargs[key] = value

                if self.thinking_enabled:
                    anthropic_kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
                else:
                    anthropic_kwargs["thinking"] = {"type": "disabled"}

                accumulated_text_parts: List[str] = []
                with self.client.messages.stream(**anthropic_kwargs) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            if getattr(event.delta, "type", "") == "text_delta":
                                accumulated_text_parts.append(getattr(event.delta, "text", ""))
                            elif getattr(event.delta, "type", "") == "thinking_delta":
                                # skip thinking deltas
                                pass

                final_message = stream.get_final_message()
                text_response = "".join(accumulated_text_parts).strip()

                if final_message and hasattr(final_message, 'usage'):
                    usage_data = final_message.usage
                    input_tokens = getattr(usage_data, 'input_tokens', None)
                    output_tokens = getattr(usage_data, 'output_tokens', None)
                    total_tokens = (input_tokens + output_tokens) if (input_tokens is not None and output_tokens is not None) else None
                    token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}

                raw_api_response_serialized = self._serialize_response(final_message)
                return text_response, token_usage, raw_api_response_serialized

            except anthropic.APIConnectionError as e:
                logger.warning(f"Anthropic API connection error (Attempt {attempt+1}/{retries}): {e}", exc_info=True)
            except anthropic.RateLimitError as e:
                logger.warning(f"Anthropic API rate limit error (Attempt {attempt+1}/{retries}): {e}", exc_info=True)
            except anthropic.APIStatusError as e:
                logger.warning(f"Anthropic API status error {e.status_code} (Attempt {attempt+1}/{retries}): {e.message}", exc_info=True)
            except Exception as e:
                logger.warning(f"Anthropic API error (Attempt {attempt+1}/{retries}): {e}", exc_info=True)

            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return None, None, None
        return None, None, None

    def _serialize_response(self, response: Any) -> Dict:
        try:
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            if hasattr(response, 'to_dict'):
                return response.to_dict()

            serializable_dict = {}
            for key in dir(response):
                if not key.startswith('_') and not callable(getattr(response, key)):
                    value = getattr(response, key)
                    if hasattr(value, 'model_dump'):
                        serializable_dict[key] = value.model_dump()
                    elif hasattr(value, 'to_dict'):
                        serializable_dict[key] = value.to_dict()
                    elif isinstance(value, (int, float, str, bool, type(None), list, dict)):
                        serializable_dict[key] = value
                    else:
                        serializable_dict[key] = str(value)
            return serializable_dict
        except Exception as e:
            logger.warning(f"Failed to serialize Anthropic response: {e}", exc_info=True)
            return {}

    def _convert_to_anthropic_messages(self, history: List[Dict]) -> List[Dict[str, Any]]:
        anthropic_messages = []
        for entry in history:
            gemini_role = entry.get("role", "user")
            parts = entry.get("parts", [])
            role = "assistant" if gemini_role == "model" else "user"

            text_parts = []
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, str):
                        text_parts.append(p)
                    elif isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                        text_parts.append(p["text"])
            elif isinstance(parts, str):
                text_parts.append(parts)
            content = "\n".join(text_parts).strip()

            if content:
                anthropic_messages.append({"role": role, "content": content})
        return anthropic_messages

    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_anthropic_api(messages, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        anthropic_messages = self._convert_to_anthropic_messages(history)
        if not anthropic_messages:
            return None, None, None
        return self._call_anthropic_api(anthropic_messages, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        # Convert simple text parts; skip images
        messages = []
        if not contents or not isinstance(contents, list) or not contents[0] or not isinstance(contents[0], dict):
            return None, None, None

        first_entry = contents[0]
        parts = first_entry.get("parts", [])
        text_content = ""
        for part in parts:
            if isinstance(part, str):
                text_content += part + "\n"
            elif isinstance(part, dict) and "text" in part:
                text_content += part["text"] + "\n"

        if text_content:
            messages.append({"role": "user", "content": text_content.strip()})
            return self._call_anthropic_api(messages, **kwargs)

        return None, None, None

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_anthropic_api(messages, **kwargs)

# ---------------------------------------------------------------------------
#  Kimi (Moonshot AI) implementation
# ---------------------------------------------------------------------------
class KimiInterface(LLMInterface):
    def __init__(self, api_key: str, model_name: str, provider_name: str):
        super().__init__(api_key, model_name, provider_name)
        if not api_key:
            raise ValueError("Kimi API Key is required for KimiInterface.")
        if OpenAI is None:
            raise ImportError("openai library is not installed. Please install it to use KimiInterface.")

        try:
            self.client = OpenAI(base_url="https://api.moonshot.ai/v1", api_key=api_key)
            self.model = model_name
            logger.info(f"KimiInterface initialised with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Kimi client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"Kimi initialization failed: {e}") from e

    def _call_kimi_api(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        resp: Optional[Any] = None
        try:
            # Strip None-valued kwargs
            safe_kwargs = {k: v for k, v in kwargs.items() if v is not None}

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=600,
                **safe_kwargs
            )
            if resp and getattr(resp, "choices", None):
                text_response = resp.choices[0].message.content

            if getattr(resp, "usage", None):
                usage = resp.usage
                token_usage = {
                    "input_tokens": getattr(usage, 'prompt_tokens', None),
                    "output_tokens": getattr(usage, 'completion_tokens', None),
                    "total_tokens": getattr(usage, 'total_tokens', None),
                }

            raw_api_response = self._serialize_response(resp)
            return text_response, token_usage, raw_api_response
        except Exception as e:
            logger.error(f"Kimi API error: {e}", exc_info=True)
            return None, None, None

    def _serialize_response(self, response: Any) -> Dict:
        try:
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to serialize Kimi response: {e}")
            return {}

    def _convert_to_kimi_messages(self, history: List[Dict]) -> List[Dict[str, Any]]:
        kimi_messages = []
        for entry in history:
            role = "assistant" if entry.get("role") == "model" else "user"
            parts = entry.get("parts", [])
            text_parts = []
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, str):
                        text_parts.append(p)
                    elif isinstance(p, dict) and "text" in p:
                        text_parts.append(p["text"])
            elif isinstance(parts, str):
                text_parts.append(parts)

            content = "\n".join(text_parts).strip()
            if content:
                kimi_messages.append({"role": role, "content": content})
        return kimi_messages

    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_kimi_api(messages, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        kimi_messages = self._convert_to_kimi_messages(history)
        if not kimi_messages:
            return None, None, None
        return self._call_kimi_api(kimi_messages, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        logger.warning("Multimodal generation is not currently supported by KimiInterface.")
        return None, None, None

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        return self.generate_action(prompt, **kwargs)