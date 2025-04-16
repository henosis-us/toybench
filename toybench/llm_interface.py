# llm_interface.py
"""
LLM interface abstractions for ToyBench.

* Google Gemini models via `google‑generativeai`
* OpenAI models via **both**
  • `/v1/chat/completions`  – used for true conversational history, and
  • `/v1/responses`         – kept for single‑shot prompts and the legacy
                              flattened‑string path.

The conversational path now sends an **array of role‑tagged messages**
(`user` / `assistant`) instead of concatenating everything into one big
string, fully matching the OpenAI chat‑completion spec.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
#  Base abstract interface                                                   #
# ---------------------------------------------------------------------------#
class LLMInterface(ABC):
    """Provider‑agnostic contract used throughout ToyBench."""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    # -------- text generation (stateless) ----------------------------------#
    @abstractmethod
    def generate_action(self, prompt: str) -> Optional[str]:
        """One‑off generation with a single prompt."""

    # -------- text generation (conversational) -----------------------------#
    @abstractmethod
    def generate_action_conversational(self, history: List[Dict]) -> Optional[str]:
        """
        Multi‑turn generation given a message history.

        `history` is a list like:
            [{"role": "user",  "parts": ["hi"]},
             {"role": "model", "parts": ["hello"]}]
        """

    # -------- multimodal generation ---------------------------------------#
    @abstractmethod
    def generate_content_multimodal(self, contents: List[Dict]) -> Optional[str]:
        """Image+text prompt where `contents` follows the Gemini style."""

    # -------- evaluation utility ------------------------------------------#
    @abstractmethod
    def evaluate_outcome(self, prompt: str) -> Optional[str]:
        """Let the same LLM act as an evaluator with a plain prompt."""


# ---------------------------------------------------------------------------#
#  Gemini implementation (unchanged from original ToyBench)                  #
# ---------------------------------------------------------------------------#
class GeminiInterface(LLMInterface):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        if not api_key:
            raise ValueError("Gemini API Key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info("GeminiInterface initialised with model: %s", model_name)

    # --- internal helper with retries -------------------------------------#
    def _call_api(
        self,
        prompt_or_contents,
        retries: int = 3,
        delay: int = 10,
    ) -> Optional[str]:
        is_list = isinstance(prompt_or_contents, list)
        for attempt in range(retries):
            try:
                resp = (
                    self.model.generate_content(prompt_or_contents)
                    if is_list
                    else self.model.generate_content(prompt_or_contents)
                )
                cand = resp.candidates[0]
                if cand and cand.content and cand.content.parts:
                    return cand.content.parts[0].text
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Gemini API error (%s/%s): %s", attempt + 1, retries, e, exc_info=True
                )
            time.sleep(delay * (attempt + 1))
        logger.error("Gemini API failed after %s attempts", retries)
        return None

    # --- public wrappers ---------------------------------------------------#
    def generate_action(self, prompt: str) -> Optional[str]:
        return self._call_api(prompt)

    def generate_action_conversational(self, history: List[Dict]) -> Optional[str]:
        return self._call_api(history)

    def generate_content_multimodal(self, contents: List[Dict]) -> Optional[str]:
        return self._call_api(contents)

    def evaluate_outcome(self, prompt: str) -> Optional[str]:
        return self._call_api(prompt)


# ---------------------------------------------------------------------------#
#  OpenAI implementation                                                     #
# ---------------------------------------------------------------------------#
class OpenAIInterface(LLMInterface):
    """Uses `/v1/chat/completions` for chat; `/v1/responses` for single‑shot."""

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        if not api_key:
            raise ValueError("OpenAI API Key is required.")
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        logger.info("OpenAIInterface initialised with model: %s", model_name)

    # ---------------- helpers ---------------------------------------------#
    @staticmethod
    def _as_dict(obj: Any) -> Any:
        """Convert pydantic BaseModel → dict with graceful fallback."""
        if obj is None:
            return obj
        if hasattr(obj, "model_dump"):
            d = obj.model_dump()
            if d:
                return d
        if hasattr(obj, "dict"):
            d = obj.dict()
            if d:
                return d
        return getattr(obj, "__dict__", obj)

    # ------------- /v1/chat/completions (true chat) ------------------------#
    def _call_chat_api(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("OpenAI chat.completions error: %s", e, exc_info=True)
            return None

        if not resp.choices:
            logger.warning("chat.completions returned no choices")
            return None
        msg = resp.choices[0].message
        return (
            msg.content
            if hasattr(msg, "content")
            else self._as_dict(msg).get("content")
        )

    # ------------- /v1/responses (single‑shot) -----------------------------#
    def _call_responses_api(
        self,
        text_input: str,
        *,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Optional[str]:
        body: Dict[str, Any] = {
            "model": self.model,
            "input": text_input,
            "tool_choice": "none",  # disable auto‑tools
        }
        if instructions:
            body["instructions"] = instructions
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        try:
            resp = self.client.responses.create(**body)
        except Exception as e:  # noqa: BLE001
            logger.error("OpenAI /responses API error: %s", e, exc_info=True)
            return None

        output_raw = getattr(resp, "output", None)
        if not output_raw:
            logger.warning("/responses returned empty output list")
            return None

        first_msg = next(
            (item for item in output_raw if getattr(item, "type", None) == "message"),
            None,
        )
        if first_msg is None:
            logger.warning("/responses: no assistant message item found")
            return None

        content_raw = getattr(first_msg, "content", None) or self._as_dict(
            first_msg
        ).get("content")
        if not content_raw:
            logger.warning("Assistant message had empty content list")
            return None

        part0 = content_raw[0]
        return (
            part0.text
            if hasattr(part0, "text")
            else self._as_dict(part0).get("text")
        )

    # ---------------- public wrappers -------------------------------------#
    def generate_action(self, prompt: str) -> Optional[str]:
        return self._call_responses_api(prompt)

    def generate_action_conversational(self, history: List[Dict]) -> Optional[str]:
        """
        Convert ToyBench history to OpenAI chat format.

        ToyBench stores roles as `'user'` and `'model'`; OpenAI expects
        `'assistant'` for the second one.
        """
        allowed_roles = {
            "system",
            "user",
            "assistant",
            "tool",
            "function",
            "developer",
        }
        messages: List[Dict[str, str]] = []
        for entry in history:
            raw_role = entry.get("role", "user")
            role = "assistant" if raw_role == "model" else raw_role
            if role not in allowed_roles:
                role = "user"
            parts = entry.get("parts", [])
            text = "".join(
                p if isinstance(p, str) else p.get("text", "") for p in parts
            )
            messages.append({"role": role, "content": text})
        return self._call_chat_api(messages)

    def generate_content_multimodal(self, contents: List[Dict]) -> Optional[str]:
        # ToyBench still sends flattened text for multimodal prompts.
        texts: List[str] = []
        for msg in contents:
            for part in msg.get("parts", []):
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
        return self._call_responses_api("\n".join(texts))

    def evaluate_outcome(self, prompt: str) -> Optional[str]:
        return self._call_responses_api(prompt)
