# llm_interface.py
"""LLM interface abstractions for ToyBench.

* Google Gemini models via `google-genai`
* OpenAI models via both
    * `/v1/chat/completions` – used for true conversational history, and
    * `/v1/responses`         – kept for single-shot prompts and the legacy
                              flattened-string path.
* xAI Grok models via `openai` with custom base URL.
* Quality Compute Simulator – added for integration with the custom backend simulator.
* Anthropic Claude models via `anthropic`

The conversational path now sends an array of role-tagged messages
(`user` / `assistant`) instead of concatenating everything into one big
string, fully matching the OpenAI chat-completion spec.

MODIFIED TO EXTRACT AND RETURN TOKEN USAGE DATA AND RAW API RESPONSE OBJECT.
CORRECTED TYPE HINTS FOR STATIC ANALYSIS USING TYPE_CHECKING.
MODIFIED GeminiInterface to improve text extraction and logging on failure,
handling `RepeatedComposite` types, and to support thinkingBudget extraction.
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

"""  # Updated for Google GenAI SDK migration, image handling fix, and additional logging

from __future__ import annotations

import logging
import time
import json
import base64
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Import heavy dependencies only for type checking
if TYPE_CHECKING:
    from google.genai.types import (  # Updated to google.genai.types
        Candidate,
        GenerateContentResponse,
        PromptFeedback,
        UsageMetadata,
        HttpOptions,  # Import HttpOptions for type checking
    )
    from google.genai import types as genai_types  # Alias for types module
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice as OpenAIChoice
    from openai.types.completion_usage import CompletionUsage
    from anthropic.types import Message

# Keep runtime imports minimal or handle ImportErrors if necessary
try:
    from google import genai
    from google.genai import types as genai_types  # Correct runtime import for new SDK
except ImportError:
    genai = None
    genai_types = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Handle missing library gracefully

try:
    import anthropic
except ImportError:
    anthropic = None  # Handle missing library gracefully

import requests  # For Quality Compute API calls

logger = logging.getLogger(__name__)

# Define type aliases for clarity
TokenUsage = Optional[Dict[str, int | None]]
RawAPIResponse = Optional[Any]  # Use Any for the raw response object type, but aim for serializability
LLMResponse = Tuple[Optional[str], TokenUsage, RawAPIResponse]  # MODIFIED: Added RawAPIResponse

# ---------------------------------------------------------------------------
#  Base abstract interface
# ---------------------------------------------------------------------------
class LLMInterface(ABC):
    """Provider-agnostic contract used throughout ToyBench."""
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    # -------- text generation (stateless) ----------------------------------
    @abstractmethod
    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:  # Added **kwargs for flexibility
        """One-off generation with a single prompt.

        Returns:
            LLMResponse: (text_response, token_usage_dict, raw_api_response)
        """
        pass

    # -------- text generation (conversational) -----------------------------
    @abstractmethod
    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:  # Added **kwargs
        """Multi-turn generation given a message history.

        `history` is a list like:
            [{"role": "user",  "parts": ["hi"]},
             {"role": "model", "parts": ["hello"]}]

        Returns:
            LLMResponse: (text_response, token_usage_dict, raw_api_response)
        """
        pass

    # -------- multimodal generation ---------------------------------------
    @abstractmethod
    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:  # Added **kwargs
        """Image+text prompt where `contents` follows the Gemini style.

        Returns:
            LLMResponse: (text_response, token_usage_dict, raw_api_response)
        """
        pass

    # -------- evaluation utility ------------------------------------------
    @abstractmethod
    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:  # Added **kwargs
        """Let the same LLM act as an evaluator with a plain prompt.

        Returns:
            LLMResponse: (text_response, token_usage_dict, raw_api_response)
        """
        pass

# ---------------------------------------------------------------------------
#  Gemini implementation (updated for Google GenAI SDK and image handling)
# ---------------------------------------------------------------------------
class GeminiInterface(LLMInterface):
    def __init__(self, api_key: str, model_name: str, thinking_enabled: bool = True, thinking_budget: Optional[int] = None):
        super().__init__(api_key, model_name)
        if not api_key:
            raise ValueError("Gemini API Key is required.")
        if genai is None or genai_types is None:
            raise ImportError("google-genai library is not installed or types module is missing. Please install it to use GeminiInterface.")
        try:
            # Set HTTP options at client level for timeout
            http_options = genai_types.HttpOptions(timeout=1000000)  # 1000 seconds in milliseconds
            self.client = genai.Client(api_key=api_key, http_options=http_options)  # Set timeout here
            self.thinking_enabled = thinking_enabled
            self.thinking_budget = thinking_budget
            self._validate_model_and_thinking_budget()  # Validate model and thinking budget on init
            logger.info("GeminiInterface initialised with model: %s, thinking_enabled: %s, thinking_budget: %s", model_name, thinking_enabled, thinking_budget if thinking_budget is not None else "auto")
        except Exception as e:
            logger.error(f"Failed to configure Gemini or initialize client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"Gemini initialization failed: {e}") from e

    def _validate_model_and_thinking_budget(self):
        """Validate the model name and thinking budget based on Google AI documentation."""
        model_lower = self.model_name.lower()
        # Assume if "preview" is in the name, it might support thinking, but validate specifically
        is_flash_2_5 = "2.5-flash" in model_lower
        is_pro_2_5 = "2.5-pro" in model_lower or "2.5-pro-preview" in model_lower

        if is_pro_2_5:
            # Gemini 2.5 Pro: thinkingBudget must be 128-32768, cannot disable thinking
            if self.thinking_budget is not None:
                if not (128 <= self.thinking_budget <= 32768):
                    logger.warning(f"Invalid thinking_budget for Gemini 2.5 Pro model ({self.model_name}). Must be between 128 and 32768. Setting to default auto behavior.")
                    self.thinking_budget = None  # Reset to auto
                    self.thinking_enabled = True  # Pro models always have thinking, lowest is 128
                elif self.thinking_budget < 128:
                    logger.warning(f"thinking_budget {self.thinking_budget} too low for Gemini 2.5 Pro. Minimum is 128. Setting to default auto behavior.")
                    self.thinking_budget = None
                    self.thinking_enabled = True
            else:
                logger.debug(f"thinking_budget not set for Gemini 2.5 Pro ({self.model_name}); model will dynamically adjust.")
                self.thinking_enabled = True  # Pro models always have thinking enabled

        elif is_flash_2_5:
            # Gemini 2.5 Flash: thinkingBudget can be 0-24576, 0 disables thinking
            if self.thinking_budget is not None:
                if not (0 <= self.thinking_budget <= 24576):
                    logger.warning(f"Invalid thinking_budget for Gemini 2.5 Flash model ({self.model_name}). Must be between 0 and 24576. Setting to default auto behavior.")
                    self.thinking_budget = None  # Reset to auto
                    self.thinking_enabled = True  # Allow auto if budget invalid
                elif self.thinking_budget == 0:
                    logger.debug(f"thinking_budget is 0 for Gemini 2.5 Flash ({self.model_name}). Thinking will be disabled.")
                    self.thinking_enabled = False  # Explicitly disable if budget is 0
                else:
                    self.thinking_enabled = True  # Enable if non-zero valid budget
            else:
                logger.debug(f"thinking_budget not set for Gemini 2.5 Flash ({self.model_name}); model will dynamically adjust.")
                self.thinking_enabled = True  # Default to enabled for auto budget if not explicitly set to 0

        else:
            # For other models, thinkingBudget is not supported
            if self.thinking_budget is not None or self.thinking_enabled:
                logger.warning(f"Model '{self.model_name}' does not explicitly support thinkingBudget. Ignoring thinking_enabled and thinking_budget settings.")
            self.thinking_enabled = False  # Ensure thinking is off for unsupported models
            self.thinking_budget = None

    # --- internal helper with retries -------------------------------------
    def _call_api(
        self,
        prompt_or_contents: Any,  # Can be str, list[dict], or other types
        retries: int = 3,
        delay: int = 10,
    ) -> LLMResponse:
        resp: Optional["GenerateContentResponse"] = None
        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        raw_api_response_for_return = None

        # Convert prompt_or_contents to a list of genai_types.Content objects with proper Part handling
        contents_to_use = []
        if isinstance(prompt_or_contents, str):
            # For single-string prompts, create a user content object with a Part
            contents_to_use.append(genai_types.Content(role="user", parts=[genai_types.Part(text=prompt_or_contents)]))
        elif isinstance(prompt_or_contents, list):
            # For lists (conversational or multimodal), convert each item to a Content object
            for item in prompt_or_contents:
                if isinstance(item, dict) and 'role' in item and 'parts' in item:
                    parts_list = []
                    for part in item.get('parts', []):
                        if isinstance(part, str):
                            parts_list.append(genai_types.Part(text=part))
                        elif isinstance(part, dict):
                            if "text" in part:
                                parts_list.append(genai_types.Part(text=part["text"]))
                            elif "inline_data" in part:  # Corrected condition to match observed data structure
                                inline_data_dict = part["inline_data"]  # Direct access to inline_data
                                mime_type = inline_data_dict.get("mime_type")
                                data = inline_data_dict.get("data")
                                if mime_type and data:
                                    try:
                                        # Decode base64 data to bytes if it's a string
                                        if isinstance(data, str):
                                            data_bytes = base64.b64decode(data)  # Decode base64 string to bytes
                                        elif isinstance(data, bytes):
                                            data_bytes = data  # Already bytes, no need to decode
                                        else:
                                            raise ValueError(f"Unsupported data type for inline_data: {type(data)}")
                                        # Now create Part from data bytes
                                        parts_list.append(genai_types.Part.from_bytes(mime_type=mime_type, data=data_bytes))
                                        logger.debug(f"Successfully created Part from decoded data with mime_type: {mime_type}")
                                    except base64.binascii.Error as decode_e:
                                        logger.error(f"Base64 decoding error for part data: {decode_e}. Part data: {part}", exc_info=True)
                                    except Exception as e:
                                        logger.error(f"Failed to create Part from data: {e}. Part data: {part}", exc_info=True)
                                else:
                                    logger.warning(f"Malformed inline_data part: {part}. Skipping.")
                            else:
                                logger.warning(f"Unsupported part dict structure: {part}. Skipping part. Debug: keys={list(part.keys())}")
                        else:
                            logger.warning(f"Unsupported part type in history: {type(part)} / {part}. Skipping part. Debug: part={part}")
                    if parts_list:
                        contents_to_use.append(genai_types.Content(role=item['role'], parts=parts_list))
                else:
                    logger.warning(f"Skipping invalid item in contents: {item}. Expected dict with 'role' and 'parts'.")

        # Add enhanced logging for contents_to_use before API call with null checks
        logger.debug(f"API call contents_to_use summary: {len(contents_to_use)} content items.")
        for idx, content in enumerate(contents_to_use):
            role = content.role
            parts = content.parts  # parts could be None, but genai_types.Content should ensure it's a list
            parts_count = len(parts) if parts is not None else 0  # Safe count with null check
            logger.debug(f"Content {idx} role: {role}, number of parts: {parts_count}")
            if parts is not None:  # Only iterate if parts is not None
                for part_idx, part in enumerate(parts):
                    if hasattr(part, 'text'):
                        part_text = getattr(part, 'text', None)
                        if part_text is not None:  # Check for None before accessing length or slicing
                            text_preview = (part_text[:50] + '...') if len(part_text) > 50 else part_text
                            logger.debug(f"  Part {part_idx} is text (preview): {text_preview}")
                        else:
                            logger.debug(f"  Part {part_idx} has 'text' attribute but it is None. Skipping preview log.")
                    elif hasattr(part, 'inline_data'):
                        inline_data = getattr(part, 'inline_data', None)
                        if inline_data is not None:
                            mime_type = inline_data.mime_type if hasattr(inline_data, 'mime_type') else 'N/A'
                            data_preview = (inline_data.data[:20] + '...') if inline_data.data and len(inline_data.data) > 20 else str(inline_data.data)
                            logger.debug(f"  Part {part_idx} is inline_data, mime_type: {mime_type}, data preview: {data_preview}")
                        else:
                            logger.debug(f"  Part {part_idx} has 'inline_data' attribute but it is None.")
                    else:
                        logger.debug(f"  Part {part_idx} is of unknown type: {type(part)}")
            else:
                logger.debug(f"Content {idx} has no parts or parts is None.")

        for attempt in range(retries):
            try:
                logger.debug(f"Calling Gemini API (Attempt {attempt+1}/{retries}) with thinking_enabled: {self.thinking_enabled}, thinking_budget: {self.thinking_budget}")

                # Prepare generation config with thinking config if enabled
                config_obj = None
                if self.thinking_enabled and genai_types is not None:
                    thinking_config_params = {}
                    if self.thinking_budget is not None:
                        thinking_config_params["thinking_budget"] = self.thinking_budget
                    thinking_config_obj = genai_types.ThinkingConfig(**thinking_config_params)
                    config_obj = genai_types.GenerateContentConfig(thinking_config=thinking_config_obj)
                    logger.debug("Successfully created GenerateContentConfig with ThinkingConfig.")

                # Call generate_content with the converted contents
                resp = self.client.models.generate_content(  # Updated to client.models.generate_content
                    model=self.model_name,
                    contents=contents_to_use,  # Use the converted list of Content objects
                    config=config_obj,  # Renamed from generation_config to config
                )
                raw_api_response_for_return = self._serialize_response(resp)

                # Log raw response type for debugging serialization issues
                logger.debug(f"Raw API Response type: {type(resp)}, serialized version (truncated): {str(raw_api_response_for_return)[:500]}...")

                # --- Enhanced Text Extraction Logic ---
                text_response = None
                finish_reason_log = "N/A"
                safety_ratings_log = "N/A"
                prompt_feedback_log = "N/A"

                # Check prompt feedback first
                prompt_feedback: Optional["PromptFeedback"] = getattr(resp, 'prompt_feedback', None)
                if prompt_feedback:
                    prompt_feedback_log = str(prompt_feedback)
                    logger.debug(f"Gemini Prompt Feedback: {prompt_feedback_log}")
                    if hasattr(prompt_feedback, 'block_reason') and getattr(prompt_feedback, 'block_reason', None):
                        block_reason_value = getattr(prompt_feedback, 'block_reason', 'Unknown')
                        block_reason_str = str(block_reason_value) if block_reason_value else 'Unknown'
                        logger.warning(f"Gemini prompt was blocked. Reason: {block_reason_str}. No candidates expected.")
                    else:
                        logger.debug("No prompt blocking reason found.")
                else:
                    logger.debug("No prompt_feedback found in Gemini response.")

                # Now check candidates
                candidates_list = getattr(resp, 'candidates', None)
                if candidates_list is not None and hasattr(candidates_list, '__len__') and len(candidates_list) > 0:
                    logger.debug(f"Found {len(candidates_list)} candidate(s). Processing candidate 0. Type: {type(candidates_list)}")

                    first_candidate: Optional["Candidate"] = None
                    try:
                        first_candidate = candidates_list[0]
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Failed to access candidates_list[0]: {e}")
                        first_candidate = None

                    if first_candidate:
                        finish_reason_value = getattr(first_candidate, 'finish_reason', None)
                        finish_reason_log = str(finish_reason_value) if finish_reason_value else 'N/A'

                        safety_ratings_value = getattr(first_candidate, 'safety_ratings', None)
                        safety_ratings_log = str(safety_ratings_value) if safety_ratings_value else 'N/A'

                        logger.debug(f"Gemini Candidate 0: Finish Reason='{finish_reason_log}', Safety Ratings='{safety_ratings_log}'")

                        if finish_reason_log not in ['STOP', 'MAX_TOKENS']:
                            logger.warning(f"Gemini candidate 0 did not finish normally (Reason: {finish_reason_log}). Text content might be missing or incomplete.")

                        content_obj = getattr(first_candidate, 'content', None)
                        if content_obj and hasattr(content_obj, 'parts'):
                            parts_list_resp = getattr(content_obj, 'parts', None)
                            if parts_list_resp is not None and hasattr(parts_list_resp, '__iter__'):
                                parts_count = len(parts_list_resp) if hasattr(parts_list_resp, '__len__') else 'unknown'
                                logger.debug(f"Candidate 0 content has {parts_count} part(s). Type: {type(parts_list_resp)}")
                                for part in parts_list_resp:
                                    if hasattr(part, 'text'):
                                        part_text = getattr(part, 'text', None)
                                        if isinstance(part_text, str) and part_text:
                                            text_response = part_text
                                            logger.debug("Successfully extracted non-empty text from a part in candidate 0.")
                                            break  # Found text, stop looking
                                        elif part_text is not None:
                                            logger.debug(f"Found part with 'text' attribute, but text is empty or not a string: {type(part_text)}")
                            elif parts_list_resp is not None:
                                logger.warning(f"Candidate 0 content.parts is not iterable. Type: {type(parts_list_resp)}")
                            else:
                                logger.warning("Candidate 0 content.parts is None.")
                        elif content_obj:
                            logger.warning("Candidate 0 has 'content' but no 'parts' attribute.")
                        else:
                            logger.warning("Candidate 0 has no 'content' attribute or it's empty.")
                elif candidates_list is None:
                    logger.warning("Gemini response has no 'candidates' attribute.")
                elif not hasattr(candidates_list, '__len__') or len(candidates_list) == 0:
                    if prompt_feedback and hasattr(prompt_feedback, 'block_reason') and getattr(prompt_feedback, 'block_reason', None):
                        logger.warning("Gemini response candidates list is empty, likely due to prompt blocking (see prompt feedback log).")
                    else:
                        logger.warning("Gemini response candidates list is empty.")

                # Extract Token Usage, including thoughts_token_count if available
                token_usage = None
                usage_meta: Optional["UsageMetadata"] = getattr(resp, 'usage_metadata', None)
                if usage_meta:
                    prompt_tokens = getattr(usage_meta, 'prompt_token_count', None)
                    candidates_tokens = getattr(usage_meta, 'candidates_token_count', None)
                    thoughts_tokens = getattr(usage_meta, 'thoughts_token_count', None)  # Extract thoughts_token_count for thinking budget
                    total_tokens = getattr(usage_meta, 'total_token_count', None)
                    
                    logger.debug(f"Raw usage_metadata attributes: prompt={prompt_tokens}, candidates={candidates_tokens}, thoughts={thoughts_tokens}, total={total_tokens}")

                    if prompt_tokens is not None or candidates_tokens is not None or total_tokens is not None or thoughts_tokens is not None:
                        token_usage = {
                            "input_tokens": prompt_tokens,
                            "output_tokens": candidates_tokens,
                            "reasoning_tokens": thoughts_tokens,  # Map to reasoning_tokens for consistency
                            "total_tokens": total_tokens
                        }
                        # Remove None values and log for debugging
                        token_usage = {k: v for k, v in token_usage.items() if v is not None}
                        logger.debug(f"Gemini API Token Usage (extracted and filtered): {token_usage}")
                else:
                    logger.debug("No usage_metadata found in Gemini response.")

                if text_response is None:
                    logger.warning(f"Failed to extract text response from Gemini API object. Finish Reason: {finish_reason_log}. Prompt Feedback: {prompt_feedback_log}. Returning None for text.")
                else:
                    logger.debug("Text response successfully extracted.")

                return text_response, token_usage, raw_api_response_for_return

            except Exception as e:
                # Enhanced logging for error diagnosis, including full input data if possible
                error_message = str(e)
                logger.error(f"Gemini API error during call/processing (Attempt {attempt+1}/{retries}): {error_message}", exc_info=True)
                logger.error(f"Error details: Status code may indicate invalid input. Input data (full): {json.dumps(prompt_or_contents, default=str, ensure_ascii=False)}")
                raw_api_response_for_return = None  # Ensure this is set even in error cases
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    logger.error("Gemini API failed after %s attempts. Check input data for validity, especially image parts.", retries)
                    return None, None, None

        logger.error("Gemini API call failed after all retries or encountered unhandled issue.")
        return None, None, None

    def _serialize_response(self, response: Any) -> Dict:
        """Serialize the API response to a dictionary for JSON compatibility."""
        try:
            # Check if response has a model_dump (Pydantic v2) or to_dict (older Pydantic/dataclasses_json)
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            elif hasattr(response, 'to_dict'):
                return response.to_dict()
            # If it's a google.genai.types.GenerateContentResponse, attempt serialization
            if isinstance(response, (dict, list)):  # Already a dict or list, no need to convert
                return response
            
            # Attempt to convert to dict via json.loads(json.dumps) for deeply nested types
            try:
                return json.loads(json.dumps(response.__dict__))
            except (TypeError, AttributeError):
                # Fallback to direct attribute access if json.dumps fails
                serializable_dict = {}
                for key in dir(response):
                    if not key.startswith('_') and not callable(getattr(response, key)):
                        value = getattr(response, key)
                        # Attempt to serialize nested objects recursively, or convert to string
                        if hasattr(value, 'model_dump'):
                            serializable_dict[key] = value.model_dump()
                        elif hasattr(value, 'to_dict'):
                            serializable_dict[key] = value.to_dict()
                        elif isinstance(value, (int, float, str, bool, type(None))):
                            serializable_dict[key] = value
                        elif isinstance(value, (list, dict)):
                            # Recursive serialization for lists/dicts containing non-serializable items
                            serializable_dict[key] = json.loads(json.dumps(value, default=str))
                        else:
                            # Convert complex objects like enums, RepeatedComposite to string
                            serializable_dict[key] = str(value)
                return serializable_dict

        except Exception as e:
            logger.warning(f"Failed to serialize response object: {e}. Returning empty dict for safety.", exc_info=True)
            return {}  # Return an empty dict to avoid serialization errors

    # --- public wrappers ---
    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        return self._call_api(prompt)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        return self._call_api(history)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        return self._call_api(contents)

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        return self._call_api(prompt)

# ---------------------------------------------------------------------------
#  OpenAI implementation
# ---------------------------------------------------------------------------
class OpenAIInterface(LLMInterface):
    """Uses `/v1/chat/completions` for chat; `/v1/responses` for single-shot."""
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        if not api_key:
            raise ValueError("OpenAI API Key is required.")
        if OpenAI is None:
            raise ImportError("openai library is not installed. Please install it to use OpenAIInterface.")
        try:
            self.client = OpenAI(api_key=api_key)
            self.model = model_name
            logger.info("OpenAIInterface initialised with model: %s", model_name)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"OpenAI initialization failed: {e}") from e

    # ---------------- helpers ---------------------------------------------
    @staticmethod
    def _as_dict(obj: Any) -> Any:
        """Convert pydantic BaseModel -> dict with graceful fallback."""
        if obj is None: return obj
        if hasattr(obj, "model_dump"):
            try: d = obj.model_dump(); return d if d is not None else obj
            except Exception: pass
        if hasattr(obj, "dict"):
            try: d = obj.dict(); return d if d is not None else obj
            except Exception: pass
        if hasattr(obj, "__dict__"):
             d = getattr(obj, "__dict__"); return d if d is not None else obj
        return obj

    # ------------- /v1/chat/completions (true chat) ------------------------
    def _call_chat_api(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:  # Added **kwargs to handle extra params like max_tokens
        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        resp: Optional["ChatCompletion"] = None
        try:
            logger.debug("Calling OpenAI Chat API (chat.completions.create) with additional kwargs: %s", kwargs)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs  # Pass any additional parameters like max_tokens, temperature
            )
            # Extract text response safely
            choice_list: Optional[List["OpenAIChoice"]] = getattr(resp, 'choices', None)
            if choice_list is not None and hasattr(choice_list, '__len__') and len(choice_list) > 0:
                first_choice: Optional["OpenAIChoice"] = None
                try:
                    first_choice = choice_list[0]
                except (IndexError, TypeError):
                    logger.warning("Failed to access choice_list[0].")
                    first_choice = None

                if first_choice and first_choice.message:
                    text_response = first_choice.message.content
                    finish_reason = getattr(first_choice, 'finish_reason', 'N/A')
                    logger.debug(f"OpenAI Chat extracted text. Finish reason: {finish_reason}")
                elif first_choice is not None:
                    logger.warning("OpenAI Chat choices[0].message is None.")
            elif choice_list is None:
                logger.warning("OpenAI Chat response has no 'choices' attribute.")
            elif not hasattr(choice_list, '__len__') or len(choice_list) == 0:
                logger.warning("OpenAI Chat response 'choices' list is empty.")

            # Extract token usage safely
            usage_data: Optional["CompletionUsage"] = getattr(resp, 'usage', None)
            if usage_data:
                prompt_tokens = getattr(usage_data, 'prompt_tokens', None)
                completion_tokens = getattr(usage_data, 'completion_tokens', None)
                total_tokens = getattr(usage_data, 'total_tokens', None)
                if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
                    token_usage = {"input_tokens": prompt_tokens, "output_tokens": completion_tokens, "total_tokens": total_tokens}
                    logger.debug(f"OpenAI Chat API Token Usage: {token_usage}")
            else:
                logger.debug("No usage data found in OpenAI Chat response.")

            # Serialize raw response for JSON compatibility
            raw_api_response_serialized = self._serialize_response(resp)
            logger.debug(f"Serialized OpenAI Raw API Response: {raw_api_response_serialized}")

            if text_response is None:
                logger.warning("Failed to extract text response from OpenAI Chat API object. Returning None for text.")
            return text_response, token_usage, raw_api_response_serialized

        except Exception as e:
            logger.error("OpenAI chat.completions error: %s", e, exc_info=True)
            return None, None, None

    def _serialize_response(self, response: Any) -> Dict:
        """Serialize the API response to a dictionary for JSON compatibility."""
        try:
            # Attempt to use built-in serialization
            if hasattr(response, 'model_dump'):
                return response.model_dump()  # For pydantic models (like OpenAI's client library uses)
            elif hasattr(response, 'to_dict'):
                return response.to_dict()  # If the response has a to_dict method
            else:
                # Fallback to direct attribute access (less robust for nested objects)
                serializable_dict = {}
                for key in dir(response):
                    if not key.startswith('_') and not callable(getattr(response, key)):
                        value = getattr(response, key)
                        # Attempt to serialize nested objects or convert to string
                        if hasattr(value, 'model_dump'):
                            serializable_dict[key] = value.model_dump()
                        elif hasattr(value, 'to_dict'):
                            serializable_dict[key] = value.to_dict()
                        elif isinstance(value, (int, float, str, bool, type(None), list, dict)):
                            serializable_dict[key] = value
                        else:
                            serializable_dict[key] = str(value)  # Convert other types to string
                return serializable_dict
        except Exception as e:
            logger.warning(f"Failed to serialize OpenAI response object: {e}. Returning empty dict.", exc_info=True)
            return {}  # Return empty dict to avoid crashing serialization

    # ------------- /v1/responses (single-shot) -----------------------------
    def _call_responses_api(
        self,
        text_input: str,
        *,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> LLMResponse:
        logger.warning("Using deprecated OpenAI /v1/responses endpoint. Consider switching to /v1/chat/completions.")
        body: Dict[str, Any] = {"model": self.model, "input": text_input, "tool_choice": "none"}
        if instructions: body["instructions"] = instructions
        if max_output_tokens is not None: body["max_output_tokens"] = max_output_tokens

        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        resp: Optional[Any] = None

        try:
            logger.debug("Calling OpenAI Responses API (responses.create)")
            resp = self.client.responses.create(**body)

            # Extract text response
            output_raw = getattr(resp, "output", None)
            if output_raw and hasattr(output_raw, '__iter__'):
                first_msg = next((item for item in output_raw if getattr(item, "type", None) == "message"), None)
                if first_msg and hasattr(first_msg, 'content'):
                    content_raw = getattr(first_msg, "content", None)
                    if content_raw and hasattr(content_raw, '__iter__') and len(list(content_raw)) > 0:
                        text_part = next((p for p in content_raw if hasattr(p, 'text')), None)
                        if text_part and hasattr(text_part, 'text'):
                            part_text = getattr(text_part, 'text', None)
                            if isinstance(part_text, str) and part_text:
                                text_response = part_text
                                logger.debug("OpenAI Responses API extracted text.")
                    elif content_raw:
                        logger.warning(f"OpenAI Responses API: first_msg.content is empty or not iterable. Type: {type(content_raw)}")
                    else:
                        logger.warning("OpenAI Responses API: first_msg has no 'content'.")
                elif first_msg:
                    logger.warning("OpenAI Responses API: first_msg exists but no 'content' attribute.")
                else:
                    logger.warning("OpenAI Responses API: No item with type 'message' found in output list.")
            elif output_raw is not None:
                logger.warning(f"OpenAI Responses API: Response 'output' is not iterable. Type: {type(output_raw)}")
            else:
                logger.warning("OpenAI Responses API: Response has no 'output' attribute.")

            # Extract token usage
            usage_attr = getattr(resp, 'usage', None)
            if usage_attr:
                usage_data = self._as_dict(usage_attr)
                if isinstance(usage_data, dict):
                    input_t = usage_data.get("input_tokens")
                    output_t = usage_data.get("output_tokens")
                    total_t = usage_data.get("total_tokens")
                    input_details = self._as_dict(usage_data.get("input_tokens_details", {}))
                    output_details = self._as_dict(usage_data.get("output_tokens_details", {}))
                    cached_input = input_details.get("cached_tokens") if isinstance(input_details, dict) else None
                    reasoning_output = output_details.get("reasoning_tokens") if isinstance(output_details, dict) else None
                    if input_t is not None or output_t is not None or total_t is not None:
                        temp_token_usage = {"input_tokens": input_t, "output_tokens": output_t, "total_tokens": total_t}
                        if cached_input is not None: temp_token_usage["cached_input_tokens"] = cached_input
                        if reasoning_output is not None: temp_token_usage["reasoning_output_tokens"] = reasoning_output
                        token_usage = {k: v for k, v in temp_token_usage.items() if v is not None} or None
                        if token_usage: logger.debug(f"OpenAI Responses API Token Usage: {token_usage}")
            else:
                logger.debug("No usage data found in OpenAI Responses response.")

            if not text_response: logger.warning("/responses API call did not yield a parsable text response.")
            if not token_usage: logger.warning("/responses API call did not yield parsable token usage.")

            # Serialize raw response for JSON compatibility
            raw_api_response_serialized = self._serialize_response(resp)
            logger.debug(f"Serialized OpenAI Raw API Response: {raw_api_response_serialized}")

            return text_response, token_usage, raw_api_response_serialized

        except Exception as e:
            logger.error("OpenAI /responses API error: %s", e, exc_info=True)
            return None, None, None

    # ---------------- public wrappers -------------------------------------
    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        # Prefer chat completions even for single turns
        logger.debug("Using chat API for single-shot generate_action request.")
        messages = [{"role": "user", "content": prompt}]
        return self._call_chat_api(messages, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        # Convert Gemini format history (list of dicts with 'role', 'parts')
        # to OpenAI messages format (list of dicts with 'role', 'content').
        openai_messages: List[Dict[str, Any]] = []
        allowed_openai_roles = {"system", "user", "assistant"}
        last_openai_role = None

        for entry in history:
            gemini_role = entry.get("role", "user")
            parts = entry.get("parts", [])

            openai_role = "assistant" if gemini_role == "model" else "user"
            if openai_role not in allowed_openai_roles:
                logger.warning(f"Mapping unexpected Gemini role '{gemini_role}' to 'user'.")
                openai_role = "user"

            content_value: Any = None

            if isinstance(parts, list):
                text_content_parts: List[str] = []
                for p in parts:
                    if isinstance(p, str):
                        text_content_parts.append(p)
                    elif isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                        text_content_parts.append(p["text"])
                    elif isinstance(p, dict) and "source" in p and "inline_data" in p.get("source", {}):
                        mime = p["source"]["inline_data"].get("mime_type", "unk")
                        logger.debug(f"Ignoring image ({mime}) part in text conversational history conversion for OpenAI.")
                    else:
                        logger.warning(f"Skipping unsupported part type in history: {type(p)} / {p!r}")
                content_value = "\n".join(text_content_parts).strip()
            elif isinstance(parts, str):
                content_value = parts.strip()
            else:
                logger.warning(f"History entry 'parts' has unexpected type {type(parts)}. Skipping entry.")
                continue

            if not content_value:
                logger.debug(f"Skipping history entry from role '{gemini_role}' with empty content after processing parts.")
                continue

            if openai_messages and last_openai_role == openai_role:
                logger.warning(f"Consecutive messages with role '{openai_role}' detected. Appending content to the last message.")
                last_message = openai_messages[-1]
                if isinstance(last_message.get('content'), str):
                    last_message['content'] += "\n" + str(content_value)
                elif isinstance(last_message.get('content'), list):
                    last_message['content'].append({"type": "text", "text": str(content_value)})
                else:
                    logger.warning(f"Last message content had unexpected type {type(last_message.get('content'))}. Appending as new message.")
                    openai_messages.append({"role": openai_role, "content": content_value})
                    last_openai_role = openai_role
            else:
                openai_messages.append({"role": openai_role, "content": content_value})
                last_openai_role = openai_role

        if not openai_messages:
            logger.error("Cannot call OpenAI chat API: No valid messages derived from history after conversion.")
            return None, None, None

        logger.debug(f"Converted history to OpenAI messages format (length: {len(openai_messages)}).")
        return self._call_chat_api(openai_messages, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        # OpenAI multimodal uses a list of content objects (text or image_url)
        # within the 'content' field of a user message in chat completions.
        logger.debug(f"Attempting OpenAI multimodal generation with model {self.model} using chat completions.")

        openai_messages: List[Dict[str, Any]] = []
        current_message_content: List[Dict[str, Any]] = []
        current_role = "user"

        if not contents or not isinstance(contents, list) or not contents[0] or not isinstance(contents[0], dict):
            logger.error("Invalid format for multimodal contents input.")
            return None, None, None

        first_entry = contents[0]
        entry_role = first_entry.get("role", "user")
        current_role = "assistant" if entry_role == "model" else "user"
        if current_role != "user":
            logger.warning(f"Unexpected role '{entry_role}' for first entry in multimodal contents. Treating as 'user'.")
            current_role = "user"

        parts = first_entry.get("parts", [])

        if isinstance(parts, list):
            has_content = False
            for part in parts:
                if isinstance(part, str):
                    current_message_content.append({"type": "text", "text": part})
                    has_content = True
                elif isinstance(part, dict):
                    if "text" in part and isinstance(part["text"], str):
                        current_message_content.append({"type": "text", "text": part["text"]})
                        has_content = True
                    elif ("source" in part and
                          isinstance(part.get("source"), dict) and
                          "inline_data" in part["source"] and
                          isinstance(part["source"].get("inline_data"), dict)):
                        inline_data = part["source"]["inline_data"]
                        mime_type = inline_data.get("mime_type")
                        b64_data = inline_data.get("data")

                        if mime_type and b64_data and isinstance(mime_type, str) and isinstance(b64_data, str):
                            if mime_type.startswith("image/"):
                                image_url_data = f"data:{mime_type};base64,{b64_data}"
                                current_message_content.append({"type": "image_url", "image_url": {"url": image_url_data}})
                                has_content = True
                                logger.debug(f"Included image ({mime_type}) for OpenAI multimodal call.")
                            else:
                                logger.warning(f"Ignoring unsupported mime type '{mime_type}' for OpenAI image part.")
                        else:
                            logger.warning("Ignoring malformed inline_data part for OpenAI.")
                    else:
                        logger.warning(f"Ignoring unsupported part structure in multimodal message: {part}")
        elif isinstance(parts, str):
            current_message_content.append({"type": "text", "text": parts})
            has_content = True
        else:
            logger.warning(f"Multimodal contents entry 'parts' has unexpected type {type(parts)}. Skipping.")
            has_content = False

        if has_content:
            openai_messages.append({"role": current_role, "content": current_message_content})
        else:
            logger.error("Cannot call OpenAI multimodal API: No valid content derived from input.")
            return None, None, None

        if len(contents) > 1:
            logger.warning(f"Multimodal input 'contents' list contains {len(contents)} entries, but only the first entry is processed for OpenAI multimodal calls.")

        logger.debug(f"Calling OpenAI chat API for multimodal request (messages: {len(openai_messages)}).")
        return self._call_chat_api(openai_messages, **kwargs)

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        # Use chat completion for evaluation consistency
        logger.debug("Using chat API for evaluation request.")
        messages = [{"role": "user", "content": prompt}]
        return self._call_chat_api(messages, **kwargs)

# ---------------------------------------------------------------------------
#  xAI Grok implementation
# ---------------------------------------------------------------------------
class GrokInterface(LLMInterface):
    """Interface for xAI Grok models, using OpenAI-compatible API with reasoning support."""
    def __init__(self, api_key: str, model_name: str, reasoning_effort: str = "low"):
        super().__init__(api_key, model_name)
        if not api_key:
            raise ValueError("xAI API Key is required for GrokInterface.")
        if OpenAI is None:
            raise ImportError("openai library is not installed. Please install it to use GrokInterface.")
        self.reasoning_effort = reasoning_effort.lower()
        if self.reasoning_effort not in ["low", "high"]:
            logger.warning(f"Invalid reasoning_effort '{reasoning_effort}'. Defaulting to 'low'.")
            self.reasoning_effort = "low"
        # Add a check for supported models
        self.supports_reasoning = model_name in ["grok-3-mini-beta", "grok-3-mini-fast-beta"]  # Based on documentation
        if not self.supports_reasoning:
            logger.warning(f"Model '{model_name}' may not support reasoning_effort. It will be omitted if unsupported.")
        try:
            self.client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=api_key,
            )
            self.model = model_name
            logger.info(f"GrokInterface initialised with model: {model_name}, reasoning_effort: {self.reasoning_effort if self.supports_reasoning else 'N/A'}")
        except Exception as e:
            logger.error(f"Failed to initialize Grok client for model {model_name}: {e}", exc_info=True)
            raise ValueError(f"Grok initialization failed: {e}") from e

    # --- Internal helper for chat API calls with retries ---
    def _call_grok_chat_api(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:  # Added **kwargs to handle extra params like max_tokens
        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        resp: Optional[Any] = None  # Raw response object
        try:
            kwargs['model'] = self.model  # Ensure model is set
            kwargs['messages'] = messages  # Ensure messages are set
            if self.supports_reasoning:
                kwargs["reasoning_effort"] = self.reasoning_effort  # Only add if supported
                logger.debug(f"Calling Grok API with reasoning_effort '{self.reasoning_effort}' and additional kwargs: {kwargs}")
            else:
                logger.debug("Calling Grok API without reasoning_effort (not supported) and additional kwargs: {kwargs}")
            resp = self.client.chat.completions.create(**kwargs)  # Pass all kwargs, including max_tokens if provided
            # Store raw response and serialize for JSON compatibility
            raw_api_response = self._serialize_response(resp)  # Serialize to dict for saving

            # Extract text response safely
            choice_list = getattr(resp, 'choices', None)
            if choice_list is not None and hasattr(choice_list, '__len__') and len(choice_list) > 0:
                first_choice = choice_list[0]
                text_response = first_choice.message.content
                logger.debug(f"Grok extracted text response. Reason: {first_choice.finish_reason}")
                # Extract and log reasoning content if available
                reasoning_content = getattr(first_choice.message, 'reasoning_content', None)
                if reasoning_content:
                    logger.debug(f"Grok Reasoning Content: {reasoning_content}")
                else:
                    logger.debug("No reasoning content found in Grok response or not applicable.")
            else:
                logger.warning("Grok response 'choices' list is empty or invalid.")

            # Extract token usage safely
            usage = getattr(resp, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', None)
                completion_tokens = getattr(usage, 'completion_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)
                completion_tokens_details = getattr(usage, 'completion_tokens_details', None)
                reasoning_tokens = None
                if completion_tokens_details and hasattr(completion_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens = getattr(completion_tokens_details, 'reasoning_tokens', None)
                token_usage = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "reasoning_tokens": reasoning_tokens if self.supports_reasoning else None,  # Only include if supported
                }
                logger.debug(f"Grok API Token Usage: {token_usage}")
            else:
                logger.debug("No usage data found in Grok response.")

            if text_response is None:
                logger.warning("Failed to extract text response from Grok API object.")
            return text_response, token_usage, raw_api_response

        except Exception as e:
            logger.error(f"Grok API error: {e}", exc_info=True)
            return None, None, None

    def _serialize_response(self, response: Any) -> Dict:
        """Serialize the API response to a dictionary for JSON compatibility."""
        try:
            # Attempt to use built-in serialization if available
            if hasattr(response, 'model_dump'):
                return response.model_dump()  # For pydantic models
            elif hasattr(response, 'to_dict'):
                return response.to_dict()  # If the response has a to_dict method
            else:
                # Fallback to direct attribute access (less robust for nested objects)
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
                            serializable_dict[key] = str(value)  # Convert other types to string
                return serializable_dict
        except Exception as e:
            logger.warning(f"Failed to serialize Grok response object: {e}. Returning empty dict for safety.", exc_info=True)
            return {}  # Return empty dict to avoid crashing serialization

    # --- Implement abstract methods ---
    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_grok_chat_api(messages, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        # Convert Gemini format history to OpenAI messages format
        openai_messages: List[Dict[str, Any]] = []
        allowed_openai_roles = {"system", "user", "assistant"}
        last_openai_role = None

        for entry in history:
            gemini_role = entry.get("role", "user")
            parts = entry.get("parts", [])

            openai_role = "assistant" if gemini_role == "model" else "user"
            if openai_role not in allowed_openai_roles:
                logger.warning(f"Mapping unexpected Gemini role '{gemini_role}' to 'user' for Grok.")
                openai_role = "user"

            content_value: Any = None

            if isinstance(parts, list):
                text_content_parts: List[str] = []
                for p in parts:
                    if isinstance(p, str):
                        text_content_parts.append(p)
                    elif isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                        text_content_parts.append(p["text"])
                    elif isinstance(p, dict) and "source" in p and "inline_data" in p.get("source", {}):
                        mime = p["source"]["inline_data"].get("mime_type", "unk")
                        logger.debug(f"Ignoring image ({mime}) part in text conversational history conversion for Grok.")
                    else:
                        logger.warning(f"Skipping unsupported part type in history for Grok: {type(p)} / {p!r}")
                content_value = "\n".join(text_content_parts).strip()
            elif isinstance(parts, str):
                content_value = parts.strip()
            else:
                logger.warning(f"History entry 'parts' has unexpected type {type(parts)}. Skipping entry for Grok.")
                continue

            if not content_value:
                logger.debug(f"Skipping history entry from role '{gemini_role}' with empty content after processing parts for Grok.")
                continue

            if openai_messages and last_openai_role == openai_role:
                logger.warning(f"Consecutive messages with role '{openai_role}' detected for Grok. Appending content.")
                last_message = openai_messages[-1]
                if isinstance(last_message.get('content'), str):
                    last_message['content'] += "\n" + content_value
                else:
                    openai_messages.append({"role": openai_role, "content": content_value})
                    last_openai_role = openai_role
            else:
                openai_messages.append({"role": openai_role, "content": content_value})
                last_openai_role = openai_role

        if not openai_messages:
            logger.error("Cannot call Grok chat API: No valid messages derived from history.")
            return None, None, None

        logger.debug(f"Converted history to Grok messages format (length: {len(openai_messages)}).")
        return self._call_grok_chat_api(openai_messages, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        logger.warning("Multimodal generation is not supported by xAI Grok models.")
        return None, None, None

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_grok_chat_api(messages, **kwargs)

# ---------------------------------------------------------------------------
#  Quality Compute Simulator implementation
# ---------------------------------------------------------------------------
class QualityComputeInterface(LLMInterface):
    """Interface for Quality Compute simulator backend, with hardcoded URL."""
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        self.base_url = "https://qualitycompute.henosis.us"  # Hardcoded URL as per instructions; change this to your actual simulator URL
        if not api_key:
            raise ValueError("Quality Compute API Key is required.")
        logger.info(f"QualityComputeInterface initialised with model: {model_name}, base URL: {self.base_url}")

    def _call_quality_compute_api(self, input_data: str or List[Dict], **kwargs) -> LLMResponse:
        """Internal helper to call Quality Compute API with immediate response logging"""
        import json  # Ensure json is imported locally

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": input_data,
        }

        for key, value in kwargs.items():
            if key not in ["model", "input"]:
                payload[key] = value

        timeout_seconds = 300
        retries = 5
        base_delay = 30

        for attempt in range(retries):
            try:
                logger.debug(f"Calling Quality Compute API (Attempt {attempt+1}/{retries}) with payload: {payload}")
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    headers=headers,
                    json=payload,
                    timeout=timeout_seconds
                )
                duration = time.time() - start_time

                # IMMEDIATE RESPONSE LOGGING
                logger.error(f"QC API Response (Attempt {attempt+1}):")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Headers: {dict(response.headers)}")
                logger.error(f"  Content (first 500 chars): {response.text[:500]}")
                logger.error(f"  Duration: {duration:.2f}s")

                if response.status_code == 200:
                    try:
                        resp_json = response.json()
                        text_response = resp_json.get("selected_text")
                        usage_dict = resp_json.get("usage", {})

                        token_usage = {
                            "input_tokens": usage_dict.get("input_tokens", 0),
                            "output_tokens": usage_dict.get("output_tokens", 0),
                            "reasoning_tokens": usage_dict.get("reasoning_tokens", 0),
                            "total_tokens": usage_dict.get("total_tokens", 0),
                        }
                        raw_api_response = resp_json  # Quality Compute response is already JSON, so store it directly
                        logger.debug(f"QC API Raw Response: {raw_api_response}")
                        return text_response, token_usage, raw_api_response
                    except json.JSONDecodeError:
                        logger.error("JSON decode failed - dumping full response")
                        logger.error(f"Full response: {response.text}")
                        raise
                else:
                    error_detail = response.text[:500]
                    logger.warning(f"Non-200 response: {error_detail}")

            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                if attempt < retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    return None, None, {"error": str(e)}

        return None, None, None

    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        return self._call_quality_compute_api(prompt, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        # --- START COPIED/ADAPTED CONVERSION LOGIC ---
        openai_messages: List[Dict[str, Any]] = []
        allowed_openai_roles = {"system", "user", "assistant"}  # Simulator likely expects these roles
        last_openai_role = None
        for entry in history:
            gemini_role = entry.get("role", "user")
            parts = entry.get("parts", [])

            openai_role = "assistant" if gemini_role == "model" else "user"
            if openai_role not in allowed_openai_roles:
                logger.warning(f"Mapping unexpected Gemini role '{gemini_role}' to 'user' for QualityCompute.")
                openai_role = "user"

            content_value: Any = None
            if isinstance(parts, list):
                text_content_parts = []
                for p in parts:
                    if isinstance(p, str):
                        text_content_parts.append(p)
                    elif isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                        text_content_parts.append(p["text"])
                    # NOTE: Quality Compute simulator likely doesn't handle images, so ignore image parts
                    elif isinstance(p, dict) and "source" in p:
                         logger.debug("Ignoring non-text part in history conversion for QualityCompute.")
                    else:
                        logger.warning(f"Skipping unsupported part type in history for QualityCompute: {type(p)} / {p!r}")
                content_value = "\n".join(text_content_parts).strip()
            elif isinstance(parts, str):
                content_value = parts.strip()
            else:
                logger.warning(f"History entry 'parts' has unexpected type {type(parts)}. Skipping entry for QualityCompute.")
                continue

            if not content_value:
                logger.debug(f"Skipping history entry from role '{gemini_role}' with empty content after processing parts for QualityCompute.")
                continue

            # Handle consecutive messages (optional but good practice, simulator might handle it too)
            if openai_messages and last_openai_role == openai_role:
                logger.warning(f"Consecutive messages with role '{openai_role}' detected for QualityCompute. Appending content.")
                last_message = openai_messages[-1]
                if isinstance(last_message.get('content'), str):
                    last_message['content'] += "\n" + content_value
                else:
                     openai_messages.append({"role": openai_role, "content": content_value})
                     last_openai_role = openai_role
            else:
                openai_messages.append({"role": openai_role, "content": content_value})
                last_openai_role = openai_role

        if not openai_messages:
            logger.error("Cannot call Quality Compute API: No valid messages derived from history after conversion.")
            return None, None, None
        logger.debug(f"Converted history to Quality Compute messages format (length: {len(openai_messages)}).")
        # --- END CONVERSION LOGIC ---

        # Call the internal API helper with the converted messages list and pass kwargs
        return self._call_quality_compute_api(openai_messages, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        logger.warning("Multimodal generation is not supported by Quality Compute interface. Falling back to text-only.")
        # Extract text from contents
        text_content = " ".join([part.get("text", "") for item in contents for part in item.get("parts", []) if isinstance(part, dict) and "text" in part])
        return self._call_quality_compute_api(text_content, **kwargs)

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        return self._call_quality_compute_api(prompt, **kwargs)

# ---------------------------------------------------------------------------
#  Anthropic implementation
# ---------------------------------------------------------------------------
class AnthropicInterface(LLMInterface):
    """Interface for Anthropic Claude models."""
    def __init__(self, api_key: str, model_name: str, thinking_enabled: bool = False, thinking_budget: int = 16000):
        super().__init__(api_key, model_name)
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

    # --- Internal helper for chat API calls with retries ---
    def _call_anthropic_api(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        text_response: Optional[str] = None
        token_usage: TokenUsage = None
        resp: Optional["Message"] = None

        retries = 3
        delay = 10  # seconds

        for attempt in range(retries):
            try:
                logger.debug(f"Calling Anthropic API with streaming (Attempt {attempt+1}/{retries}) and additional kwargs: {kwargs}")

                anthropic_kwargs = {
                    "model": self.model,
                    "max_tokens": 64000,  # Default max_tokens, can be overridden by kwargs
                    "messages": messages,
                }

                # Add system prompt if provided
                if system_prompt:
                    anthropic_kwargs["system"] = system_prompt

                # Add any additional kwargs (e.g., max_tokens, temperature)
                for key, value in kwargs.items():
                    if key not in ["model", "messages", "system"]:  # Avoid overwriting core fields
                        anthropic_kwargs[key] = value  # Pass through parameters like max_tokens

                if self.thinking_enabled:
                    anthropic_kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
                    logger.debug(f"Extended thinking enabled with budget: {self.thinking_budget}")
                else:
                    anthropic_kwargs["thinking"] = {"type": "disabled"}  # Explicitly disable if not enabled

                accumulated_text_parts: List[str] = []
                with self.client.messages.stream(**anthropic_kwargs) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                accumulated_text_parts.append(event.delta.text)
                            elif event.delta.type == "thinking_delta":
                                # Skip thinking_delta as per instruction to exclude thoughts
                                logger.debug("Skipping thinking_delta content as per configuration.")
                # After stream completes, get the final message for usage and other details
                final_model_response_obj = stream.get_final_message()

                text_response = "".join(accumulated_text_parts).strip()

                # Extract token usage from the final message object
                if final_model_response_obj and hasattr(final_model_response_obj, 'usage'):
                    usage_data = final_model_response_obj.usage
                    input_tokens = getattr(usage_data, 'input_tokens', None)
                    output_tokens = getattr(usage_data, 'output_tokens', None)
                    total_tokens = None
                    if input_tokens is not None and output_tokens is not None:
                        total_tokens = input_tokens + output_tokens

                    token_usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    }
                    logger.debug(f"Anthropic API Token Usage (from stream): {token_usage}")
                else:
                    logger.debug("No usage data found in Anthropic final_model_response_obj from stream.")

                # Serialize raw response for JSON compatibility
                raw_api_response_serialized = self._serialize_response(final_model_response_obj)
                logger.debug(f"Serialized Anthropic Raw API Response: {raw_api_response_serialized}")

                # Log stop reason or other details if available
                if final_model_response_obj:
                    stop_reason = getattr(final_model_response_obj, 'stop_reason', 'N/A')
                    logger.debug(f"Anthropic stream completed. Stop reason: {stop_reason}")

                return text_response, token_usage, raw_api_response_serialized

            except anthropic.APIConnectionError as e:
                logger.warning(f"Anthropic API connection error (Attempt {attempt+1}/{retries}): {e}", exc_info=True)
            except anthropic.RateLimitError as e:
                logger.warning(f"Anthropic API rate limit error (Attempt {attempt+1}/{retries}): {e}", exc_info=True)
            except anthropic.APIStatusError as e:
                logger.warning(f"Anthropic API status error {e.status_code} (Attempt {attempt+1}/{retries}): {e.message}", exc_info=True)
            except Exception as e:
                logger.warning(f"Anthropic API general error during streaming/processing (Attempt {attempt+1}/{retries}): {e}", exc_info=True)

            # If an exception occurred and it's not the last attempt, sleep and retry
            if attempt < retries - 1:
                actual_delay = delay * (attempt + 1)
                logger.info(f"Retrying Anthropic API call in {actual_delay} seconds...")
                time.sleep(actual_delay)
            else:
                logger.error(f"Anthropic API failed after {retries} attempts")
                return None, None, None

        return None, None, None  # Fallback if all retries fail

    def _serialize_response(self, response: Any) -> Dict:
        """Serialize the API response to a dictionary for JSON compatibility."""
        try:
            # Attempt to use built-in serialization
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            elif hasattr(response, 'to_dict'):
                return response.to_dict()
            else:
                # Fallback to direct attribute access (less robust for nested objects)
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
                            serializable_dict[key] = str(value)  # Convert other types to string
                return serializable_dict
        except Exception as e:
            logger.warning(f"Failed to serialize Anthropic response object: {e}. Returning empty dict.", exc_info=True)
            return {}  # Return empty dict to avoid crashing serialization

    # --- Convert Gemini format to Anthropic format ---
    def _convert_to_anthropic_messages(self, history: List[Dict]) -> List[Dict[str, Any]]:
        anthropic_messages = []

        for entry in history:
            gemini_role = entry.get("role", "user")
            parts = entry.get("parts", [])

            # Map Gemini roles to Anthropic roles
            anthropic_role = "assistant" if gemini_role == "model" else "user"

            content_value = None
            if isinstance(parts, list):
                text_content_parts = []
                for p in parts:
                    if isinstance(p, str):
                        text_content_parts.append(p)
                    elif isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                        text_content_parts.append(p["text"])
                    else:
                        logger.warning(f"Skipping unsupported part type in history for Anthropic: {type(p)}")
                content_value = "\n".join(text_content_parts).strip()
            elif isinstance(parts, str):
                content_value = parts.strip()

            if content_value:
                anthropic_messages.append({"role": anthropic_role, "content": content_value})

        if not anthropic_messages:
            logger.error("No valid messages derived from history for Anthropic")
            return []
        return anthropic_messages

    # --- Implement abstract methods ---
    def generate_action(self, prompt: str, **kwargs) -> LLMResponse:
        anthropic_messages = [{"role": "user", "content": prompt}]
        return self._call_anthropic_api(anthropic_messages, **kwargs)

    def generate_action_conversational(self, history: List[Dict], **kwargs) -> LLMResponse:
        anthropic_messages = self._convert_to_anthropic_messages(history)
        if not anthropic_messages:
            return None, None, None

        logger.debug(f"Converted history to Anthropic messages format (length: {len(anthropic_messages)})")
        return self._call_anthropic_api(anthropic_messages, **kwargs)

    def generate_content_multimodal(self, contents: List[Dict], **kwargs) -> LLMResponse:
        logger.warning("Multimodal support for Anthropic is not fully implemented yet. Converting to text-only.")

        messages = []
        if not contents or not isinstance(contents, list):
            logger.error("Invalid format for multimodal contents input")
            return None, None, None

        # Extract text from first content entry
        first_entry = contents[0]
        entry_role = first_entry.get("role", "user")
        parts = first_entry.get("parts", [])

        anthropic_role = "assistant" if entry_role == "model" else "user"

        text_content = ""
        for part in parts:
            if isinstance(part, str):
                text_content += part + "\n"
            elif isinstance(part, dict) and "text" in part:
                text_content += part["text"] + "\n"
            elif isinstance(part, dict) and "source" in part:
                # Image parts are not supported in this basic implementation
                logger.warning("Skipping image part in multimodal content for Anthropic")

        if text_content:
            messages.append({"role": anthropic_role, "content": text_content.strip()})
            return self._call_anthropic_api(messages, **kwargs)
        else:
            logger.error("No valid text content found in multimodal input")
            return None, None, None

    def evaluate_outcome(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call_anthropic_api(messages, **kwargs)