# llm_interface.py
import google.generativeai as genai
import logging
import time
from abc import ABC, abstractmethod
import re

# Attempt to import relogger if available, otherwise ignore
try:
    import relogger
except ImportError:
    relogger = None # Set to None if not installed

logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    """Abstract Base Class for LLM API interaction."""
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    def generate_action(self, prompt: str) -> str | None:
        """Generates an action based on a single prompt (for non-conversational tasks)."""
        pass

    @abstractmethod
    def generate_action_conversational(self, history: list[dict]) -> str | None:
        """
        Generates the next action based on a conversation history.
        History format: [{'role': 'user'/'model', 'parts': [text]}, ...]
        """
        pass

    # --- NEW ABSTRACT METHOD ---
    @abstractmethod
    def generate_content_multimodal(self, contents: list[dict]) -> str | None:
        """
        Generates content based on multimodal input.
        'contents' is expected to follow the API structure, e.g., for Gemini:
        [{'role': 'user', 'parts': [{'text': '...'}, {'inline_data': {...}}]}]
        """
        pass
    # --- END NEW METHOD ---

    @abstractmethod
    def evaluate_outcome(self, prompt: str) -> str | None:
        """Evaluates a final state/outcome based on a prompt (typically text-based)."""
        # Often, this might just call generate_action internally.
        pass

    # This parser might need adjustment depending on whether generate_action is used by evaluate_outcome
    # and whether evaluation responses need special parsing (like score tags).
    # The score parsing is handled by utils.parse_llm_score now.
    # def _parse_action(self, response: str) -> str:
    #     """Extracts the action string from the raw LLM response."""
    #     pass


class GeminiInterface(LLMInterface):
    """Implementation for Google Gemini models."""
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        if not api_key:
             raise ValueError("Gemini API Key is required for GeminiInterface.")
        try:
            genai.configure(api_key=api_key)
            # Consider adding generation_config here if needed (e.g., temperature, max_output_tokens)
            # self.generation_config = genai.types.GenerationConfig(...)
            self.model = genai.GenerativeModel(model_name) #, generation_config=self.generation_config)
            logger.info(f"GeminiInterface initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini GenerativeModel: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize Gemini model '{model_name}'. Check API key and model name.") from e

    # MODIFIED: _call_api now handles direct content lists for multimodal
    def _call_api(self, prompt_or_contents: str | list[dict], retries=3, delay=10) -> str | None:
        """
        Internal method to call the Gemini API with retry logic.
        Handles text prompts, chat history (list of dicts), and direct multimodal content lists.
        """
        last_exception = None
        payload = prompt_or_contents # Default payload

        # Check if input is a chat history or multimodal content list
        is_list_input = isinstance(prompt_or_contents, list)

        for attempt in range(retries):
            try:
                client = self.model
                if is_list_input:
                     # Input is already formatted as contents list (chat history or multimodal)
                     logger.debug(f"Calling Gemini API (List Input - Attempt {attempt + 1}/{retries}). Content items: {len(prompt_or_contents)}")
                     response = client.generate_content(payload) # Pass the list directly
                else:
                     # Input is a simple string prompt
                     logger.debug(f"Calling Gemini API (String Prompt - Attempt {attempt + 1}/{retries}). Prompt starts with: {prompt_or_contents[:100]}...")
                     response = client.generate_content(payload) # Pass the string directly

                # Handle potential safety blocks or empty responses
                # Check response structure based on documentation
                # https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content
                if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                    reason = "Unknown reason (empty response structure)"
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "N/A"
                    safety_ratings = response.candidates[0].safety_ratings if response.candidates else "N/A"
                    prompt_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else "N/A"

                    # More detailed logging for empty/blocked responses
                    logger.warning(f"Gemini API call returned empty/blocked response (Attempt {attempt+1}).")
                    logger.warning(f"  Finish Reason: {finish_reason}")
                    logger.warning(f"  Safety Ratings: {safety_ratings}")
                    logger.warning(f"  Prompt Feedback: {prompt_feedback}")

                    if finish_reason == genai.types.FinishReason.SAFETY:
                         reason = f"Blocked due to SAFETY. Ratings: {safety_ratings}"
                         last_exception = Exception(reason)
                         # Safety blocks are often not retryable
                         logger.error(f"Gemini API call blocked for safety after {attempt+1} attempts.")
                         return None # Hard failure on safety block
                    elif finish_reason == genai.types.FinishReason.RECITATION:
                         reason = f"Blocked due to RECITATION."
                         last_exception = Exception(reason)
                         logger.error(f"Gemini API call blocked for recitation after {attempt+1} attempts.")
                         return None # Hard failure on recitation block
                    else: # Other reasons or empty structure
                         reason = f"Empty response structure. Finish Reason: {finish_reason}"
                         last_exception = Exception(reason)
                         logger.warning("Retrying on potentially empty/malformed response...")
                         # Continue to retry logic below

                else:
                    # Extract text response
                    # Assuming the first candidate and first part contain the primary text response
                    response_text = response.candidates[0].content.parts[0].text
                    logger.debug(f"Gemini API Response (truncated): {response_text[:500]}...")
                    return response_text # Success

            except Exception as e:
                logger.warning(f"Gemini API call failed (Attempt {attempt + 1}/{retries}): {e}", exc_info=True)
                last_exception = e
                # Check for specific API errors that might not be retryable (e.g., auth, invalid arg)
                # Placeholder for specific error handling if needed
                # if isinstance(e, google.api_core.exceptions.PermissionDenied): return None

            # Retry logic
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1)) # Exponential backoff
            else:
                logger.error(f"Gemini API call failed after {retries} retries.")
                # Optionally re-raise the last exception or just return None
                # raise last_exception from None
                return None # Return None after all retries fail

        return None # Should only be reached if all retries fail


    def generate_action(self, prompt: str) -> str | None:
        """Generates an action using the Gemini model (non-conversational)."""
        response_text = self._call_api(prompt)
        # Parsing is handled by the caller (toybench_cli.parse_agent_response)
        return response_text


    def generate_action_conversational(self, history: list[dict]) -> str | None:
        """
        Generates the next action based on conversation history.
        Assumes history is already in the correct Gemini format:
        [{'role': 'user'/'model', 'parts': ["text content"]}, ...]
        """
        # Validate history format minimally?
        if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
             logger.error("Invalid history format passed to generate_action_conversational.")
             return None
        # Pass the history list directly to _call_api
        response_text = self._call_api(history)
        # Parsing is handled by the caller (toybench_cli.parse_agent_response)
        return response_text


    # --- IMPLEMENT NEW METHOD ---
    def generate_content_multimodal(self, contents: list[dict]) -> str | None:
        """
        Generates content based on multimodal input using the Gemini model.

        Args:
            contents (list[dict]): The structured multimodal content list, expected
                                  to follow Gemini API format, e.g.,
                                  [{'role': 'user', 'parts': [img_part, text_part]}].
        """
        logger.info("Calling Gemini API for multimodal generation.")
        # Validate contents format minimally?
        if not isinstance(contents, list) or not all(isinstance(item, dict) for item in contents):
             logger.error("Invalid contents format passed to generate_content_multimodal.")
             return None

        # Pass the structured contents list directly to _call_api
        response_text = self._call_api(contents)
        # Caller (e.g., Evaluator) will handle parsing the response (e.g., extracting score)
        return response_text
    # --- END IMPLEMENTATION ---


    def evaluate_outcome(self, prompt: str) -> str | None:
        """Evaluates an outcome using the Gemini model (text-based)."""
        # This simply calls the standard text generation.
        logger.info("Calling Gemini API for text-based evaluation.")
        response_text = self._call_api(prompt)
        # Caller (Evaluator.evaluate_final_outcome) handles parsing score etc.
        return response_text

    # _parse_action is removed as parsing is now handled by toybench_cli.parse_agent_response