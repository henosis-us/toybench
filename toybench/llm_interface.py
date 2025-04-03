import google.generativeai as genai
import logging
import time
from abc import ABC, abstractmethod
import re

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

    # --- NEW METHOD ---
    @abstractmethod
    def generate_action_conversational(self, history: list[dict]) -> str | None:
        """
        Generates the next action based on a conversation history.
        History format: [{'role': 'user'/'model', 'parts': [text]}, ...]
        """
        pass
    # --- END NEW METHOD ---

    @abstractmethod
    def evaluate_outcome(self, prompt: str) -> str | None:
        """Evaluates a final state/outcome based on a prompt."""
        pass

    @abstractmethod
    def _parse_action(self, response: str) -> str:
        """Extracts the action string from the raw LLM response."""
        pass # Might need adjustment based on conversational output


class GeminiInterface(LLMInterface):
    """Implementation for Google Gemini models."""
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        if not api_key:
             raise ValueError("Gemini API Key is required for GeminiInterface.")
        genai.configure(api_key=api_key)
        # Consider adding generation_config here if needed
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"GeminiInterface initialized with model: {model_name}")

    def _call_api(self, prompt_or_history: str | list[dict], retries=3, delay=5, is_chat=False) -> str | None:
        """Internal method to call the Gemini API with retry logic."""
        last_exception = None
        for attempt in range(retries):
            try:
                client = self.model
                # Use start_chat for conversational history if needed by underlying API/library version
                # Or just pass history directly if generate_content supports it well
                # Current google-generativeai library supports history directly in generate_content
                if is_chat:
                     logger.debug(f"Calling Gemini API (Chat - Attempt {attempt + 1}/{retries}). History length: {len(prompt_or_history)}")
                     # Ensure history format is correct for the API
                     response = client.generate_content(prompt_or_history)
                else:
                     logger.debug(f"Calling Gemini API (Non-Chat - Attempt {attempt + 1}/{retries}). Prompt starts with: {prompt_or_history[:100]}...")
                     response = client.generate_content(prompt_or_history)

                # Handle potential safety blocks or empty responses
                if not response.parts:
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                         reason = response.prompt_feedback.block_reason
                         logger.warning(f"Gemini API call blocked. Reason: {reason}")
                         last_exception = Exception(f"API call blocked: {reason}")
                         # Treat block as non-retryable for now? Or based on reason?
                         # return None # Blocked content is a hard failure
                         continue # Or retry? Let's retry for now.
                    else:
                         logger.warning("Gemini API returned no parts in response.")
                         last_exception = Exception("API returned no parts")
                         continue # Retry on empty response

                response_text = response.text
                logger.debug(f"Gemini API Response: {response_text[:500]}...") # Log truncated response
                return response_text
            except Exception as e:
                logger.warning(f"Gemini API call failed (Attempt {attempt + 1}/{retries}): {e}")
                last_exception = e
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1)) # Exponential backoff
                else:
                    logger.error(f"Gemini API call failed after {retries} retries.")
                    return None # Return None after all retries fail
        return None # Should only be reached if all retries fail

    def generate_action(self, prompt: str) -> str | None:
        """Generates an action using the Gemini model (non-conversational)."""
        response_text = self._call_api(prompt, is_chat=False)
        if response_text:
            # Parsing might be different depending on whether completion signal is expected here too
            return self._parse_action(response_text) # Use existing parser for now
        return None

    # --- IMPLEMENT NEW METHOD ---
    def generate_action_conversational(self, history: list[dict]) -> str | None:
        """Generates the next action based on conversation history."""
        # The google-generativeai library expects history in the format:
        # [{'role': 'user'/'model', 'parts': ["text content"]}]
        # Ensure the passed history conforms to this.
        formatted_history = []
        for msg in history:
            role = msg.get('role')
            content = msg.get('content') or msg.get('parts') # Adapt based on how history is stored
            if role and content:
                 # Ensure parts is a list containing the text
                 parts_list = [content] if isinstance(content, str) else content
                 formatted_history.append({'role': role, 'parts': parts_list})
            else:
                 logger.warning(f"Skipping message in history due to missing role or content: {msg}")


        # Pass the history directly to generate_content
        response_text = self._call_api(formatted_history, is_chat=True)
        # No need to call specific parse_action here, return the raw response
        # The calling function (run_attempt) will handle parsing command + completion signal
        return response_text
    # --- END IMPLEMENTATION ---

    def evaluate_outcome(self, prompt: str) -> str | None:
        """Evaluates an outcome using the Gemini model."""
        # Evaluation likely doesn't need chat history
        return self._call_api(prompt, is_chat=False)

    def _parse_action(self, response: str) -> str:
        """
        Extracts the primary action string (e.g., command) from the raw LLM response.
        Tries to find ```action ... ``` block first.
        Falls back to first non-empty line if block not found.
        NOTE: This does NOT look for completion signals like TASK_COMPLETE.
        """
        action_match = re.search(r"```action\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            logger.debug(f"Parsed action (from ```action block): {action}")
            return action

        # Fallback: Take the first non-empty line as action
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        if lines:
             action = lines[0]
             # Avoid picking TASK_COMPLETE as the action if it's the only line
             if "TASK_COMPLETE" in action.upper() and len(lines)==1:
                  logger.debug("Response contains only TASK_COMPLETE, parsing action as empty.")
                  return ""
             logger.debug(f"Parsed action (first non-empty line): {action}")
             return action

        logger.warning(f"Could not parse action command from response: {response[:100]}")
        return "" # Return empty string if no command-like action found