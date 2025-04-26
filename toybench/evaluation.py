# evaluation.py
import logging
import os
from llm_interface import LLMInterface
from utils import parse_llm_score

# Import necessary helper from browser utils
try:
    # Ensure this imports the *real* function now
    from browser_utils import encode_file_inline_data_gemini
    BROWSER_UTILS_AVAILABLE = True
except ImportError:
    BROWSER_UTILS_AVAILABLE = False
    # Define dummy only if needed (e.g., for testing without full setup)
    def encode_file_inline_data_gemini(file_path: str) -> dict | None:
        logger = logging.getLogger(__name__)
        logger.error("encode_file_inline_data_gemini called but utils.browser_utils failed to import.")
        return None

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, evaluator_llm: LLMInterface, task_eval_prompt_template: str):
        """
        Initializes the Evaluator.
        Args:
            evaluator_llm: An instance of LLMInterface used for evaluation.
            task_eval_prompt_template: The prompt template string used for final evaluation.
                                      This might be text-based or multimodal depending on the task.
        """
        self.llm = evaluator_llm
        self.eval_prompt_template = task_eval_prompt_template
        # Stores status from previous turn for regression check {attempt_id: status}
        self.previous_statuses = {} # Currently unused for solar_gen

    def evaluate_final_outcome(self, final_eval_input: str) -> tuple[int, str]:
        """
        Uses the Evaluator LLM to score the final outcome based on text input and the task prompt.
        (Not used for solar_gen final evaluation).
        Returns score (1, 2, or 3) and raw response text.
        Returns score 1 on failure to parse or LLM error.
        """
        # Ensure the evaluator has the correct template (might be set externally before call)
        if not self.eval_prompt_template:
             logger.error("Final evaluation prompt template is not set for Evaluator.")
             return 1, "Evaluator prompt template missing."

        # Basic formatting assuming the template has a {final_outcome} placeholder
        try:
             prompt = self.eval_prompt_template.format(final_outcome=final_eval_input)
        except KeyError:
             logger.error(f"Failed to format final eval prompt. Placeholder likely missing. Input: {final_eval_input[:100]}...")
             # Fallback: Send raw input + generic instruction? Or fail? Let's fail clearly.
             return 1, "Failed to format final evaluation prompt (missing placeholder?)."

        logger.info("Requesting final text evaluation from LLM.")
        logger.debug(f"Final Text Evaluation Prompt (truncated): {prompt[:500]}")

        # Use the standard text generation method of the LLM interface
        # Assuming evaluate_outcome uses a text-based generation method
        response_text = self.llm.evaluate_outcome(prompt) # evaluate_outcome might just call generate_action internally
        if response_text is None:
            logger.error("Failed to get response from evaluator LLM for text evaluation.")
            return 1, "Evaluator LLM failed to respond." # Fail score

        logger.info(f"Received final text evaluation response: {response_text[:100]}...")
        score = parse_llm_score(response_text)
        if score is None:
            logger.warning(f"Could not parse score from text evaluation response: {response_text}")
            # Return score 1 but include the response for debugging
            return 1, f"Could not parse score. LLM Response: {response_text}"

        logger.info(f"Parsed final text score: {score}")
        return score, response_text

    # --- Method for Multimodal Image Evaluation (Used by solar_gen) ---
    def evaluate_final_image_outcome(self, image_path: str) -> tuple[int, str]:
        """
        Uses the Evaluator LLM (multimodal capable) to score the final outcome based
        on an image input and the task prompt template.
        Args:
            image_path (str): Path to the final image file (e.g., screenshot).
        Returns:
            tuple[int, str]: (score (1, 2, or 3), raw_response_text).
                            Returns score 1 on failure (file missing, encode error, LLM error, parse error).
        """
        logger.info(f"Requesting final image evaluation for: {image_path}")

        # 0. Check if browser utils were available for encoding
        if not BROWSER_UTILS_AVAILABLE:
            logger.error("Cannot perform image evaluation: browser_utils failed to import.")
            return 1, "Evaluation failed due to missing browser utilities."

        # 1. Check if image path exists and is valid
        if not image_path or image_path == "SCREENSHOT_UNAVAILABLE" or not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            logger.error(f"Final image file is missing, empty, or invalid: {image_path}")
            return 1, f"Final image file missing or invalid: {image_path}"

        # 2. Encode the image using the imported function
        image_data = encode_file_inline_data_gemini(image_path)
        if image_data is None:
            logger.error(f"Failed to encode image file: {image_path}")
            return 1, f"Failed to encode image file: {image_path}"  # Encoding error is a failure

        # 3. Construct the multimodal payload
        # Ensure the final evaluation prompt template is set
        if not self.eval_prompt_template:
            logger.error("Final image evaluation prompt template is not set for Evaluator.")
            return 1, "Evaluator prompt template missing for image evaluation."

        # Gemini API's generateContent payload structure:
        multimodal_payload = [
            {
                "role": "user",
                "parts": [
                    # image_data['source'] contains the {'inline_data': {...}} structure
                    image_data["source"],
                    {"text": self.eval_prompt_template}  # The text instructions
                ]
            }
        ]

        # --- CORRECTED DEBUG LINE ---
        try:
            # Safely access nested keys for logging
            mime_type_log = multimodal_payload[0]['parts'][0].get('inline_data', {}).get('mime_type', 'N/A')
            text_log = multimodal_payload[0]['parts'][1].get('text', '')[:100]
            logger.debug(f"Multimodal Payload Structure (Image data truncated):\nMimeType={mime_type_log}..., Text: {text_log}...")
        except (IndexError, KeyError, AttributeError) as log_e:
            logger.warning(f"Could not log multimodal payload structure details: {log_e}")
        # --- END CORRECTION ---

        # 4. Call the LLM using a multimodal method
        # Ensure the LLM interface instance supports multimodal calls
        if not hasattr(self.llm, 'generate_content_multimodal'):
            logger.error("LLM interface provided to Evaluator does not support 'generate_content_multimodal'.")
            return 1, "Evaluator LLM interface does not support multimodal generation."

        response_text = None
        try:
            logger.info("Sending request to multimodal evaluator LLM.")
            response_text, _, _ = self.llm.generate_content_multimodal(multimodal_payload)  # Unpack the tuple returned by generate_content_multimodal
        except Exception as e:
            logger.error(f"Error during multimodal LLM call: {e}", exc_info=True)
            return 1, f"Multimodal LLM call failed: {e}"  # LLM call failure -> Score 1

        if response_text is None:
            logger.error("Failed to get response from multimodal evaluator LLM.")
            return 1, "Multimodal Evaluator LLM failed to respond."  # Fail score if no response

        # Add type check for response_text before parsing
        if not isinstance(response_text, str):
            logger.error(f"Response text is not a string (type: {type(response_text)}), cannot parse score. Raw response: {response_text}")
            return 1, f"Invalid response type from LLM: {type(response_text)}"

        logger.info(f"Received final multimodal evaluation response: {response_text[:100]}...")

        # 5. Parse the score
        score = parse_llm_score(response_text)
        if score is None:
            logger.warning(f"Could not parse score from multimodal evaluation response: {response_text}")
            # Fail score if parsing fails, but include the response text
            return 1, f"Could not parse score. LLM Response: {response_text}"

        logger.info(f"Parsed final multimodal score: {score}")
        return score, response_text

    # --- END METHOD ---

    # Regression tracking is not currently used by SolarSystemEnv
    def track_regression(self, attempt_id: int, current_status: any, turn: int) -> bool:
        """Compares current status to the previous status for the same attempt."""
        if current_status is None: return False
        last_status = self.previous_statuses.get(attempt_id)
        regression_detected = False
        if last_status is not None and turn > 0:
            try:
                if current_status < last_status:
                    regression_detected = True
                    logger.warning(f"Regression detected! Attempt {attempt_id}, Turn {turn}: Status changed from {last_status} to {current_status}")
            except TypeError:
                logger.debug(f"Cannot directly compare statuses for regression: {last_status} vs {current_status}")
                pass
        self.previous_statuses[attempt_id] = current_status
        return regression_detected

    def reset_attempt_tracking(self, attempt_id: int):
        """Clears the stored status for a given attempt ID."""
        if attempt_id in self.previous_statuses:
            del self.previous_statuses[attempt_id]