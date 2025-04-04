# environments/solar_system_env.py
import logging
import os
import re
import time
import json
from typing import Tuple, Any, Dict

# Correctly import the required functions from utils.browser_utils
# No more placeholder functions should be defined in this file.
try:
    from browser_utils import render_and_capture, encode_file_inline_data_gemini, format_feedback_message
    BROWSER_UTILS_AVAILABLE = True
    logger_init = logging.getLogger(__name__) # Use logger early
    logger_init.info("Successfully imported browser utilities.")
except ImportError as e:
    BROWSER_UTILS_AVAILABLE = False
    # Define dummy functions ONLY IF import fails, to prevent crashes but indicate the issue.
    # This allows the CLI to potentially run other tasks, but Solar will fail gracefully.
    logger_init = logging.getLogger(__name__) # Use logger early
    logger_init.error(f"Failed to import from utils.browser_utils: {e}. SolarSystemEnv will not function correctly.")

    def render_and_capture(html_file_path: str, screenshot_path: str, browser_log_path: str, browser_type="chrome") -> tuple[bool, str]:
        logger = logging.getLogger(__name__)
        err_msg = "Error: render_and_capture called, but browser_utils failed to import."
        logger.error(err_msg)
        # Attempt to create empty files for consistency
        try: os.makedirs(os.path.dirname(screenshot_path), exist_ok=True); open(screenshot_path, 'w').close()
        except OSError: pass
        try: os.makedirs(os.path.dirname(browser_log_path), exist_ok=True); open(browser_log_path, 'w').write(err_msg)
        except OSError: pass
        return False, err_msg

    def encode_file_inline_data_gemini(file_path: str) -> dict | None:
        logger = logging.getLogger(__name__)
        logger.error("Error: encode_file_inline_data_gemini called, but browser_utils failed to import.")
        return None

    def format_feedback_message(eval_response: str, browser_logs: str) -> str:
        logger = logging.getLogger(__name__)
        logger.error("Error: format_feedback_message called, but browser_utils failed to import.")
        return f"FEEDBACK UNAVAILABLE (browser_utils import failed)\nEval Response: {eval_response}\nLogs: {browser_logs}"

from .base_env import BaseEnvironment
# Import the LLM interface - needed for type hinting
from llm_interface import LLMInterface

logger = logging.getLogger(__name__) # Standard logger for the rest of the class

class SolarSystemEnv(BaseEnvironment):
    """
    Environment for the Solar System HTML generation task.
    Manages state (HTML, screenshot, logs), handles browser rendering,
    intermediate evaluation, and prepares final evaluation input.
    Relies on functions imported from utils.browser_utils.
    """
    # Define file naming conventions relative to the attempt's output directory
    HTML_FILENAME = "solar_iteration_{turn}.html"
    SCREENSHOT_FILENAME = "solar_screenshot_iteration_{turn}.png"
    LOG_FILENAME = "browser_logs_iteration_{turn}.txt"
    # TEMP_EVAL_FILENAME = "intermediate_eval_response_{turn}.json" # Optional: If saving intermediate eval JSON

    def __init__(self,
                 goal_description: str,
                 intermediate_eval_prompt: str,
                 intermediate_eval_llm: LLMInterface,
                 output_dir_base: str, # Base directory for THIS attempt's files
                 max_steps: int):
        """
        Initializes the Solar System Environment.
        Args:
            goal_description (str): The main goal prompt content.
            intermediate_eval_prompt (str): The prompt template for intermediate visual evaluation.
            intermediate_eval_llm (LLMInterface): LLM interface for intermediate evaluations.
            output_dir_base (str): The specific directory to store files for the current attempt.
            max_steps (int): Maximum number of refinement turns allowed.
        """
        if not BROWSER_UTILS_AVAILABLE:
             # Raise an error during initialization if critical dependencies are missing
             # This prevents the benchmark from trying to run Solar without the necessary tools
             raise ImportError("Cannot initialize SolarSystemEnv: Failed to import required functions from utils.browser_utils. Check installation and logs.")

        self._goal = goal_description
        self._intermediate_eval_prompt = intermediate_eval_prompt
        self._intermediate_eval_llm = intermediate_eval_llm
        self._output_dir = output_dir_base
        self._max_steps = max_steps
        self.current_turn = 0
        self.current_html_path = None
        self.current_screenshot_path = None
        self.current_browser_log_path = None
        self.last_feedback_text = "No feedback yet. Generate the initial HTML based on the goal."
        self.last_render_success = False

        # Ensure the output directory exists
        try:
            os.makedirs(self._output_dir, exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create output directory {self._output_dir}: {e}", exc_info=True)
             raise # Re-raise error if directory creation fails

        logger.info(f"SolarSystemEnv initialized. Output directory: {self._output_dir}")

    def _get_path(self, filename_template: str, turn: int) -> str:
        """Constructs the full path for a file based on the turn number."""
        return os.path.join(self._output_dir, filename_template.format(turn=turn))

    def reset(self) -> str:
        """
        Resets the environment for a new attempt. Cleans up temporary files (optional)
        and returns the initial state description (the goal).
        """
        self.current_turn = 0
        self.current_html_path = None
        self.current_screenshot_path = None
        self.current_browser_log_path = None
        self.last_feedback_text = "No feedback yet. Generate the initial HTML based on the goal."
        self.last_render_success = False

        # Optional: Clean up files from previous runs within this specific directory if desired
        # logger.debug(f"Resetting Solar Env - clearing files in {self._output_dir} if needed...")

        logger.info("SolarSystemEnv Reset. Ready for first generation.")
        # Initial state for the agent is simply the goal description
        return self._goal # The prompt template will combine this with feedback later

    def get_prompt_context(self) -> Dict[str, Any]:
        """
        Returns the context dictionary for formatting the agent's prompt.
        Includes the main goal and the feedback from the last step.
        """
        # Ensure feedback is always a string, even if initialization had issues (though reset should handle)
        feedback = self.last_feedback_text if isinstance(self.last_feedback_text, str) else "Error retrieving feedback."
        return {
            "goal": self._goal,
            "current_state": feedback # 'current_state' maps to feedback for the agent prompt
        }

    def validate_action(self, action: str) -> bool:
        """
        Validates the generated HTML code (the 'action').
        Checks if it's non-empty. Basic structure checks are optional.
        """
        if not action or not action.strip():
            logger.warning("Validation failed: Received empty action (HTML code).")
            return False

        # More robust check might involve trying to parse with an HTML library, but keep it simple for now.
        # Check for the required wrapper tags strictly if the prompt enforces them.
        # If parse_agent_response extracts the content, this check might be redundant here,
        # but it's a safeguard if the raw response is passed directly.
        # if "<solar.html>" not in action or "</solar.html>" not in action:
        #     logger.warning("Validation warning: Action missing <solar.html> wrapper tags.")
            # return False # Make it strict?

        return True

    def step(self, action: str) -> Tuple[str, bool]:
        """
        Executes one step of the refinement process:
        1. Saves the generated HTML (`action` - potentially pre-parsed).
        2. Renders it using a real headless browser via `render_and_capture`.
        3. Captures screenshot and browser logs.
        4. Performs intermediate LLM evaluation on the screenshot via `encode_file_inline_data_gemini` and the LLM.
        5. Formats feedback (eval + logs) via `format_feedback_message`.
        6. Returns the feedback as the new state and terminal status (False unless error/max steps).

        Args:
            action (str): The HTML/JS code generated by the agent (potentially pre-parsed to remove tags).

        Returns:
            Tuple[str, bool]: (feedback_string, is_terminal)
        """
        self.current_turn += 1
        logger.info(f"--- Solar Env Step {self.current_turn}/{self._max_steps} ---")

        # 1. Save the generated HTML (action should be the content already)
        html_to_save = action # Assume action is the relevant code block passed from run_attempt
        self.current_html_path = self._get_path(self.HTML_FILENAME, self.current_turn)
        try:
            with open(self.current_html_path, "w", encoding='utf-8') as f:
                f.write(html_to_save)
            logger.info(f"Saved generated HTML to: {self.current_html_path}")
        except Exception as e:
            logger.error(f"Failed to save HTML file: {e}", exc_info=True)
            self.last_feedback_text = f"CRITICAL ERROR: Failed to save the generated HTML code. Cannot proceed.\nDetails: {e}"
            return self.last_feedback_text, True # Terminal failure if we can't even save the file

        # 2. Render, Capture Screenshot & Logs using the REAL function
        self.current_screenshot_path = self._get_path(self.SCREENSHOT_FILENAME, self.current_turn)
        self.current_browser_log_path = self._get_path(self.LOG_FILENAME, self.current_turn)

        # Call the imported function from browser_utils
        # It returns (success_bool, logs_or_error_string)
        self.last_render_success, browser_logs_or_error = render_and_capture(
            self.current_html_path,
            self.current_screenshot_path,
            self.current_browser_log_path
            # browser_type can be added if needed, defaults to chrome in browser_utils
        )

        # 3. Perform Intermediate Evaluation (only if rendering succeeded)
        intermediate_eval_response = "Evaluation Skipped: Rendering or screenshot capture failed."
        if self.last_render_success:
            # Check screenshot path validity again before encoding
            if self.current_screenshot_path and os.path.exists(self.current_screenshot_path) and os.path.getsize(self.current_screenshot_path) > 0:
                logger.info("Performing intermediate evaluation on screenshot...")
                # Call the imported function from browser_utils
                screenshot_data = encode_file_inline_data_gemini(self.current_screenshot_path)

                if screenshot_data:
                    # Construct payload for multimodal LLM
                    eval_payload = [
                        {"role": "user", "parts": [
                            screenshot_data["source"], # The dict returned by encode_file...
                            {"text": self._intermediate_eval_prompt}
                        ]}
                    ]
                    try:
                        # Call the multimodal generation method of the LLM interface
                        raw_response = self._intermediate_eval_llm.generate_content_multimodal(eval_payload)
                        if raw_response:
                            intermediate_eval_response = raw_response # Use the raw response directly as feedback text
                            logger.info(f"Intermediate eval response received (truncated): {intermediate_eval_response[:150]}...")
                        else:
                            logger.warning("Intermediate evaluator LLM returned no response.")
                            intermediate_eval_response = "Evaluation Error: Intermediate evaluator LLM returned an empty response."
                    except AttributeError:
                         logger.error("LLM Interface provided does not support 'generate_content_multimodal'.", exc_info=True)
                         intermediate_eval_response = "Evaluation Error: LLM Interface lacks the required multimodal capability."
                         # This might be a critical setup error, consider making it terminal?
                         # return format_feedback_message(intermediate_eval_response, browser_logs_or_error), True
                    except Exception as e:
                        logger.error(f"Error during intermediate LLM evaluation call: {e}", exc_info=True)
                        intermediate_eval_response = f"Evaluation Error: Exception during intermediate API call - {e}"
                else:
                    logger.warning("Could not encode screenshot for intermediate evaluation.")
                    intermediate_eval_response = "Evaluation Error: Failed to encode the captured screenshot."
            else:
                logger.warning(f"Screenshot file missing or empty ({self.current_screenshot_path}), skipping intermediate evaluation.")
                intermediate_eval_response = "Evaluation Skipped: Screenshot file missing or empty after capture attempt."
        else:
             logger.warning(f"Rendering or capture failed for turn {self.current_turn}. Intermediate evaluation skipped.")
             # browser_logs_or_error already contains the error message from render_and_capture

        # 4. Format Feedback using the REAL function
        # Pass the potentially error-containing message from render_and_capture if it failed
        self.last_feedback_text = format_feedback_message(intermediate_eval_response, browser_logs_or_error)
        logger.debug(f"Formatted feedback for next turn (truncated):\n{self.last_feedback_text[:500]}...")

        # 5. Return state and terminal status
        # Terminal status is determined by the main loop (max steps) or critical errors above.
        # Rendering failure is not treated as terminal here, allowing the agent to try fixing it.
        is_terminal = False # Default to non-terminal for a regular step
        return self.last_feedback_text, is_terminal

    def get_final_eval_input(self) -> str:
        """
        Returns the path to the *final* screenshot generated in the last successful step.
        """
        # Check if the last render was successful AND the path exists and is non-empty
        if self.last_render_success and self.current_screenshot_path and os.path.exists(self.current_screenshot_path) and os.path.getsize(self.current_screenshot_path) > 0:
            logger.info(f"Providing final screenshot path for evaluation: {self.current_screenshot_path}")
            return self.current_screenshot_path
        else:
            logger.warning("Final evaluation input requested, but last render failed or screenshot missing/empty.")
            # Return a specific indicator string or the potentially invalid path
            # The Evaluator must handle this case.
            return self.current_screenshot_path if self.current_screenshot_path else "SCREENSHOT_UNAVAILABLE"

    # --- Other Required BaseEnvironment Methods ---

    def check_goal_achieved(self) -> bool:
        """Goal achievement is determined by the final multimodal evaluation, not checkable deterministically here."""
        return False # Final score is the indicator

    def assess_intermediate_status(self) -> Any:
        """Intermediate status/regression tracking is currently not implemented for this visual task."""
        # Could potentially parse intermediate eval scores if they were numeric, but current setup uses text feedback.
        return None # Disabled

    def get_agent_player_mark(self) -> str | None:
        """Not applicable for this environment."""
        return None

    def get_state(self) -> str:
        """
        Returns a representation of the current state. For this env,
        the feedback from the last turn is the most relevant state component for the agent.
        """
        return self.last_feedback_text

    def get_goal(self) -> str:
        """Returns the main goal description."""
        return self._goal