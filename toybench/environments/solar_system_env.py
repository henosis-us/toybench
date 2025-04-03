# environments/solar_system_env.py

import logging
import os
import base64
import mimetypes
import re
import time
import json
import subprocess # For potential cleanup if needed
from typing import Tuple, Any, Dict

# Assuming browser utilities are moved here or imported
# For this example, let's define simplified versions or placeholders
# In a real implementation, import these from utils.browser_utils
# from utils.browser_utils import render_and_capture, encode_file_inline_data_gemini, format_feedback_message, capture_browser_logs

# Placeholder/Simplified Helper Functions (Replace with actual imports/implementations)
# ==============================================================================
# --- Assume these are imported from utils.browser_utils ---
def setup_browser_placeholder():
    """Placeholder: Sets up Selenium Chrome browser."""
    # In real implementation: use selenium webdriver setup
    print("[Placeholder] Setting up browser...")
    # Needs to return a driver-like object or handle internally
    class MockDriver:
        def save_screenshot(self, path): print(f"[Placeholder] Saving screenshot to {path}")
        def get_log(self, log_type): return [{'level': 'INFO', 'message': '[Placeholder] Browser log message.', 'timestamp': time.time()*1000}]
        def execute_script(self, script): print("[Placeholder] Executing script.") # Needed for console log capture
        def get(self, url): print(f"[Placeholder] Navigating to {url}")
        def quit(self): print("[Placeholder] Quitting browser.")
    return MockDriver()

def capture_browser_logs_placeholder(driver) -> str:
    """Placeholder for capturing browser logs."""
    logs = []
    try:
        # Get browser console logs
        browser_logs = driver.get_log('browser')
        for log in browser_logs:
            level = log.get('level', 'UNKNOWN')
            message = log.get('message', '')
            # Simple formatting
            logs.append(f"[{level}] {message}")
        if not logs:
            logs.append("[INFO] No console logs captured.")
    except Exception as e:
        logs.append(f"Error capturing logs: {str(e)}")
    return "\n".join(logs)

def render_and_capture_placeholder(html_file_path: str, screenshot_path: str, browser_log_path: str) -> tuple[bool, str]:
    """Placeholder: Simulates rendering HTML, capturing screenshot and logs."""
    print(f"[Placeholder] Rendering {html_file_path}...")
    driver = None
    try:
        driver = setup_browser_placeholder()
        # Simulate loading the page
        file_url = f"file:///{os.path.abspath(html_file_path).replace(os.sep, '/')}"
        driver.get(file_url)
        time.sleep(0.1) # Simulate load time

        # Simulate capturing screenshot
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        # Create a dummy file for the screenshot
        with open(screenshot_path, 'w') as f:
            f.write("dummy_screenshot_data")
        print(f"[Placeholder] Captured dummy screenshot: {screenshot_path}")

        # Simulate capturing logs
        logs = capture_browser_logs_placeholder(driver)
        with open(browser_log_path, 'w', encoding='utf-8') as f:
            f.write(logs)
        print(f"[Placeholder] Captured browser logs: {browser_log_path}")

        # Simulate success
        render_success = True
        final_logs = logs

    except Exception as e:
        error_msg = f"Error during placeholder rendering: {str(e)}"
        print(error_msg)
        # Create empty files on error
        try: os.makedirs(os.path.dirname(screenshot_path), exist_ok=True); open(screenshot_path, 'w').close()
        except: pass
        try: os.makedirs(os.path.dirname(browser_log_path), exist_ok=True); open(browser_log_path, 'w').close()
        except: pass
        render_success = False
        final_logs = error_msg
    finally:
        if driver:
            driver.quit()

    return render_success, final_logs

def encode_file_base64(file_path: str) -> str | None:
    """Encodes a file to Base64 string."""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"File not found for encoding: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding file {file_path}: {e}")
        return None

def encode_file_inline_data_gemini(file_path: str) -> dict | None:
    """Encodes a file to Base64 inline data format for Gemini API."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'image/png' # Default fallback
    base64_data = encode_file_base64(file_path)
    if base64_data is None:
        return None
    return {
        "type": "image",
        "source": {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64_data
            }
        }
    }

def format_feedback_message(eval_response: str, browser_logs: str) -> str:
    """Formats evaluation and browser logs into a structured feedback message."""
    return f"""EVALUATION FEEDBACK:
{eval_response}

BROWSER CONSOLE LOGS:
{browser_logs}
---
Based on the above feedback and the original goal, please generate the *complete* and *corrected* HTML/JS code for the solar system simulation, wrapped in <solar.html>...</solar.html> tags. Ensure all necessary HTML structure, CSS, and JavaScript (including Three.js setup and logic) are present in your response.
""".strip()
# ==============================================================================


from .base_env import BaseEnvironment
# Import the LLM interface
from llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class SolarSystemEnv(BaseEnvironment):
    """
    Environment for the Solar System HTML generation task.

    This environment manages the state (current HTML, screenshot, logs),
    handles the interaction loop involving browser rendering and intermediate
    LLM evaluation, and prepares input for the final evaluation.
    """
    # Define file naming conventions relative to the attempt's output directory
    HTML_FILENAME = "solar_iteration_{turn}.html"
    SCREENSHOT_FILENAME = "solar_screenshot_iteration_{turn}.png"
    LOG_FILENAME = "browser_logs_iteration_{turn}.txt"
    TEMP_EVAL_FILENAME = "intermediate_eval_response_{turn}.json" # For intermediate eval results

    def __init__(self,
                 goal_description: str,
                 intermediate_eval_prompt: str, # Content of evalprompt.txt
                 intermediate_eval_llm: LLMInterface,
                 output_dir_base: str, # Base directory for THIS attempt's files
                 max_steps: int):
        """
        Initializes the Solar System Environment.

        Args:
            goal_description (str): The main goal prompt content (from solarsystemprompt.txt).
            intermediate_eval_prompt (str): The prompt template for intermediate visual evaluation.
            intermediate_eval_llm (LLMInterface): LLM interface for intermediate evaluations.
            output_dir_base (str): The specific directory to store files for the current attempt.
            max_steps (int): Maximum number of refinement turns allowed.
        """
        self._goal = goal_description
        self._intermediate_eval_prompt = intermediate_eval_prompt
        self._intermediate_eval_llm = intermediate_eval_llm
        self._output_dir = output_dir_base
        self._max_steps = max_steps # Track internally if needed

        self.current_turn = 0
        self.current_html_path = None
        self.current_screenshot_path = None
        self.current_browser_log_path = None
        self.last_feedback_text = "No feedback yet. Generate the initial HTML based on the goal."
        self.last_render_success = False

        # Ensure the output directory exists
        os.makedirs(self._output_dir, exist_ok=True)
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
        # Be cautious if running multiple attempts in parallel to the same base dir structure
        logger.info("SolarSystemEnv Reset. Ready for first generation.")

        # Initial state for the agent is simply the goal description
        return self._goal # The prompt template will combine this with feedback later

    def get_prompt_context(self) -> Dict[str, Any]:
        """
        Returns the context dictionary for formatting the agent's prompt.
        Includes the main goal and the feedback from the last step.
        """
        return {
            "goal": self._goal,
            "current_state": self.last_feedback_text # 'current_state' maps to feedback
        }

    def validate_action(self, action: str) -> bool:
        """
        Validates the generated HTML code (the 'action').
        Checks if it's non-empty and potentially contains expected structure.
        """
        if not action or not action.strip():
            logger.warning("Validation failed: Received empty action (HTML code).")
            return False

        # Basic check: Does it look like HTML? (Very naive)
        if not ("<html" in action.lower() and "</html" in action.lower()):
             logger.warning("Validation failed: Action doesn't contain <html> tags.")
             # Allow it for now, maybe the agent is outputting partial code or just JS
             # return False

        # Check for the required wrapper tags if the prompt enforces them
        if "<solar.html>" not in action or "</solar.html>" not in action:
            logger.warning("Validation warning: Action missing <solar.html> wrapper tags.")
            # Depending on strictness, could return False here. Let's allow it for now.

        return True

    def step(self, action: str) -> Tuple[str, bool]:
        """
        Executes one step of the refinement process:
        1. Saves the generated HTML (`action`).
        2. Renders it using a headless browser.
        3. Captures screenshot and browser logs.
        4. Performs intermediate LLM evaluation on the screenshot.
        5. Formats feedback (eval + logs).
        6. Returns the feedback as the new state and terminal status (False unless error/max steps).

        Args:
            action (str): The HTML/JS code generated by the agent.

        Returns:
            Tuple[str, bool]: (feedback_string, is_terminal)
        """
        self.current_turn += 1
        logger.info(f"--- Solar Env Step {self.current_turn}/{self._max_steps} ---")

        # 1. Save the generated HTML
        # Extract content within <solar.html> tags if present
        html_to_save = action.strip()
        match = re.search(r"<solar\.html>(.*?)</solar\.html>", action, re.DOTALL | re.IGNORECASE)
        if match:
            html_to_save = match.group(1).strip()
            logger.debug("Extracted content within <solar.html> tags.")
        else:
            logger.warning("Could not find <solar.html> tags, saving entire action content.")

        self.current_html_path = self._get_path(self.HTML_FILENAME, self.current_turn)
        try:
            with open(self.current_html_path, "w", encoding='utf-8') as f:
                f.write(html_to_save)
            logger.info(f"Saved generated HTML to: {self.current_html_path}")
        except Exception as e:
            logger.error(f"Failed to save HTML file: {e}", exc_info=True)
            self.last_feedback_text = f"Error: Failed to save the generated HTML code. Please try again.\nDetails: {e}"
            return self.last_feedback_text, True # Terminal failure

        # 2. Render, Capture Screenshot & Logs
        self.current_screenshot_path = self._get_path(self.SCREENSHOT_FILENAME, self.current_turn)
        self.current_browser_log_path = self._get_path(self.LOG_FILENAME, self.current_turn)

        # Use the actual rendering function (imported or defined)
        self.last_render_success, browser_logs = render_and_capture_placeholder(
            self.current_html_path,
            self.current_screenshot_path,
            self.current_browser_log_path
        )

        if not self.last_render_success:
            logger.warning(f"Rendering or capture failed for turn {self.current_turn}.")
            # Feedback includes only the browser logs/error message
            self.last_feedback_text = format_feedback_message("Evaluation skipped: Rendering/Screenshot failed.", browser_logs)
            # Decide if rendering failure is terminal. Let's allow recovery for now.
            # return self.last_feedback_text, True # Make it terminal?
            return self.last_feedback_text, False # Allow agent to try fixing render errors

        # 3. Perform Intermediate Evaluation (if rendering succeeded)
        intermediate_eval_response = "Evaluation Error: Could not process screenshot or call LLM."
        if os.path.exists(self.current_screenshot_path) and os.path.getsize(self.current_screenshot_path) > 0:
            logger.info("Performing intermediate evaluation on screenshot...")
            screenshot_data = encode_file_inline_data_gemini(self.current_screenshot_path)

            if screenshot_data:
                eval_payload = [
                    {"role": "user", "parts": [
                        screenshot_data["source"], # Gemini expects parts content directly
                        {"text": self._intermediate_eval_prompt}
                    ]}
                ]
                try:
                    # Use the LLM interface's multimodal method
                    # Note: Adjust method name/params based on llm_interface.py definition
                    raw_response = self._intermediate_eval_llm.generate_content_multimodal(eval_payload) # Assuming this method exists

                    if raw_response:
                        # Basic extraction - assumes response is just text for eval
                        # In reality, might need JSON parsing if Gemini returns structured multimodal response
                        intermediate_eval_response = raw_response # Use the raw response for now
                        logger.info(f"Intermediate eval response received: {intermediate_eval_response[:100]}...")
                    else:
                        logger.warning("Intermediate evaluator LLM returned no response.")
                        intermediate_eval_response = "Evaluation Error: Intermediate evaluator LLM returned no response."

                except AttributeError:
                     logger.error("LLM Interface does not support required 'generate_content_multimodal' method.", exc_info=True)
                     intermediate_eval_response = "Evaluation Error: LLM Interface lacks multimodal capability."
                except Exception as e:
                    logger.error(f"Error during intermediate LLM evaluation call: {e}", exc_info=True)
                    intermediate_eval_response = f"Evaluation Error: Exception during API call - {e}"
            else:
                logger.warning("Could not encode screenshot for intermediate evaluation.")
                intermediate_eval_response = "Evaluation Error: Failed to encode screenshot."
        else:
            logger.warning("Screenshot file missing or empty, skipping intermediate evaluation.")
            intermediate_eval_response = "Evaluation Skipped: Screenshot file missing or empty."

        # 4. Format Feedback
        self.last_feedback_text = format_feedback_message(intermediate_eval_response, browser_logs)
        logger.debug(f"Formatted feedback for next turn:\n{self.last_feedback_text}")

        # 5. Return state and terminal status
        # Terminal only if max steps reached (checked by runner) or critical unrecoverable error happened (handled above)
        return self.last_feedback_text, False

    def get_final_eval_input(self) -> str:
        """
        Returns the path to the *final* screenshot generated in the last successful step.
        If the last step failed to render, it might return None or the path to an empty/failed file.
        """
        if self.last_render_success and self.current_screenshot_path and os.path.exists(self.current_screenshot_path):
            logger.info(f"Providing final screenshot path for evaluation: {self.current_screenshot_path}")
            return self.current_screenshot_path
        else:
            logger.warning("Final evaluation input requested, but last render failed or screenshot missing.")
            # Return the path anyway, evaluator needs to handle missing file
            return self.current_screenshot_path if self.current_screenshot_path else "SCREENSHOT_UNAVAILABLE"

    # --- Other Required BaseEnvironment Methods ---

    def check_goal_achieved(self) -> bool:
        """Goal achievement is determined by the final multimodal evaluation, not checkable here."""
        return False

    def assess_intermediate_status(self) -> Any:
        """Intermediate status/regression tracking is not the primary focus for this task."""
        return None # Disabled

    def get_agent_player_mark(self) -> str | None:
        """Not applicable for this environment."""
        return None

    def get_state(self) -> str:
        """
        Returns a representation of the current state. For this env,
        the most relevant 'state' for the agent is the feedback from the last turn.
        """
        # Could also return the path to the current HTML file, but feedback is more direct.
        return self.last_feedback_text

    def get_goal(self) -> str:
        """Returns the main goal description."""
        return self._goal
