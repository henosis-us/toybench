import logging
from llm_interface import LLMInterface
from utils import parse_llm_score

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, evaluator_llm: LLMInterface, task_eval_prompt_template: str):
        self.llm = evaluator_llm
        self.eval_prompt_template = task_eval_prompt_template
        # Stores status from previous turn for regression check {attempt_id: status}
        self.previous_statuses = {}

    def evaluate_final_outcome(self, final_eval_input: str) -> tuple[int, str]:
        """
        Uses the Evaluator LLM to score the final outcome based on the task prompt.
        Returns score (1, 2, or 3) and raw response text.
        Returns score 1 on failure to parse.
        """
        prompt = self.eval_prompt_template.format(final_outcome=final_eval_input)
        logger.info("Requesting final evaluation from LLM.")
        logger.debug(f"Final Evaluation Prompt: {prompt}")

        response_text = self.llm.evaluate_outcome(prompt)

        if response_text is None:
            logger.error("Failed to get response from evaluator LLM.")
            return 1, "Evaluator LLM failed to respond." # Fail score

        logger.info(f"Received final evaluation response: {response_text[:100]}...")
        score = parse_llm_score(response_text)

        if score is None:
            logger.warning(f"Could not parse score from evaluation response: {response_text}")
            return 1, response_text # Fail score if parsing fails

        logger.info(f"Parsed final score: {score}")
        return score, response_text

    def track_regression(self, attempt_id: int, current_status: any, turn: int) -> bool:
        """
        Compares current status to the previous status for the same attempt.
        Logs and returns True if regression is detected.
        Requires that status objects are comparable (e.g., via >, < or specific logic).
        """
        if current_status is None: # Cannot assess regression if no status
            return False

        last_status = self.previous_statuses.get(attempt_id)
        regression_detected = False

        if last_status is not None and turn > 0:
            # Define "worse" based on status type
            # This is task-dependent and relies on assess_intermediate_status() output
            try:
                # Simple comparison assumes higher value is better
                if current_status < last_status:
                    regression_detected = True
                    logger.warning(f"Regression detected! Attempt {attempt_id}, Turn {turn}: Status changed from {last_status} to {current_status}")
            except TypeError:
                # Handle cases where comparison isn't straightforward (e.g., complex objects)
                # Add task-specific comparison logic here if needed
                 logger.debug(f"Cannot directly compare statuses for regression: {last_status} vs {current_status}")
                 # Example for TTT tuple status (winner, is_draw):
                 # A regression could be moving from (None, False) [ongoing] to ('O', False) [loss for X]
                 # if last_status == (None, False) and current_status[0] is not None and current_status[0] != 'TARGET_PLAYER':
                 #     regression_detected = True # Simplified example
                 pass # Default: no regression if comparison fails

        # Store current status for next turn's comparison
        self.previous_statuses[attempt_id] = current_status
        return regression_detected

    def reset_attempt_tracking(self, attempt_id: int):
        """Clears the stored status for a given attempt ID."""
        if attempt_id in self.previous_statuses:
            del self.previous_statuses[attempt_id]
