# toybench_cli.py
# VERSION MODIFIED TO HANDLE SOLAR_GEN CONVERSATIONALLY + Specific Tag Error Handling + OpenAI provider
# VERSION MODIFIED TO CORRECTLY DETERMINE EVALUATOR PROVIDER INDEPENDENTLY
# VERSION MODIFIED TO ADD XAI GROK PROVIDER SUPPORT WITH REASONING EFFORT
# FIXED: API key name mismatch for Grok provider (mapped 'grok' to 'xai_api_key')
# ADDED: Quality Compute provider support with hardcoded URL in llm_interface.py
# ADDED: Anthropic Claude support
# UPDATED: Passing 'task_name' to calculate_metrics for task-specific metrics like Differentiated Score in SolarGen.
# ADDED: --thinking and --thinking_budget arguments for Anthropic extended thinking support

import argparse
import logging
import os
import json
import time
import re  # Import regex for parsing completion signal/action
from config import load_config
from utils import setup_logging, create_output_dir, parse_llm_score  # Use updated parse_llm_score if needed

# Import Interfaces and Base Classes
from llm_interface import LLMInterface, GeminiInterface, OpenAIInterface, GrokInterface, QualityComputeInterface, AnthropicInterface  # Added AnthropicInterface
from environments.base_env import BaseEnvironment
from evaluation import Evaluator
from reporting import calculate_metrics, format_report, save_results

# Import Specific Environments
from environments.file_system_env import FileSystemEnv
from environments.tic_tac_toe_env import TicTacToeEnv
from environments.solar_system_env import SolarSystemEnv

# Configure logging early
logger = logging.getLogger(__name__)

# --- Task Definitions (Load from files) ---
def load_task_prompts(task_name: str, task_dir: str) -> dict:
    """Loads goal description, generation prompt, intermediate eval, and final eval prompt for a task."""
    task_path = os.path.join(task_dir, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"Task directory not found: {task_path}")
    prompts = {}
    files_to_load = {
        'goal_description':           f"{task_name}_goal.txt",
        'generate_template':          f"{task_name}_generate.txt",
        'intermediate_eval_template': f"{task_name}_intermediate_eval.txt",
        'finaleval_template':         f"{task_name}_finaleval.txt",
    }
    for key, filename in files_to_load.items():
        file_path = os.path.join(task_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompts[key] = f.read().strip()
                logger.debug(f"Loaded prompt '{key}' from {filename}")
            except Exception as e:
                logger.error(f"Error loading prompt file {filename}: {e}", exc_info=True)
                prompts[key] = None
        else:
            prompts[key] = None
            # Intermediate eval is optional for some tasks
            if key in ['goal_description', 'generate_template', 'finaleval_template'] and key != 'intermediate_eval_template':
                # Only warn if core or final eval prompts are missing (except for TTT/FS final which are deterministic)
                if not (key == 'finaleval_template' and task_name in ["tic_tac_toe", "file_system"]):
                    logger.warning(f"Potentially required prompt file not found: {file_path}")

    # Validate essential prompts
    if prompts['generate_template'] is None:
        raise FileNotFoundError(f"Essential generation prompt missing: {files_to_load['generate_template']}")
    if prompts['finaleval_template'] is None and task_name not in ["tic_tac_toe", "file_system"]:
        logger.error(f"Final evaluation prompt missing for LLM-evaluated task {task_name}. Using default.")
        prompts['finaleval_template'] = (
            "Evaluate the final state based on the goal.\n"
            "Goal: {goal}\nFinal State: {final_outcome}\n"
            "Rate 1-3 (3=Success): <rating>X</rating>"
        )
    elif prompts['finaleval_template'] is None and task_name in ["tic_tac_toe", "file_system"]:
        logger.info(f"Final evaluation prompt not needed/found for task {task_name} (using deterministic eval).")

    logger.info(f"Loaded prompts for task: {task_name}")
    return prompts

# --- Environment Factory ---
def get_environment(task_name: str,
                    goal: str,
                    prompts: dict,
                    evaluator_llm: LLMInterface | None,  # Intermediate evaluator for Solar
                    output_dir: str,
                    max_rounds: int) -> BaseEnvironment:
    """Factory function to instantiate the correct environment."""
    logger.info(f"Creating environment for task: {task_name}")
    if task_name == "file_system":
        return FileSystemEnv(goal)
    elif task_name == "tic_tac_toe":
        return TicTacToeEnv(goal)
    elif task_name == "solar_gen":
        intermediate_eval_prompt = prompts.get('intermediate_eval_template')
        if not intermediate_eval_prompt:
            raise ValueError("SolarSystemEnv requires an intermediate evaluation prompt (_intermediate_eval.txt).")
        if not evaluator_llm:
            raise ValueError("SolarSystemEnv requires an intermediate evaluator LLM instance.")
        try:
            # Ensure BROWSER_UTILS_AVAILABLE is checked internally by SolarSystemEnv constructor
            return SolarSystemEnv(
                goal_description=goal,
                intermediate_eval_prompt=intermediate_eval_prompt,
                intermediate_eval_llm=evaluator_llm,  # This is the intermediate evaluator
                output_dir_base=output_dir,  # Pass attempt-specific dir
                max_steps=max_rounds
            )
        except NameError as e:
            logger.error(f"Failed to instantiate SolarSystemEnv. Is 'selenium' installed and browser_utils available? Error: {e}", exc_info=True)
            raise ImportError("Failed to initialize SolarSystemEnv. Check dependencies (selenium, browser_utils).") from e
        except ImportError as e:  # Catch import error raised by SolarSystemEnv itself
            logger.error(f"Failed to initialize SolarSystemEnv due to missing browser utilities: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing SolarSystemEnv: {e}", exc_info=True)
            raise
    else:
        raise ValueError(f"Unknown task name: {task_name}")

# --- LLM Provider Factory ---
def get_llm_interface(provider_name: str, api_key: str, model_name: str, thinking_enabled: bool = False, thinking_budget: int = 16000) -> LLMInterface:
    """Factory function to instantiate the correct LLM interface."""
    p = provider_name.lower()
    if p == "gemini":
        if not api_key:
            raise ValueError("Gemini API Key is required for GeminiInterface.")
        return GeminiInterface(api_key=api_key, model_name=model_name)
    elif p == "openai":
        if not api_key:
            raise ValueError("OpenAI API Key is required for OpenAIInterface.")
        return OpenAIInterface(api_key=api_key, model_name=model_name)
    elif p == "grok":
        if not api_key:
            raise ValueError("xAI API Key is required for GrokInterface.")
        return GrokInterface(api_key=api_key, model_name=model_name, reasoning_effort="low")  # Reasoning effort defaulted, can be overridden if needed
    elif p == "quality_compute":
        if not api_key:
            raise ValueError("Quality Compute API Key is required for QualityComputeInterface.")
        return QualityComputeInterface(api_key=api_key, model_name=model_name)
    elif p == "anthropic":
        if not api_key:
            raise ValueError("Anthropic API Key is required for AnthropicInterface.")
        return AnthropicInterface(api_key=api_key, model_name=model_name, thinking_enabled=thinking_enabled, thinking_budget=thinking_budget)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")

# --- Action Parsing Helper ---
def parse_agent_response(response_text: str, task_name: str) -> tuple[str | None, bool]:
    """Parses agent response based on the task type.
    - For 'solar_gen', extracts the code block.
    - For other tasks (TTT/FS), uses original logic (```action or first line).
    Returns (command_or_None, task_complete_bool).
    """
    if response_text is None:
        return None, False

    # --- Task-Specific Parsing for Solar ---
    if task_name == "solar_gen":
        # Use regex to find content within <solar.html>...</solar.html> tags, case-insensitive and multiline
        match = re.search(r"<solar\.html>(.*?)</solar\.html>", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            code_content = match.group(1).strip()
            logger.debug("Parsed action (solar_gen): Extracted content from <solar.html> tags.")
            # For solar_gen, completion isn't signaled via text, it's determined by rounds
            return code_content, False
        else:
            # Check for incomplete tags specifically
            if "<solar.html>" in response_text and "</solar.html>" not in response_text:
                logger.error("Solar task specific error: Agent response missing closing </solar.html> tag.")
                # Return None to indicate parsing failure
                return None, False  # Treat missing closing tag as parse failure
            elif "</solar.html>" in response_text and "<solar.html>" not in response_text:
                logger.error("Solar task specific error: Agent response missing opening <solar.html> tag.")
                return None, False  # Treat missing opening tag as parse failure
            else:
                # Log if the tags are completely absent
                logger.warning("Could not find or parse <solar.html>...</solar.html> tags in response for solar_gen.")
                # Return None, let the calling code handle the lack of action
                return None, False

    # --- Original Parsing Logic (for TTT, FS, Generic) ---
    task_complete = "TASK_COMPLETE" in response_text.upper()

    # Try finding ```action block first
    action_match = re.search(r"```action\n(.*?)\n```", response_text, re.DOTALL | re.IGNORECASE)
    if action_match:
        command = action_match.group(1).strip()
        logger.debug(f"Parsed command (block): '{command}'")
        # Check if the *only* thing in the block is TASK_COMPLETE
        if "TASK_COMPLETE" in command.upper() and len(command.split()) == 1:
            logger.debug("Parsed signal (TASK_COMPLETE in action block).")
            return None, True  # Signal completion, no command
        return command, task_complete  # Return command, respect completion signal elsewhere if present

    # If no action block, check the first line heuristic
    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        # Check if the first line *only* contains TASK_COMPLETE
        if "TASK_COMPLETE" in first_line.upper() and len(first_line.split()) == 1:
            logger.debug("Parsed signal (TASK_COMPLETE line).")
            return None, True  # Signal completion

        # Heuristic: Check if first word looks like a known command
        common_cmds = ["ls", "cd", "mkdir", "touch", "rm", "pwd", "mv", "place", "cat", "cp", "echo"]
        first_word = first_line.split(maxsplit=1)[0].lower() if first_line else ""
        if first_word in common_cmds:
            logger.debug(f"Parsed command (first line heuristic): '{first_line}'")
            # Return the whole first line as the command, respect separate TASK_COMPLETE signal
            return first_line, task_complete

    # If no command parsed by either method, return None for command
    logger.debug(f"No command parsed from response (Original Logic Path). Signal status: {task_complete}. Response: {response_text[:100]}...")
    return None, task_complete

def run_attempt(attempt_id: int,
                env: BaseEnvironment,
                agent: LLMInterface,
                evaluator: Evaluator,  # This is the FINAL evaluator
                max_rounds: int,
                prompts: dict,
                task_name: str,
                output_dir: str) -> dict:
    """Runs a single attempt, handling specific logic for TTT, FS, and Solar."""
    logger.info(f"--- Starting Attempt {attempt_id + 1} (Task: {task_name}, Max Rounds/Steps: {max_rounds}) ---")

    # Note: The Evaluator passed here is for FINAL evaluation.
    # SolarSystemEnv receives its intermediate evaluator LLM during its own initialization.
    evaluator.reset_attempt_tracking(attempt_id)  # Reset regression tracking for this attempt

    start_time = time.time()
    initial_state_desc = env.reset()
    logger.debug(
        f"Environment reset. Initial state description (truncated): {str(initial_state_desc)[:200]}..."
    )

    history = []  # Detailed turn-by-turn log for this attempt
    conversation_history = []  # For conversational LLM APIs
    regressions_detected_in_attempt = False
    premature_failure = False
    failure_reason = ""
    final_eval_input_summary = "N/A"  # Summary of what was passed to final eval
    agent_made_any_invalid_move_in_attempt = False  # Specific to TTT

    # Determine agent player mark (for TTT)
    agent_player_mark = env.get_agent_player_mark()  # Returns 'X' for TTT, None otherwise

    turn_count = 0
    is_ttt = isinstance(env, TicTacToeEnv)
    is_fs = isinstance(env, FileSystemEnv)
    is_solar = isinstance(env, SolarSystemEnv)

    # Max steps/rounds logic clarification
    if is_ttt:
        logger.info(
            f"Task '{task_name}' runs until game completion or error. --rounds ({max_rounds}) ignored."
        )
        # Set a very high internal limit for safety, but shouldn't be needed
        effective_max_rounds = 100
    elif is_fs:
        logger.info(
            f"Task '{task_name}' runs until completion signal, error, or max_rounds. --rounds ({max_rounds}) acts as step limit."
        )
        effective_max_rounds = max_rounds
    elif is_solar:
        logger.info(
            f"Task '{task_name}' runs for exactly {max_rounds} refinement rounds (steps)."
        )
        effective_max_rounds = max_rounds
    else:
        # Generic task, respect max_rounds
        logger.info(
            f"Task '{task_name}' runs for max {max_rounds} steps or until completion signal/error."
        )
        effective_max_rounds = max_rounds

    # Determine if conversational API should be used based on task
    # Solar requires back-and-forth feedback, FS benefits from context. TTT is stateless per turn.
    use_conversational_api = is_fs or is_solar
    if use_conversational_api:
        logger.info(
            f"Using conversational API path for agent in task '{task_name}'."
        )
    else:
        logger.info(
            f"Using single-shot API path for agent in task '{task_name}'."
        )

    # --- Initial Prompt Construction ---
    try:
        # Get initial context from environment (state, available commands, etc.)
        initial_context = env.get_prompt_context()
        if 'goal' not in initial_context:  # Ensure goal is always present
            initial_context['goal'] = prompts.get(
                'goal_description', 'Goal not provided.'
            )

        template = prompts.get('generate_template')
        if not template:
            raise ValueError("Generate template prompt is missing.")

        # Check if all placeholders in the template are present in the context
        required_keys = set(re.findall(r"\{(\w+)\}", template))
        missing_keys = required_keys - set(initial_context.keys())
        if missing_keys:
            logger.warning(
                f"Initial prompt context missing keys required by template: {missing_keys}. "
                "Attempting to proceed with placeholders."
            )
            # Add placeholders for missing keys to avoid crashing format()
            for key in missing_keys:
                initial_context[key] = f"<{key}_placeholder>"

        generation_prompt_text = template.format(**initial_context)

        if use_conversational_api:
            # Start the conversation history
            conversation_history.append(
                {'role': 'user', 'parts': [generation_prompt_text]}
            )
            logger.debug(
                f"Initialized conversational history for {task_name}. First user message created."
            )
        else:
            # For non-conversational, this text will be used directly in the first call
            logger.debug(
                f"Initial Non-Conversational Prompt prepared for task {task_name}."
            )

    except KeyError as e:
        logger.error(
            f"Initial prompt formatting failed: Missing key {e} in context or template.",
            exc_info=True
        )
        premature_failure = True
        failure_reason = f"Initial prompt key error: {e}"
    except ValueError as e:
        logger.error(
            f"Initial prompt setup failed: {e}", exc_info=True
        )
        premature_failure = True
        failure_reason = f"Initial prompt value error: {e}"
    except Exception as e:
        logger.error(
            f"Unexpected error during initial prompt setup: {e}", exc_info=True
        )
        premature_failure = True
        failure_reason = f"Unexpected initial prompt error: {e}"

    # This variable will hold the result/feedback from the environment step
    env_feedback_or_result = ""

# --- Main Interaction Loop ---
    while not premature_failure:
        # Check termination conditions: max rounds/steps reached
        if turn_count >= effective_max_rounds:
            logger.warning(
                f"Reached maximum allowed rounds/steps ({effective_max_rounds}) for task '{task_name}'. "
                "Terminating attempt."
            )
            if not is_solar:  # Solar runs fixed rounds, reaching max isn't inherently a failure yet
                failure_reason = f"Reached max_rounds ({effective_max_rounds}) without success signal"
                # Don't set premature_failure here, let final eval happen, but break loop
            break

        current_turn_number = turn_count + 1
        log_prefix = f"Attempt {attempt_id + 1}, Round/Turn {current_turn_number}"
        if not is_ttt:  # TTT doesn't have a fixed round limit concept
            log_prefix += f"/{effective_max_rounds}"
        logger.info(f"--- {log_prefix} ---")

        # --- Prepare Turn Variables ---
        is_terminal_this_turn = False  # Did the env signal terminal state AFTER this turn?
        action_taken_this_turn = "N/A"  # What action was parsed/attempted
        action_valid_this_turn = False  # Was the action valid according to env.validate?
        step_error_this_turn = None  # Error string if something failed this turn
        new_state_desc_this_turn = "N/A"  # State description AFTER action
        turn_player = "Agent"  # Default player
        current_turn_feedback = ""  # Feedback generated by this turn's step

        # <<< NEW VARIABLES TO CAPTURE LLM USAGE >>> 
        token_usage_this_turn = None
        raw_api_response_this_turn = None

        # Get state description BEFORE the agent acts this turn
        prompt_context_before_action = env.get_prompt_context()
        if 'goal' not in prompt_context_before_action:  # Ensure goal is present
            prompt_context_before_action['goal'] = prompts.get('goal_description', '')
        state_before_action_desc = str(prompt_context_before_action.get('current_state', 'State Unavailable'))
        new_state_desc_this_turn = state_before_action_desc  # Initialize with current state

        # --- Tic Tac Toe: Handle Player Turns ---
        if is_ttt:
            current_env_player = getattr(env, 'current_player', agent_player_mark)
            is_agent_turn = (current_env_player == agent_player_mark)
            turn_player = current_env_player  # Log who's turn it actually is

            if not is_agent_turn:
                # If it's the opponent's turn, skip agent generation and processing
                logger.debug(f"Turn {current_turn_number}: Skipping agent action, it's Opponent ({turn_player})'s turn.")
                # Opponent move is handled later in the loop after logging agent's turn (if any)
                turn_count += 1  # Increment turn count even if opponent plays
                continue  # Skip to next iteration (or opponent move section)
            else:
                logger.debug(f"Turn {current_turn_number}: Agent ({turn_player})'s turn.")

        # --- Agent Action Generation ---
        agent_response_text = None
        try:
            if use_conversational_api:
                # Add previous turn's feedback as a user message IF the last message was from the model
                if turn_count > 0 and env_feedback_or_result:
                    if conversation_history and conversation_history[-1]['role'] == 'model':
                        conversation_history.append({'role': 'user', 'parts': [env_feedback_or_result]})
                        logger.debug("Appended environment feedback from previous turn to conversation history.")
                    elif not conversation_history:
                        logger.warning("Conversation history empty, cannot append feedback (should not happen after first turn).")
                    else:
                        logger.warning("Last message in history was not from 'model'. Skipping append of env feedback to avoid user->user sequence.")

                # Safety check for empty history
                if not conversation_history:
                    logger.error(f"Conversational history is unexpectedly empty before calling agent for {task_name}.")
                    raise ValueError("Cannot generate action with empty conversational history.")

                logger.debug(f"Calling conversational agent ({agent.model_name}) for {task_name}. History length: {len(conversation_history)}")
                (agent_response_text,
                 token_usage_this_turn,
                 raw_api_response_this_turn) = agent.generate_action_conversational(conversation_history)
            else:  # Use single-shot generation (Tic Tac Toe)
                # Format the prompt using the current context
                current_context = prompt_context_before_action
                template = prompts['generate_template']  # Already checked if None earlier
                # Ensure required keys are present (maybe redundant after initial check, but safe)
                required_keys = set(re.findall(r"\{(\w+)\}", template))
                missing_keys = required_keys - set(current_context.keys())
                if missing_keys:
                    logger.warning(f"Turn {current_turn_number}: Prompt context missing keys {missing_keys}. Using defaults.")
                    # Add defaults specifically needed by TTT template if missing
                    if is_ttt:
                        current_context.setdefault('last_invalid_action_feedback', '')
                    for key in missing_keys:
                        current_context.setdefault(key, f"<{key}_placeholder>")
                generation_prompt = template.format(**current_context)
                logger.debug(f"Agent Prompt (Turn {current_turn_number}, Task: {task_name}, Non-Conversational).")  # Truncate if too long?
                (agent_response_text,
                 token_usage_this_turn,
                 raw_api_response_this_turn) = agent.generate_action(generation_prompt)
        except KeyError as e:
            logger.error(f"Agent prompt formatting error during turn {current_turn_number}: Missing key {e}", exc_info=True)
            step_error_this_turn = f"Prompt format error: {e}"
            is_terminal_this_turn = True  # Treat prompt error as terminal for the attempt
            premature_failure = True
            failure_reason = step_error_this_turn
            action_taken_this_turn = 'AGENT_API_ERROR'  # Mark specific error type
        except ValueError as e:  # Catch specific errors like empty history
            logger.error(f"Agent generation error during turn {current_turn_number}: {e}", exc_info=True)
            step_error_this_turn = f"Agent value error: {e}"
            is_terminal_this_turn = True
            premature_failure = True
            failure_reason = step_error_this_turn
            action_taken_this_turn = 'AGENT_API_ERROR'  # Mark specific error type
        except Exception as e:  # Catch other LLM API errors
            logger.error(f"Agent generation failed during turn {current_turn_number}: {e}", exc_info=True)
            step_error_this_turn = f"Agent generation API error: {e}"
            is_terminal_this_turn = True
            premature_failure = True
            failure_reason = step_error_this_turn

        # Check if agent response is None (could happen due to API errors caught above or other issues)
        if agent_response_text is None and not premature_failure:
            logger.error(f"Agent ({agent.model_name}) returned None response for turn {current_turn_number}.")
            # --- ADDED LOGGING FOR RAW RESPONSE ---
            logger.error(f"Raw API Response when text was None: {raw_api_response_this_turn}")
            # --- END ADDED LOGGING ---
            step_error_this_turn = "Agent LLM returned no response"
            is_terminal_this_turn = True
            premature_failure = True
            failure_reason = step_error_this_turn
            action_taken_this_turn = 'AGENT_API_ERROR'  # Mark specific error type

        # If a failure occurred during generation, break the loop
        if premature_failure:
            break

        # --- <<< END LLM CALL SECTION >>> ---

        # Log raw response (no truncation)
        logger.debug(f"Raw Agent Response (Turn {current_turn_number}): {str(agent_response_text)}")  # No truncation

        # Add agent's response to conversational history if applicable
        if use_conversational_api:
            # Ensure agent_response_text is a string before appending
            if isinstance(agent_response_text, str):
                conversation_history.append({'role': 'model', 'parts': [agent_response_text]})
            else:
                # Handle the case where it's still None, maybe log a warning?
                logger.warning(f"Cannot append non-string agent response to conversational history (Type: {type(agent_response_text)}). Skipping append for turn {current_turn_number}.")

        # --- Parse Agent Response ---
        action_content, agent_signaled_completion = parse_agent_response(agent_response_text, task_name)

        improperly_closed_tags = False  # Flag for solar-specific tag error
        if action_content is not None:
            # Log parsed action (truncated if long, e.g., HTML code)
            log_action = action_content[:100] + ('...' if len(action_content) > 100 else '')
            action_taken_this_turn = log_action.replace('\n', '\\n')  # Store concise version for log
            logger.info(f"Agent proposed action/output: '{action_taken_this_turn}'")
        elif agent_signaled_completion:
            action_taken_this_turn = "(Completion Signal: TASK_COMPLETE)"
            logger.info("Agent signaled TASK_COMPLETE.")
        else:
            # Handle case where parsing failed (e.g., missing solar tags)
            action_taken_this_turn = "(No Action Parsed)"
            logger.warning(f"Could not parse valid action/output from agent response for turn {current_turn_number}.")
            action_valid_this_turn = False  # Action is inherently invalid if not parsed
            step_error_this_turn = "Failed to parse action/output from agent response"
            # Specific check for Solar tag error after parsing attempt
            if is_solar and isinstance(agent_response_text, str):  # Check type before 'in'
                # Check raw response again for partial tags if parsing function returned None
                if "<solar.html>" in agent_response_text and "</solar.html>" not in agent_response_text:
                    improperly_closed_tags = True
                    logger.error("Solar specific error: Missing closing </solar.html> tag detected in raw response.")
                    step_error_this_turn = "Agent output missing closing </solar.html> tag"
                elif "</solar.html>" in agent_response_text and "<solar.html>" not in agent_response_text:
                    improperly_closed_tags = True
                    logger.error("Solar specific error: Missing opening <solar.html> tag detected in raw response.")
                    step_error_this_turn = "Agent output missing opening <solar.html> tag"
            # For TTT/FS, parsing failure usually means an invalid move/command format
            if is_ttt or is_fs:
                agent_made_any_invalid_move_in_attempt = True  # Track any invalid move for TTT scoring

        # --- Environment Step Execution ---
        should_execute_step = False
        current_turn_feedback = ""  # Reset feedback for this turn
        # Determine if env.step should be called
        if is_solar:
            # For solar, step executes if content was parsed *and* tags were proper (if checked)
            if action_content is not None and not improperly_closed_tags:
                should_execute_step = True
            else:
                # Provide the specific parsing/tag error as feedback
                current_turn_feedback = step_error_this_turn or "Skipping step due to parsing error/invalid output format."
                action_valid_this_turn = False  # Ensure marked invalid
        elif action_content is not None:
            # For TTT/FS, step executes if an action was parsed
            should_execute_step = True
        else:
            # No action parsed, provide feedback about parsing failure
            current_turn_feedback = step_error_this_turn or "Skipping step: No action parsed from agent response."
            action_valid_this_turn = False

        # Execute step if conditions met
        if should_execute_step:
            try:
                # --- Action Validation (BEFORE executing step for TTT/FS) ---
                if not is_solar:  # Solar validation happens implicitly in step/render
                    action_valid_this_turn = env.validate_action(action_content)  # type: ignore
                    if not action_valid_this_turn:
                        logger.warning(f"Action '{action_taken_this_turn}' failed environment validation.")
                        step_error_this_turn = f"Invalid action based on environment rules: {action_content}"
                        current_turn_feedback = step_error_this_turn  # Use validation error as feedback
                        if is_ttt:
                            agent_made_any_invalid_move_in_attempt = True
                            # TTT: Invalid moves usually end the attempt immediately in simple setups
                            logger.error("TTT: Agent made invalid move. Failing attempt.")
                            premature_failure = True
                            failure_reason = step_error_this_turn
                            is_terminal_this_turn = True  # Ensure loop breaks
                        elif is_fs:
                            # FS: Invalid command might also be treated as failure
                            logger.error(f"FS: Agent action '{action_taken_this_turn}' invalid. Failing attempt.")
                            premature_failure = True
                            failure_reason = step_error_this_turn
                            is_terminal_this_turn = True
                        # Prevent step execution if validation failed
                        should_execute_step = False
                # --- Execute Environment Step ---
                if should_execute_step and not premature_failure:  # Re-check failure flag
                    logger.debug(f"Executing env.step with action for task {task_name}...")
                    # env.step returns differently for different tasks
                    if is_fs:
                        # FS step returns only the result string (output or error)
                        step_result_output = env.step(action_content)  # type: ignore
                        is_terminal_from_step = False  # FS termination decided by agent signal or max rounds
                        # Determine validity based on FS output convention
                        action_valid_this_turn = not str(step_result_output).startswith("Error:")
                        current_turn_feedback = step_result_output
                        new_state_desc_this_turn = env.get_state()  # Get updated state string after step
                    else:  # TTT and Solar return (new_state_description, is_terminal)
                        # Pass empty string if action_content is None? Should not happen if should_execute_step is true.
                        step_result_output, is_terminal_from_step = env.step(action_content if action_content is not None else "")
                        action_valid_this_turn = True  # Assume valid if step executed without error for TTT/Solar
                        current_turn_feedback = step_result_output  # Feedback is the new state description for TTT/Solar
                        new_state_desc_this_turn = step_result_output  # New state is the feedback itself
                    # Determine overall terminal status for the turn
                    is_terminal_this_turn = is_terminal_from_step or agent_signaled_completion
                    logger.debug(f"Env step executed. Terminal state reached: {is_terminal_this_turn}")
                    if current_turn_feedback:
                        log_feedback = str(current_turn_feedback).replace('\n', '\\n')
                        logger.debug(f"Env Result/Feedback (trunc): {log_feedback[:300]}...")
            except Exception as e:
                logger.error(f"Environment step failed for action '{action_taken_this_turn}': {e}", exc_info=True)
                step_error_this_turn = f"Environment step execution error: {e}"
                is_terminal_this_turn = True
                premature_failure = True
                failure_reason = step_error_this_turn
                action_valid_this_turn = False  # Step failed, action considered invalid result
                new_state_desc_this_turn = state_before_action_desc  # Revert to state before failed step
                current_turn_feedback = step_error_this_turn  # Use error as feedback
        else:  # Step was skipped due to parsing or validation failure
            logger.warning(f"Skipping environment step execution for Turn {current_turn_number} due to previous error/invalid action.")
            action_valid_this_turn = False
            # Ensure feedback reflects the reason for skipping
            if not current_turn_feedback:  # If feedback wasn't already set by parsing/validation
                current_turn_feedback = step_error_this_turn or "Step skipped due to error or invalid action."
            new_state_desc_this_turn = state_before_action_desc  # State remains unchanged

        # Check if agent signaled completion outside of environment step result
        if agent_signaled_completion and not is_terminal_this_turn:
            logger.info("Agent signaled TASK_COMPLETE, marking turn as terminal.")
            is_terminal_this_turn = True

        # --- Post-Turn Processing & History Logging ---
        # Assess intermediate status for regression tracking (currently only for non-TTT/Solar)
        intermediate_status = None
        if not is_ttt and not is_solar:  # Regression tracking not used/meaningful for TTT/Solar
            intermediate_status = env.assess_intermediate_status()
            regression = evaluator.track_regression(attempt_id, intermediate_status, turn_count)
            if regression:
                regressions_detected_in_attempt = True
        else:
            regression = False  # No regression tracking for TTT/Solar

        # Log turn details
        turn_data = {
            'turn': current_turn_number,
            'player': turn_player,  # Who acted this turn (Agent or Opponent if applicable)
            'state_before_action': state_before_action_desc,
            'agent_raw_response': agent_response_text if turn_player == "Agent" else None,  # Log raw response only if agent's turn
            'agent_raw_api_response': raw_api_response_this_turn if turn_player == "Agent" else None,  # ADD THIS
            'action_parsed': action_content if action_content is not None else None,
            'action_taken_for_log': action_taken_this_turn,  # Concise action string
            'action_valid': action_valid_this_turn,
            'agent_signaled_completion': agent_signaled_completion,
            'env_feedback_or_result': current_turn_feedback,  # Output from env.step or error message
            'state_after_action': new_state_desc_this_turn,  # State after action was attempted
            'intermediate_status': str(intermediate_status) if intermediate_status is not None else None,
            'regression_detected_this_turn': regression,
            'is_terminal_after_turn': is_terminal_this_turn,
            'error_this_turn': step_error_this_turn,  # Log any specific error message
            'token_usage': token_usage_this_turn,  # Store token usage for the turn
        }
        history.append(turn_data)

        # Prepare feedback for the next agent turn (if conversational)
        # For solar, feedback includes intermediate eval; for FS, it's command output; for TTT, it's new board state.
        env_feedback_or_result = current_turn_feedback

        # Check if the loop should terminate based on this turn's outcome
        if is_terminal_this_turn:
            if not premature_failure:
                logger.info(f"Environment reached terminal state or agent signaled completion at turn {current_turn_number}.")
            else:
                # Log the reason for premature failure if not already clear
                logger.warning(f"Terminating attempt {attempt_id+1} early due to failure: {failure_reason or step_error_this_turn or 'Unknown reason'}")
            break  # Exit the main while loop

        # Increment turn counter ONLY if the agent completed a turn successfully or loop continues
        turn_count += 1

        # --- Tic Tac Toe: Opponent Move ---
        if is_ttt and not is_terminal_this_turn:
            # Check if it's now the opponent's turn after agent's move
            opponent_player = getattr(env, 'opponent_player_mark', 'O')
            if getattr(env, 'current_player', None) == opponent_player:
                logger.info(f"--- Opponent ({opponent_player}) Turn {current_turn_number}.5 ---")
                try:
                    # TTT Env needs a method to make the opponent move internally
                    if hasattr(env, 'make_opponent_move'):
                        opp_action, opp_new_state, opp_terminal = env.make_opponent_move()  # type: ignore

                        # Log opponent's move similarly to agent's turn
                        opp_turn_data = {
                            'turn': current_turn_number + 0.5,  # Indicate opponent move
                            'player': opponent_player,
                            'state_before_action': new_state_desc_this_turn,  # State after agent's move
                            'agent_raw_response': None,
                            'action_parsed': opp_action or "N/A",
                            'action_taken_for_log': opp_action or "(Opponent Failed Move)",
                            'action_valid': opp_action is not None,
                            'agent_signaled_completion': False,
                            'env_feedback_or_result': "N/A",  # Opponent move doesn't generate feedback for agent
                            'state_after_action': opp_new_state,
                            'intermediate_status': None,
                            'regression_detected_this_turn': False,
                            'is_terminal_after_turn': opp_terminal,
                            'error_this_turn': None if opp_action else "Opponent failed to make a move"
                        }
                        history.append(opp_turn_data)

                        # Update state description for the next potential agent turn
                        new_state_desc_this_turn = opp_new_state
                        env_feedback_or_result = opp_new_state  # Update feedback for agent

                        if opp_terminal:
                            logger.info("Game ended after opponent's move.")
                            is_terminal_this_turn = True  # Mark as terminal to break outer loop
                            break  # Exit main loop
                        if opp_action is None:
                            logger.error("Opponent failed to make a move (logic error?). Failing attempt.")
                            premature_failure = True
                            failure_reason = "Opponent move error (returned None)"
                            is_terminal_this_turn = True
                            break  # Exit main loop
                    else:
                        logger.error("TicTacToeEnv is missing the 'make_opponent_move' method.")
                        premature_failure = True
                        failure_reason = "Environment implementation error (missing make_opponent_move)"
                        is_terminal_this_turn = True
                        break
                except Exception as e:
                    logger.error(f"Error during opponent's move execution: {e}", exc_info=True)
                    premature_failure = True
                    failure_reason = f"Opponent move execution error: {e}"
                    is_terminal_this_turn = True
                    break  # Exit main loop

# --- Attempt End ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- Attempt {attempt_id + 1} Finished (Duration: {duration:.2f}s, Rounds/Turns Ran: {turn_count}) ---")

    # --- Final Evaluation ---
    final_score = 1  # Default to failure
    final_eval_response = "Evaluation not performed."  # Default response text
    final_eval_input = None  # What was actually sent to the evaluator (or used for deterministic eval)
    final_premature_failure = premature_failure  # Capture if loop ended early due to error

    if final_premature_failure:
        # If the loop broke due to an error, score is 1
        final_score = 1
        final_eval_response = f"Fail (Premature Failure: {failure_reason or 'Unknown'})"
        logger.info(f"Final Score: 1 ({final_eval_response})")
        final_eval_input_summary = failure_reason  # Use reason as summary
    else:
        # Attempt completed normally (or reached max steps), proceed with final evaluation
        logger.info("Performing final evaluation...")
        try:
            # Get the specific input needed for final evaluation from the environment
            final_eval_input = env.get_final_eval_input()
            # Create a summary for logging (e.g., file path for solar, state string for others)
            final_eval_input_summary = str(final_eval_input)[:200] + ('...' if len(str(final_eval_input)) > 200 else '')

            # --- Task-Specific Final Evaluation ---
            if is_ttt:
                # Outcome flags
                agent_won    = getattr(env, "winner", None) == env.agent_player_mark      # X wins
                opponent_won = getattr(env, "winner", None) == getattr(env, "opponent_player_mark", "O")
                is_draw      = getattr(env, "is_draw", False)                             # board full, no winner
                made_invalid = agent_made_any_invalid_move_in_attempt                     # tracked during run

                if opponent_won:                                         # Player X lost
                    final_score = 2 if not made_invalid else 1           # partial if clean loss, fail if bad + loss
                    final_eval_response = ( 
                        "Partial (Player X lost but made no invalid moves)" 
                        if final_score == 2 
                        else f"Fail (Opponent '{getattr(env, 'opponent_player_mark', 'O')}' won and X made invalid moves)" 
                    )
                elif agent_won or is_draw:                               # Player X won **or** tied
                    final_score = 3 if not made_invalid else 2           # success if clean, partial if any invalids
                    outcome_str = "won" if agent_won else "tied"
                    final_eval_response = ( 
                        f"{'Success' if final_score == 3 else 'Partial'} " 
                        f"(Player X {outcome_str}, {'no' if final_score == 3 else 'some'} invalid moves)" 
                    )
                else:                                                    # should not occur in normal play
                    final_score = 1
                    final_eval_response = "Fail (Game ended inconclusively)"
                logger.info(f"Deterministic TTT Score: {final_score} ({final_eval_response})")
            elif is_fs:
                # Deterministic evaluation based on final file system state
                # FS env needs an internal method to evaluate against its goal
                if hasattr(env, 'evaluate_final_state'):
                    final_score = env.evaluate_final_state()  # Method returns 1, 2, or 3
                    score_map = {3: "Success", 2: "Partial", 1: "Fail"}
                    final_eval_response = f"{score_map.get(final_score, 'Unknown')} (FS Deterministic Eval)"
                    logger.info(f"Deterministic FS Score: {final_score} ({final_eval_response})")
                else:
                    logger.error("FileSystemEnv missing 'evaluate_final_state' method for deterministic eval.")
                    final_score = 1
                    final_eval_response = "Fail (Environment evaluation method missing)"
            elif is_solar:
                # Multimodal evaluation using the FINAL evaluator LLM instance
                template = prompts.get('finaleval_template')
                if not template:
                    logger.error("Final evaluation prompt template (finaleval.txt) missing for solar_gen.")
                    final_score = 1
                    final_eval_response = "Fail (Missing final evaluation prompt)"
                else:
                    # The final_eval_input is the path to the last screenshot
                    image_path = final_eval_input
                    if not isinstance(image_path, str):
                        logger.error(f"Solar final evaluation input is not a string path: {type(image_path)}")
                        final_score = 1
                        final_eval_response = "Fail (Invalid image path type for final eval)"
                    else:
                        # Pass the template to the evaluator instance
                        evaluator.eval_prompt_template = template
                        # Use the evaluator's specific image evaluation method
                        if hasattr(evaluator, 'evaluate_final_image_outcome'):
                            # This method handles path checking, encoding, API call, and score parsing
                            final_score, raw_resp = evaluator.evaluate_final_image_outcome(image_path)
                            final_eval_response = f"Multimodal Eval Raw Response: {raw_resp}"  # Store raw for details
                            logger.info(f"Multimodal Final Score: {final_score} (Raw response logged)")
                        else:
                            logger.error("Evaluator instance is missing the 'evaluate_final_image_outcome' method.")
                            final_score = 1
                            final_eval_response = "Fail (Evaluator configuration error - missing image eval method)"
            else:  # Generic Task - Assume text-based LLM evaluation
                template = prompts.get('finaleval_template')
                if not template:
                    logger.error(f"Final evaluation prompt template (finaleval.txt) missing for task {task_name}.")
                    final_score = 1
                    final_eval_response = "Fail (Missing final evaluation prompt)"
                else:
                    # Pass the template to the evaluator instance
                    evaluator.eval_prompt_template = template
                    # final_eval_input should be the final state string from the environment
                    # Use the evaluator's standard text evaluation method
                    if hasattr(evaluator, 'evaluate_final_outcome'):
                        final_score, raw_resp = evaluator.evaluate_final_outcome(str(final_eval_input))
                        final_eval_response = f"LLM Text Eval Raw Response: {raw_resp}"
                        logger.info(f"LLM Text Final Score: {final_score} (Raw response logged)")
                    else:
                        logger.error("Evaluator instance is missing the 'evaluate_final_outcome' method.")
                        final_score = 1
                        final_eval_response = "Fail (Evaluator configuration error - missing text eval method)"
        except Exception as e:
            logger.error(f"Error during final evaluation phase: {e}", exc_info=True)
            final_score = 1  # Treat evaluation errors as failure
            final_eval_response = f"Fail (Final evaluation phase error: {e})"
            # Keep summary if available, otherwise indicate error here too
            final_eval_input_summary = final_eval_input_summary if final_eval_input_summary != "N/A" else f"Error during eval: {e}"

    # --- Consolidate Result ---
    final_failed_flag = final_premature_failure or (final_score == 1)
    final_failure_reason_str = ""
    if final_failed_flag:
        # Prioritize premature failure reason, otherwise state it was evaluation failure
        final_failure_reason_str = failure_reason if final_premature_failure else "Final evaluation resulted in Fail score (1)"

    is_successful_run = (final_score == 3 and not final_premature_failure)

    result = {
        'attempt_id': attempt_id + 1,
        'task_name': task_name,
        'success': is_successful_run,  # Strict success (score 3, no premature fail)
        'score': final_score,  # 1, 2, or 3 (or 1 if premature fail)
        'failed': final_failed_flag,  # True if score 1 OR premature fail
        'premature_failure': final_premature_failure,  # Explicit flag for early termination due to error
        'failure_reason': final_failure_reason_str,  # Text description if failed=True
        'final_outcome_description': final_eval_response,  # Raw eval response or deterministic outcome string
        'regressions_detected': regressions_detected_in_attempt,  # Boolean flag from regression tracking
        'agent_made_invalid_moves': agent_made_any_invalid_move_in_attempt if is_ttt else None,  # TTT specific
        'rounds_completed': turn_count,  # How many turns/rounds actually ran
        'duration_seconds': duration,  # Wall clock time for the attempt
        'final_eval_input_summary': str(final_eval_input_summary),  # Log what was evaluated (truncated)
        'history': history,  # Detailed list of turn data dictionaries
    }
    return result

# --- Helper to Infer Provider from Model Name ---
def infer_provider_from_model(model_name: str) -> str | None:
    """Infers the provider ('gemini', 'openai', or 'grok') from model name prefix."""
    if not model_name: return None
    name_lower = model_name.lower()
    if name_lower.startswith(('gemini', 'google/')):
        return "gemini"
    elif name_lower.startswith(('gpt-', 'o4-', 'openai/')):
        return "openai"
    elif name_lower.startswith('grok-'):
        return "grok"
    elif name_lower.startswith(('qc-', 'quality_compute-')):
        return "quality_compute"
    elif name_lower.startswith('claude-'):  
        return "anthropic"
    # Add other providers here if needed
    logger.debug(f"Could not infer provider from model name: {model_name}")
    return None  # Cannot determine

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="ToyBench: LLM Agentic Benchmark Suite"
    )
    parser.add_argument("-t", "--task", required=True,
                        help="Task name (e.g., file_system, tic_tac_toe, solar_gen)")
    parser.add_argument("-p", "--provider", default="gemini", 
                        choices=["gemini", "openai", "grok", "quality_compute", "anthropic"],  # Add anthropic
                        help="LLM Provider for the AGENT")
    parser.add_argument("-m", "--model", default=None,
                        help="Agent LLM model name. Defaults to provider default (e.g., gemini-1.5-flash, gpt-4o-mini, grok-3-mini-beta).")
    parser.add_argument("-n", "--attempts", type=int, default=5,
                        help="Number of independent attempts to run")
    parser.add_argument("-r", "--rounds", type=int, default=5,
                        help="Max rounds/steps per attempt (exact for solar_gen, limit for others)")
    parser.add_argument("--evaluator_model", default=None,
                        help="Evaluator LLM model name. Defaults to config default (gemini-1.5-flash). Provider inferred.")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--reasoning_effort", default="low", choices=["low", "high"],
                        help="Reasoning effort for Grok models (low, high). Ignored for other providers.")
    parser.add_argument("--output_dir", default="results",
                        help="Base directory for saving results")
    parser.add_argument("--thinking", action='store_true', help="Enable extended thinking for Anthropic models")
    parser.add_argument("--thinking_budget", type=int, default=16000, help="Token budget for extended thinking (default 16000)")

    args = parser.parse_args()

    try:
        config = load_config()
    except Exception as e:  # Catch potential errors during config load
        print(f"FATAL: Error loading configuration (.env or defaults): {e}")
        exit(1)

    # --- Determine Agent Configuration ---
    agent_provider = args.provider.lower()
    agent_model = args.model
    if not agent_model:
        # Use provider-specific default from config if --model not set
        default_key = f"default_{agent_provider}_model"
        agent_model = config.get(default_key)
        if not agent_model:
            print(f"Error: Agent model not specified via --model and no default found for provider '{agent_provider}' in config.")
            exit(1)
        else:
            print(f"Info: Agent model not specified. Using default for '{agent_provider}': '{agent_model}'")
    logger.info(f"Agent Configuration: Provider='{agent_provider}', Model='{agent_model}'")

    # --- Determine Evaluator Configuration ---
    # Get evaluator model name: from args -> config default -> hardcoded default
    evaluator_model_name = args.evaluator_model or config.get('evaluator_model', 'gemini-1.5-flash')  # Fallback needed

    # Infer provider from evaluator model name
    evaluator_provider = infer_provider_from_model(evaluator_model_name)

    # Handle cases where provider inference fails
    if evaluator_provider is None:
        # Critical for solar_gen: it NEEDS a Gemini multimodal evaluator
        if args.task == 'solar_gen':
            logger.warning(f"Could not infer provider for evaluator model '{evaluator_model_name}'. Task '{args.task}' requires Gemini. Forcing evaluator provider to Gemini.")
            evaluator_provider = 'gemini'
            # Ensure the model is actually Gemini-compatible if forced? Best if user provides correct name.
            if not evaluator_model_name.lower().startswith('gemini'):
                logger.error(f"Evaluator model '{evaluator_model_name}' forced to Gemini provider, but name doesn't look like a Gemini model. Evaluation may fail.")
        else:
            # For other tasks, maybe default to agent's provider? Log clearly.
            logger.warning(f"Could not infer provider for evaluator model '{evaluator_model_name}'. Defaulting to agent's provider '{agent_provider}'.")
            evaluator_provider = agent_provider
    else:
        logger.info(f"Inferred provider '{evaluator_provider}' for evaluator model '{evaluator_model_name}'.")

    logger.info(f"Evaluator Configuration: Provider='{evaluator_provider}', Model='{evaluator_model_name}'")

    # --- Setup Output and Logging ---
    # Use the AGENT model name for the primary output directory structure
    base_output_dir = create_output_dir(args.output_dir, args.task, agent_provider, agent_model)
    log_file = os.path.join(base_output_dir, "run.log")
    # Ensure log level is valid attribute of logging
    log_level_attr = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level=log_level_attr, log_file=log_file)

    logger.info("--- ToyBench Run Initializing ---")
    logger.info(f"Task: {args.task}")
    logger.info(f"Agent Provider: {agent_provider}, Agent Model: {agent_model}")
    logger.info(f"Evaluator Provider: {evaluator_provider}, Evaluator Model: {evaluator_model_name}")
    logger.info(f"Attempts: {args.attempts}, Max Rounds/Steps: {args.rounds}")
    logger.info(f"Reasoning Effort (for Grok): {args.reasoning_effort}")  # Log reasoning effort if applicable
    logger.info(f"Log Level: {args.log_level}, Base Output Directory: {base_output_dir}")
    logger.info(f"Thinking Enabled for Anthropic: {args.thinking}, Thinking Budget: {args.thinking_budget}")

    # --- API Key Checks ---
    # Define a mapping from provider to API key name in config for correct lookup
    api_key_map = {
    'gemini': 'gemini_api_key',
    'openai': 'openai_api_key',
    'grok': 'xai_api_key',
    'quality_compute': 'quality_compute_api_key',
    'anthropic': 'anthropic_api_key',  # ADD THIS LINE
}

    # Agent API key check with mapping
    agent_api_key_name = api_key_map.get(agent_provider)
    if not agent_api_key_name:
        logger.error(f"Unsupported agent provider: '{agent_provider}'. Cannot determine API key name.")
        print(f"Error: Unsupported provider '{agent_provider}'. Supported providers: gemini, openai, grok, quality_compute.")
        exit(1)
    agent_api_key = config.get(agent_api_key_name)
    if not agent_api_key:
        logger.error(f"Agent API Key '{agent_api_key_name}' for provider '{agent_provider}' is required but not found in config/env.")
        print(f"Error: Agent API Key '{agent_api_key_name}' not found. Set the appropriate environment variable (e.g., GOOGLE_API_KEY, OPENAI_API_KEY, XAI_API_KEY, QUALITY_COMPUTE_API_KEY).")
        exit(1)
    else:
        logger.info(f"Agent API Key loaded successfully for provider '{agent_provider}'.")

    # Evaluator API key check with mapping
    evaluator_api_key_name = api_key_map.get(evaluator_provider)
    if not evaluator_api_key_name:
        logger.error(f"Unsupported evaluator provider: '{evaluator_provider}'. Cannot determine API key name.")
        print(f"Error: Unsupported evaluator provider '{evaluator_provider}'.")
        exit(1)
    evaluator_api_key = config.get(evaluator_api_key_name)
    if not evaluator_api_key:
        logger.error(f"Evaluator API Key '{evaluator_api_key_name}' for provider '{evaluator_provider}' is required but not found in config/env.")
        print(f"Error: Evaluator API Key '{evaluator_api_key_name}' not found. Ensure the necessary key is set.")
        exit(1)
    else:
        if evaluator_provider != agent_provider:
            logger.info(f"Evaluator API Key loaded successfully for provider '{evaluator_provider}'.")

    # --- Initialize Components ---
    try:
        # Load task prompts from files
        prompts = load_task_prompts(args.task, config.get('task_definitions_dir', 'tasks'))

        # Instantiate Agent LLM Interface
        agent_llm = get_llm_interface(agent_provider, agent_api_key, agent_model, thinking_enabled=args.thinking, thinking_budget=args.thinking_budget)
        logger.info(f"Agent LLM interface ({type(agent_llm).__name__}) initialized for model {agent_model}.")

        # Instantiate Evaluator LLM Interface (using potentially different provider/key/model)
        final_evaluator_llm = get_llm_interface(evaluator_provider, evaluator_api_key, evaluator_model_name, thinking_enabled=args.thinking, thinking_budget=args.thinking_budget)
        logger.info(f"Final Evaluator LLM interface ({type(final_evaluator_llm).__name__}) initialized for model {evaluator_model_name}.")

        # Instantiate the Final Evaluator class (uses the final_evaluator_llm for scoring)
        # The final eval prompt template is loaded from prompts dict
        final_evaluator = Evaluator(final_evaluator_llm, prompts.get('finaleval_template', ''))
        logger.info("Final Evaluator class initialized.")

        # Note: The intermediate evaluator LLM for SolarSystemEnv is passed directly during env creation later.
        # It uses the same instance as the final_evaluator_llm in this setup.

    except FileNotFoundError as e:
        logger.error(f"Initialization failed: Task prompt files not found. {e}", exc_info=True)
        print(f"Initialization Error: Required task prompt file not found: {e}")
        exit(1)
    except ValueError as e:  # Catch errors from get_llm_interface or Evaluator init
        logger.error(f"Initialization failed: Configuration or value error. {e}", exc_info=True)
        print(f"Initialization Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: Unexpected error during LLM/prompt setup. {e}", exc_info=True)
        print(f"Initialization Error: An unexpected error occurred during setup: {e}")
        exit(1)

    # --- Run Benchmark Attempts ---
    logger.info("--- Starting Benchmark Attempts ---")
    all_results = []
    for i in range(args.attempts):
        attempt_start_time = time.time()
        attempt_history = []  # Store history locally in case of crash
        # Create a unique subdirectory for each attempt's artifacts (logs, screenshots, html)
        attempt_output_dir = os.path.join(base_output_dir, f"attempt_{i+1}")
        try:
            os.makedirs(attempt_output_dir, exist_ok=True)
            logger.info(f"Attempt {i+1} output directory: {attempt_output_dir}")

            # --- Get Environment for the Attempt ---
            # Pass the *final_evaluator_llm* instance to be used for intermediate eval in SolarEnv
            # This assumes the intermediate evaluator should use the same model/provider as final eval.
            # If they need to be different, more config options would be needed.
            environment = get_environment(
                task_name=args.task,
                goal=prompts.get('goal_description', 'No goal provided.'),
                prompts=prompts,
                evaluator_llm=final_evaluator_llm,  # Pass LLM for intermediate Solar eval
                output_dir=attempt_output_dir,     # Pass attempt-specific dir
                max_rounds=args.rounds
            )
            logger.info(f"Environment instance ({type(environment).__name__}) created for attempt {i+1}.")

            # --- Run the Single Attempt ---
            attempt_result = run_attempt(
                attempt_id=i,
                env=environment,
                agent=agent_llm,
                evaluator=final_evaluator,  # Pass the Final Evaluator instance
                max_rounds=args.rounds,
                prompts=prompts,
                task_name=args.task,  # Pass task_name for use in metrics calculation
                output_dir=attempt_output_dir  # Pass attempt-specific dir (used by SolarEnv implicitly now)
            )
            attempt_history = attempt_result.get('history', [])  # Get history even if run failed mid-way
            all_results.append(attempt_result)

        except KeyboardInterrupt:
            logger.warning(f"--- Run interrupted by user during attempt {i+1} ---")
            print("\nRun interrupted by user.")
            # Create a partial result indicating interruption
            all_results.append({
                'attempt_id': i + 1,
                'task_name': args.task,
                'score': 1, 'failed': True, 'premature_failure': True,
                'failure_reason': "User interrupt", 'success': False,
                'final_outcome_description': "Interrupted by user",
                'regressions_detected': False, 'rounds_completed': len(attempt_history),
                'agent_made_invalid_moves': None,  # N/A usually
                'duration_seconds': time.time() - attempt_start_time,
                'history': attempt_history, 'final_eval_input_summary': 'N/A'
            })
            break  # Stop running further attempts

        except ImportError as e:  # Catch missing dependencies if env init failed
            logger.error(f"--- CRITICAL ERROR during attempt {i+1} setup: Missing dependency. {e} ---", exc_info=True)
            print(f"\nFATAL Error: Missing dependency required for task '{args.task}'. {e}")
            print("Please install the required libraries (e.g., 'pip install selenium webdriver-manager') and ensure WebDriver is set up.")
            # Create a partial result indicating the import error
            all_results.append({
                'attempt_id': i + 1,
                'task_name': args.task,
                'score': 1, 'failed': True, 'premature_failure': True,
                'failure_reason': f"Missing dependency: {e}", 'success': False,
                'final_outcome_description': f"Missing dependency: {e}",
                'regressions_detected': False, 'rounds_completed': 0,
                'agent_made_invalid_moves': None,
                'duration_seconds': time.time() - attempt_start_time,
                'history': [], 'final_eval_input_summary': 'N/A'
            })
            # Stop further attempts if a core dependency is missing
            break

        except Exception as e:
            logger.error(f"--- CRITICAL UNHANDLED ERROR during attempt {i+1}: {e} ---", exc_info=True)
            print(f"\nFATAL Error: An unexpected error occurred during attempt {i+1}: {e}")
            # Create a partial result indicating the crash
            all_results.append({
                'attempt_id': i + 1,
                'task_name': args.task,
                'score': 1, 'failed': True, 'premature_failure': True,
                'failure_reason': f"Unhandled exception in run_attempt or setup: {e}", 'success': False,
                'final_outcome_description': f"Unhandled exception: {e}",
                'regressions_detected': False,  # Unknown
                'rounds_completed': len(attempt_history),  # Log turns completed before crash
                'agent_made_invalid_moves': None,  # Unknown
                'duration_seconds': time.time() - attempt_start_time,
                'history': attempt_history,  # Include history up to the crash point
                'final_eval_input_summary': 'N/A'
            })
            # Optionally break here, or let it try subsequent attempts? Let's continue for now.

    logger.info("--- Benchmark Run Finished ---")

    # --- Reporting and Saving ---
    if not all_results:
        logger.warning("No attempts were completed or recorded.")
        print("No results to report.")
        exit(0)

    # Calculate metrics based on the collected results, now including task_name
    try:
        metrics = calculate_metrics(all_results, args.attempts, task_name=args.task)  # Pass task_name to calculate_metrics
        report = format_report(metrics, args.task, agent_provider, agent_model, args.rounds)
        print("\n" + report + "\n")
        logger.info(f"Final Report:\n{report}")
    except Exception as e:
        logger.error(f"Failed to calculate or format metrics/report: {e}", exc_info=True)
        print(f"\nError generating final report: {e}\n")
        # Attempt to save raw results anyway

    # Prepare configuration details to save alongside results
    run_config_args = vars(args).copy()  # Copy args namespace to dict
    run_config_args['agent_provider_used'] = agent_provider
    run_config_args['agent_model_used'] = agent_model
    run_config_args['evaluator_provider_used'] = evaluator_provider
    run_config_args['evaluator_model_used'] = evaluator_model_name
    run_config_args[f'{agent_provider}_api_key_loaded'] = (agent_api_key is not None)
    if evaluator_provider != agent_provider:
        run_config_args[f'{evaluator_provider}_api_key_loaded'] = (evaluator_api_key is not None)
    run_config_args['base_output_directory'] = base_output_dir
    run_config_args['task_definitions_dir_used'] = config.get('task_definitions_dir', 'tasks')

    # Save results (detailed JSONL, summary report, config)
    try:
        save_results(base_output_dir, all_results, report if 'report' in locals() else "Report generation failed.", run_config_args)
    except Exception as e:
        logger.error(f"Failed to save results to {base_output_dir}: {e}", exc_info=True)
        print(f"Error saving results: {e}")

    logger.info(f"Results, logs, and artifacts saved in base directory: {base_output_dir}")
    logger.info("--- ToyBench Run Complete ---")

if __name__ == "__main__":
    main()