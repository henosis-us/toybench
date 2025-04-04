# <toybench_cli.py>
# VERSION MODIFIED TO HANDLE SOLAR_GEN CONVERSATIONALLY + Specific Tag Error Handling

import argparse
import logging
import os
import json
import time
import re # Import regex for parsing completion signal/action
from config import load_config
from utils import setup_logging, create_output_dir, parse_llm_score # Use updated parse_llm_score if needed

# Import Interfaces and Base Classes
from llm_interface import LLMInterface, GeminiInterface # Import specific implementation + base
# Import other provider interfaces when added
from environments.base_env import BaseEnvironment
from evaluation import Evaluator
# Ensure reporting.py uses os and has the corrected calculate_metrics
from reporting import calculate_metrics, format_report, save_results

# Import Specific Environments
from environments.file_system_env import FileSystemEnv # Import specifically to check its type
from environments.tic_tac_toe_env import TicTacToeEnv # Import specifically to check its type
from environments.solar_system_env import SolarSystemEnv # <-- IMPORT NEW ENV

# Import Browser Utilities (Needed for SolarSystemEnv evaluation)
try:
    # These might be needed by the evaluator for Solar
    from browser_utils import encode_file_inline_data_gemini
except ImportError:
    # Keep logger definition accessible globally if needed here
    logger_global_init = logging.getLogger(__name__) # Use a distinct name
    logger_global_init.warning("Could not import browser_utils. SolarSystemEnv task may not function correctly.")
    # Define dummy if needed by evaluator outside the env
    def encode_file_inline_data_gemini(file_path: str) -> dict | None:
        logger = logging.getLogger(__name__)
        logger.error("encode_file_inline_data_gemini called but utils.browser_utils failed to import.")
        return None


# Configure logging early
logger = logging.getLogger(__name__) # Standard logger instance


# --- Task Definitions (Load from files) ---
# (load_task_prompts function remains unchanged)
def load_task_prompts(task_name: str, task_dir: str) -> dict:
    """Loads goal description, generation prompt, intermediate eval, and final eval prompt for a task."""
    task_path = os.path.join(task_dir, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"Task directory not found: {task_path}")

    prompts = {}
    files_to_load = {
        'goal_description': f"{task_name}_goal.txt",
        'generate_template': f"{task_name}_generate.txt",
        'intermediate_eval_template': f"{task_name}_intermediate_eval.txt", # For envs like Solar
        'finaleval_template': f"{task_name}_finaleval.txt",
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
                prompts[key] = None # Indicate loading failure
        else:
            prompts[key] = None # File doesn't exist
            # Warn only for essential prompts unless it's intermediate eval which is task specific
            if key in ['goal_description', 'generate_template', 'finaleval_template'] and key != 'intermediate_eval_template':
                 logger.warning(f"Potentially required prompt file not found: {file_path}")

    # Validation for essential prompts
    if prompts['generate_template'] is None:
        raise FileNotFoundError(f"Essential generation prompt missing: {files_to_load['generate_template']}")
    if prompts['finaleval_template'] is None and task_name not in ["tic_tac_toe", "file_system"]: # Tasks with built-in final eval
         logger.error(f"Final evaluation prompt missing for LLM-evaluated task {task_name}. Using basic default.")
         prompts['finaleval_template'] = "Evaluate the final state based on the goal. Goal: {goal} Final State: {final_outcome}. Rate 1-3 (3=Success): <rating>X</rating>"
    elif prompts['finaleval_template'] is None and task_name in ["tic_tac_toe", "file_system"]:
         logger.info(f"Final evaluation prompt not needed/found for task {task_name} (using deterministic eval).")

    logger.info(f"Loaded prompts for task: {task_name}")
    return prompts


# --- Environment Factory ---
# (get_environment function remains unchanged)
def get_environment(task_name: str, goal: str, prompts: dict, evaluator_llm: LLMInterface | None, output_dir: str, max_rounds: int) -> BaseEnvironment:
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
            return SolarSystemEnv(
                goal_description=goal,
                intermediate_eval_prompt=intermediate_eval_prompt,
                intermediate_eval_llm=evaluator_llm,
                output_dir_base=output_dir,
                max_steps=max_rounds
            )
        except NameError as e:
             logger.error(f"Failed to instantiate SolarSystemEnv. Is 'selenium' installed and browser_utils available? Error: {e}", exc_info=True)
             raise ImportError("Failed to initialize SolarSystemEnv. Check dependencies (selenium, browser_utils).") from e
        except Exception as e:
             logger.error(f"Unexpected error initializing SolarSystemEnv: {e}", exc_info=True)
             raise
    else:
        raise ValueError(f"Unknown task name: {task_name}")

# --- LLM Provider Factory ---
# (get_llm_interface remains unchanged)
def get_llm_interface(provider_name: str, api_key: str, model_name: str) -> LLMInterface:
    """Factory function to instantiate the correct LLM interface."""
    if provider_name.lower() == "gemini":
        if not api_key: raise ValueError("Gemini API Key is required.")
        return GeminiInterface(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")


# --- Action Parsing Helper ---
# (parse_agent_response remains unchanged)
def parse_agent_response(response_text: str, task_name: str) -> tuple[str | None, bool]:
    """
    Parses agent response based on the task type.
    - For 'solar_gen', extracts the code block.
    - For other tasks (TTT/FS), uses original logic (```action or first line).
    Returns (command_or_None, task_complete_bool).
    """
    if response_text is None: return None, False

    # --- Task-Specific Parsing for Solar ---
    if task_name == "solar_gen":
        match = re.search(r"<solar\.html>(.*?)</solar\.html>", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            code_content = match.group(1).strip()
            logger.debug("Parsed action (solar_gen): Extracted content from <solar.html> tags.")
            return code_content, False
        else:
            # Check specifically for missing closing tag *if* opening tag exists
            if "<solar.html>" in response_text and "</solar.html>" not in response_text:
                 logger.error("Solar task specific error: Agent response missing closing </solar.html> tag.")
                 # Return None to indicate parsing failure, let run_attempt handle the specific error message
                 return None, False
            else: # Other parsing issues (no tags, malformed, etc.)
                 logger.warning("Could not find or parse <solar.html>...</solar.html> tags in response for solar_gen.")
                 # Still return None for consistency
                 return None, False

    # --- Original Parsing Logic (for TTT, FS, Generic) ---
    task_complete = "TASK_COMPLETE" in response_text.upper()
    action_match = re.search(r"```action\n(.*?)\n```", response_text, re.DOTALL | re.IGNORECASE)
    if action_match:
        command = action_match.group(1).strip()
        logger.debug(f"Parsed command (block): '{command}'")
        if "TASK_COMPLETE" in command.upper() and len(command.split()) == 1:
             logger.debug("Action block contains only TASK_COMPLETE signal.")
             return None, task_complete
        return command, task_complete

    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if "TASK_COMPLETE" in first_line.upper() and len(first_line.split()) == 1:
            logger.debug("Parsed signal (TASK_COMPLETE line).")
            return None, task_complete
        common_cmds = ["ls", "cd", "mkdir", "touch", "rm", "pwd", "mv", "place"]
        first_word = first_line.split(maxsplit=1)[0].lower() if first_line else ""
        if first_word in common_cmds:
            logger.debug(f"Parsed command (first line heuristic): '{first_line}'")
            return first_line, task_complete

    logger.debug(f"No command parsed from response (Original Logic Path). Signal status: {task_complete}. Response: {response_text[:100]}...")
    return None, task_complete


# --- Core Benchmark Logic ---
# MODIFIED: Added check and handling for improperly closed solar tags
def run_attempt(attempt_id: int, env: BaseEnvironment, agent: LLMInterface, evaluator: Evaluator, max_rounds: int, prompts: dict, task_name: str, output_dir: str) -> dict:
    """Runs a single attempt, handling specific logic for TTT, FS, and Solar."""
    logger.info(f"--- Starting Attempt {attempt_id + 1} (Task: {task_name}, Max Rounds/Steps: {max_rounds}) ---")
    evaluator.reset_attempt_tracking(attempt_id)
    start_time = time.time()

    initial_state_desc = env.reset()
    logger.debug(f"Environment reset. Initial state description (truncated): {str(initial_state_desc)[:200]}...")

    history = []; conversation_history = []
    regressions_detected_in_attempt = False; premature_failure = False
    failure_reason = ""; final_eval_input_summary = "N/A"
    agent_made_any_invalid_move_in_attempt = False
    agent_player_mark = env.get_agent_player_mark()

    turn_count = 0
    is_ttt = isinstance(env, TicTacToeEnv)
    is_fs = isinstance(env, FileSystemEnv)
    is_solar = isinstance(env, SolarSystemEnv)

    # Max steps/rounds logic
    ignore_steps_cli = is_ttt
    if is_ttt: logger.info(f"Task '{task_name}' runs until game completion, --rounds ({max_rounds}) ignored.")
    elif is_fs: logger.info(f"Task '{task_name}' runs until completion/error or FS internal limit. --rounds ({max_rounds}) acts as safeguard.")
    elif is_solar: logger.info(f"Task '{task_name}' runs for exactly {max_rounds} refinement rounds.")
    else: logger.info(f"Task '{task_name}' runs for max {max_rounds} steps.")

    # Use conversational API for FS and Solar
    use_conversational_api = is_fs or is_solar
    if use_conversational_api: logger.info(f"Using conversational API for task '{task_name}'.")

    # --- Initial Prompt Construction ---
    try:
        initial_context = env.get_prompt_context()
        if 'goal' not in initial_context: initial_context['goal'] = prompts.get('goal_description', '')
        template = prompts['generate_template']
        if not template: raise ValueError("Generate template missing.")
        required_keys = set(re.findall(r"\{(\w+)\}", template))
        missing_keys = required_keys - set(initial_context.keys())
        if missing_keys:
             logger.warning(f"Initial prompt context missing keys required by template: {missing_keys}. Attempting defaults.")
             if is_fs: initial_context.setdefault('available_commands', 'ls cd mkdir touch rm pwd mv')
             if is_ttt: initial_context.setdefault('last_invalid_action_feedback', '')
             for key in missing_keys: initial_context.setdefault(key, f"<{key}_placeholder>")
        generation_prompt_text = template.format(**initial_context)
        if use_conversational_api: conversation_history.append({'role': 'user', 'parts': [generation_prompt_text]}); logger.debug(f"Initialized conversational history for {task_name}...")
        else: logger.debug(f"Initial Non-Conversational Prompt (TTT/Generic)...")
    except KeyError as e: logger.error(f"Initial prompt format failed: Key {e}", exc_info=True); premature_failure=True; failure_reason=f"Init prompt key error: {e}"
    except Exception as e: logger.error(f"Initial prompt format failed: {e}", exc_info=True); premature_failure=True; failure_reason=f"Init prompt format error: {e}"

    # --- Main Interaction Loop ---
    env_feedback_or_result = ""
    while not premature_failure:
        # --- Check Max Rounds/Steps ---
        if not is_ttt and turn_count >= max_rounds:
            logger.warning(f"Reached max_rounds ({max_rounds}) set by CLI for task '{task_name}'. Terminating attempt.")
            if not is_solar: failure_reason = f"Reached max_rounds ({max_rounds} without success or completion signal)"
            break

        current_turn_number = turn_count + 1
        log_prefix = f"Attempt {attempt_id + 1}, Round/Turn {current_turn_number}"
        if not is_ttt: log_prefix += f"/{max_rounds}"
        logger.info(f"--- {log_prefix} ---")

        # --- Prepare Turn Variables ---
        is_terminal_this_turn = False; action_taken_this_turn = "N/A"; action_valid_this_turn = False; step_error_this_turn = None
        new_state_desc_this_turn = "N/A"; turn_player = "Agent"; current_turn_feedback = "" # Feedback generated *this* turn
        prompt_context_before_action = env.get_prompt_context()
        if 'goal' not in prompt_context_before_action: prompt_context_before_action['goal'] = prompts.get('goal_description', '')
        state_before_action_desc = str(prompt_context_before_action.get('current_state', 'State Unavailable'))
        new_state_desc_this_turn = state_before_action_desc

        # --- TTT Player Turn Check ---
        if is_ttt:
             current_env_player = getattr(env, 'current_player', agent_player_mark); is_agent_turn = (current_env_player == agent_player_mark); turn_player = current_env_player
             if not is_agent_turn: logger.debug(f"Turn {current_turn_number}: Skipping agent, Opponent ({turn_player})'s turn."); turn_count += 1; continue

        logger.debug(f"Turn {current_turn_number}: Player = {turn_player}")

        # --- Agent Action Generation ---
        agent_response_text = None
        try:
            if use_conversational_api: # FS and Solar Path
                # Add environment feedback from PREVIOUS turn as NEW USER message
                if turn_count > 0 and env_feedback_or_result: # env_feedback_or_result holds feedback from turn N-1
                    if conversation_history and conversation_history[-1]['role'] == 'model':
                        conversation_history.append({'role': 'user', 'parts': [env_feedback_or_result]})
                        logger.debug(f"Appended env feedback from previous turn to history as user message...")
                    else: logger.warning("Skipping appending feedback as last message was not from model.")
                if not conversation_history: logger.error(f"{task_name} conversational history empty."); raise ValueError("Empty history")
                logger.debug(f"Calling conversational API ({task_name}). Hist len: {len(conversation_history)}")
                agent_response_text = agent.generate_action_conversational(conversation_history)
            else: # TTT / Generic Path
                current_context = prompt_context_before_action; template = prompts['generate_template']
                required_keys = set(re.findall(r"\{(\w+)\}", template)); missing_keys = required_keys - set(current_context.keys())
                if missing_keys: logger.warning(f"Turn {current_turn_number}: Prompt context missing keys {missing_keys}. Using defaults.");
                if is_ttt: current_context.setdefault('last_invalid_action_feedback', '')
                for key in missing_keys: current_context.setdefault(key, f"<{key}_placeholder>")
                generation_prompt = template.format(**current_context)
                logger.debug(f"Agent Prompt (Turn {current_turn_number}, Non-FS/Solar)...")
                agent_response_text = agent.generate_action(generation_prompt)
        except KeyError as e: logger.error(f"Prompt format error: Key {e}", exc_info=True); step_error_this_turn=f"Prompt key error: {e}"; is_terminal_this_turn=True; premature_failure=True; failure_reason=step_error_this_turn
        except Exception as e: logger.error(f"Agent generation error: {e}", exc_info=True); step_error_this_turn=f"Agent generation error: {e}"; is_terminal_this_turn=True; premature_failure=True; failure_reason=step_error_this_turn
        if agent_response_text is None and not premature_failure: logger.error(f"Agent failed response (API Error?)."); step_error_this_turn="Agent LLM failed"; is_terminal_this_turn=True; premature_failure=True; failure_reason=step_error_this_turn; action_taken_this_turn='AGENT_API_ERROR'
        if premature_failure: break

        logger.debug(f"Raw Agent Response (Turn {current_turn_number})...")
        if use_conversational_api: conversation_history.append({'role': 'model', 'parts': [agent_response_text]}) # Add agent response now

        # --- Parse Agent Response ---
        action_content, agent_signaled_completion = parse_agent_response(agent_response_text, task_name)
        if action_content is not None: log_action = action_content[:100] + '...'; action_taken_this_turn = log_action; logger.info(f"Agent proposed action/output: '{log_action}'")
        elif agent_signaled_completion: action_taken_this_turn = "(Completion Signal)"; logger.info(f"Agent signaled completion.")
        else: action_taken_this_turn = "(No Action Parsed)"; logger.warning(f"Could not parse action/output."); action_valid_this_turn = False; step_error_this_turn = "Failed to parse action/output" # General parse error
        if (is_ttt or is_fs) and action_content is None and not agent_signaled_completion: agent_made_any_invalid_move_in_attempt = True

        # *** ADDED: Specific check for Solar tag error ***
        improperly_closed_tags = False
        if is_solar and action_content is None and not agent_signaled_completion:
            # Check if parsing failed specifically due to missing closing tag
            if "<solar.html>" in agent_response_text and "</solar.html>" not in agent_response_text:
                 improperly_closed_tags = True
                 logger.error("Solar specific error: Missing closing </solar.html> tag.")
                 step_error_this_turn = "did not properly enclose in solar tags" # Specific error message
                 action_valid_this_turn = False # Action is invalid
            # If parse failed for other reasons, step_error_this_turn is already set

        # --- Environment Step Execution ---
        should_execute_step = False # Flag to control if env.step is called
        current_turn_feedback = ""  # Initialize feedback/result for *this* turn

        # Determine if step should be executed
        if is_solar:
             # Execute solar step only if code was parsed correctly (action_content not None)
             # AND tags were not improperly closed
             if action_content is not None and not improperly_closed_tags:
                  should_execute_step = True
             else:
                  # Use the specific error message if tags were bad, otherwise use general parse error
                  current_turn_feedback = step_error_this_turn or "Skipping step due to parsing error."
                  action_valid_this_turn = False # Ensure action is marked invalid
        elif action_content is not None: # For TTT/FS/Generic, step only if action was parsed
             should_execute_step = True # Tentatively true, validation below might prevent it

        # Execute step if needed
        if should_execute_step:
             try:
                if not is_solar: # Pre-step validation for TTT/FS/Generic
                    action_valid_this_turn = env.validate_action(action_content)
                    if not action_valid_this_turn:
                         logger.warning(f"Invalid action: '{action_content}'"); step_error_this_turn = f"Invalid action: {action_content}"
                         if is_ttt: agent_made_any_invalid_move_in_attempt = True; logger.error("TTT Invalid move. Failing (simple)."); premature_failure=True; failure_reason=step_error_this_turn; is_terminal_this_turn=True
                         else: premature_failure=True; failure_reason=step_error_this_turn; is_terminal_this_turn=True
                         should_execute_step = False # Prevent execution if invalid
                    # else: Action valid for pre-check

                if should_execute_step and not premature_failure: # Re-check flags
                    logger.debug(f"Executing env.step for {task_name}...")
                    step_result_output = None; is_terminal_from_step = False # Init locals
                    if is_fs: step_result_output = env.step(action_content); is_terminal_from_step = False; action_valid_this_turn = not str(step_result_output).startswith("Error:") # FS Workaround
                    elif is_ttt or is_solar or True: step_result_output, is_terminal_from_step = env.step(action_content if action_content is not None else ""); action_valid_this_turn = True # Standard
                    is_terminal_this_turn = is_terminal_from_step or agent_signaled_completion

                    # Store feedback/result from THIS turn's step
                    current_turn_feedback = step_result_output

                    # Update state description for history logging
                    if is_solar: new_state_desc_this_turn = "See feedback log" # Handled via feedback
                    elif is_fs: new_state_desc_this_turn = env.get_state() # Get explicit state
                    else: new_state_desc_this_turn = step_result_output # State is the output for TTT/Generic
                    logger.debug(f"Env step executed. Terminal: {is_terminal_this_turn}")
                    if current_turn_feedback: logger.debug(f"Env Result/Feedback (trunc): {str(current_turn_feedback)[:300]}...")

             except Exception as e: logger.error(f"Env step failed: {e}", exc_info=True); step_error_this_turn=f"Env step error: {e}"; is_terminal_this_turn=True; premature_failure=True; failure_reason=step_error_this_turn; action_valid_this_turn=False; new_state_desc_this_turn = state_before_action_desc; current_turn_feedback = step_error_this_turn # Use error as feedback
        # If step wasn't executed (parse fail, invalid action, solar tag error)
        elif not should_execute_step:
             logger.warning(f"Skipping environment step for Turn {current_turn_number} due to previous error.")
             action_valid_this_turn = False # Explicitly mark invalid if step skipped
             # current_turn_feedback already contains the error message from parsing/validation/tag check
             if not current_turn_feedback: current_turn_feedback = step_error_this_turn or "Step skipped due to error."
             new_state_desc_this_turn = state_before_action_desc # State does not change

        # Check for agent signaling completion AFTER step attempt
        if agent_signaled_completion and not is_terminal_this_turn: logger.info("Agent signaled TASK_COMPLETE."); is_terminal_this_turn = True

        # --- Post-Turn Processing & History Logging ---
        intermediate_status = env.assess_intermediate_status(); regression = evaluator.track_regression(attempt_id, intermediate_status, turn_count);
        if regression: regressions_detected_in_attempt = True
        turn_data = {'turn': current_turn_number, 'player': turn_player, 'state_before_action': state_before_action_desc, 'action_taken': action_taken_this_turn, 'action_valid': action_valid_this_turn, 'agent_signaled_completion': agent_signaled_completion, 'env_feedback_or_result': current_turn_feedback, 'state_after_action': new_state_desc_this_turn, 'intermediate_status': str(intermediate_status) if intermediate_status is not None else None, 'regression_detected_this_turn': regression, 'is_terminal_after_turn': is_terminal_this_turn, 'error_this_turn': step_error_this_turn}; history.append(turn_data)

        # Store feedback from THIS turn for the NEXT conversational turn
        env_feedback_or_result = current_turn_feedback

        if is_terminal_this_turn:
            if not premature_failure: logger.info(f"Environment reached terminal state or agent signaled completion at turn {current_turn_number}.")
            else: logger.warning(f"Terminating attempt {attempt_id+1} due to failure: {failure_reason or step_error_this_turn or 'Unknown reason'}")
            break
        turn_count += 1

        # --- Special TTT Opponent Move ---
        if is_ttt and not is_terminal_this_turn:
             opponent_player = getattr(env, 'opponent_player_mark', 'O')
             if getattr(env, 'current_player', None) == opponent_player:
                  logger.info(f"--- Opponent ({opponent_player}) Turn ---")
                  try:
                       opp_action, opp_new_state, opp_terminal = env.make_opponent_move()
                       opp_turn_data = {'turn': current_turn_number + 0.5, 'player': opponent_player, 'state_before_action': new_state_desc_this_turn, 'action_taken': opp_action or "N/A", 'action_valid': opp_action is not None, 'agent_signaled_completion': False, 'env_feedback_or_result': "N/A", 'state_after_action': opp_new_state, 'intermediate_status': None, 'regression_detected_this_turn': False, 'is_terminal_after_turn': opp_terminal, 'error_this_turn': None if opp_action else "Opponent failed move"}; history.append(opp_turn_data); new_state_desc_this_turn = opp_new_state
                       if opp_terminal: logger.info(f"Game ended after opponent's move."); is_terminal_this_turn = True; break
                       if opp_action is None: logger.error("Opponent failed move."); premature_failure = True; failure_reason = "Opponent move error"; is_terminal_this_turn = True; break
                  except Exception as e: logger.error(f"Error during opponent's move: {e}", exc_info=True); premature_failure = True; failure_reason = f"Opponent move error: {e}"; is_terminal_this_turn = True; break

    # --- Attempt End ---
    # (Final Score Assignment remains unchanged)
    end_time = time.time(); duration = end_time - start_time
    logger.info(f"--- Attempt {attempt_id + 1} Finished (Duration: {duration:.2f}s, Rounds/Turns Ran: {turn_count}) ---")
    final_score = 1; final_eval_response = "Evaluation not performed."; final_eval_input = None
    final_premature_failure = premature_failure
    if final_premature_failure: final_score = 1; final_eval_response = f"Fail (Premature: {failure_reason or 'Unknown'})"; logger.info(f"Final Score: 1 ({final_eval_response})"); final_eval_input_summary = failure_reason
    else:
        logger.info("Performing final evaluation...")
        try:
             final_eval_input = env.get_final_eval_input(); final_eval_input_summary = str(final_eval_input)[:200]
             if is_ttt or is_fs: logger.info(f"Performing deterministic eval for {task_name}...");
             if is_ttt: agent_won=env.winner == env.agent_player_mark; is_draw=getattr(env, 'is_draw', False); opponent_won=env.winner == getattr(env, 'opponent_player_mark', 'O')
             if is_ttt:
                 if opponent_won: final_score=1; final_eval_response=f"Fail (Opponent '{env.opponent_player_mark}' won)"
                 elif agent_won or is_draw: outcome="Win" if agent_won else "Draw"; final_score = 3 if not agent_made_any_invalid_move_in_attempt else 2; final_eval_response = f"{'Success' if final_score==3 else 'Partial'} ({outcome}{' no invalid' if final_score==3 else ' invalid'} moves)"
                 else: final_score=1; final_eval_response="Fail (Inconclusive)"
             elif is_fs: final_score = env.evaluate_final_state(); final_eval_response = {3:"Success", 2:"Partial", 1:"Fail"}.get(final_score, "Fail") + " (FS state eval)"
             if is_ttt or is_fs: logger.info(f"Deterministic Score: {final_score} ({final_eval_response})")
             elif is_solar: logger.info(f"Performing Multimodal eval for {task_name}..."); logger.debug(f"Eval Input (Path): {final_eval_input}"); template=prompts.get('finaleval_template')
             if is_solar:
                 if template:
                     evaluator.eval_prompt_template=template; path=final_eval_input
                     if path == "SCREENSHOT_UNAVAILABLE" or not os.path.exists(path) or os.path.getsize(path)==0: logger.error("Screenshot unavailable."); final_score=1; final_eval_response="Fail (Screenshot missing)"
                     elif hasattr(evaluator, 'evaluate_final_image_outcome'): final_score, resp=evaluator.evaluate_final_image_outcome(path); final_eval_response=f"Multimodal Eval Raw: {resp}"; logger.info(f"Multimodal score: {final_score}")
                     else: logger.error("Evaluator lacks image eval."); final_score=1; final_eval_response="Fail (Evaluator config error)"
                 else: logger.error(f"Missing final eval prompt for solar."); final_score=1; final_eval_response="Fail (Missing prompt)"
             elif not is_ttt and not is_fs: # Generic LLM Eval
                 logger.info(f"Performing LLM eval for {task_name}..."); logger.debug(f"Eval Input (State): {final_eval_input_summary}"); template=prompts.get('finaleval_template')
                 if template:
                     evaluator.eval_prompt_template=template
                     try: final_score, resp = evaluator.evaluate_final_outcome(final_eval_input); final_eval_response = f"LLM Eval Raw: {resp}"; logger.info(f"LLM eval score: {final_score}")
                     except Exception as e: logger.error(f"LLM eval error: {e}", exc_info=True); final_score=1; final_eval_response=f"Fail (LLM eval error: {e})"
                 else: logger.error(f"Missing final eval prompt for {task_name}."); final_score=1; final_eval_response="Fail (Missing prompt)"
        except Exception as e: logger.error(f"Final eval phase error: {e}", exc_info=True); final_score=1; final_eval_response=f"Fail (Eval phase error: {e})"; final_eval_input_summary=final_eval_input_summary or f"Error: {e}"

    # --- Compile final result dictionary ---
    final_failed_flag = final_premature_failure or (final_score == 1)
    final_failure_reason_str = ""
    if final_failed_flag: final_failure_reason_str = failure_reason if final_premature_failure else "Evaluation resulted in Fail score"
    is_successful_run = (final_score == 3 and not final_premature_failure)
    result = {
        'attempt_id': attempt_id + 1, 'success': is_successful_run, 'score': final_score,
        'failed': final_failed_flag, 'premature_failure': final_premature_failure, 'failure_reason': final_failure_reason_str,
        'final_outcome_description': final_eval_response, 'regressions_detected': regressions_detected_in_attempt,
        'agent_made_invalid_moves': agent_made_any_invalid_move_in_attempt if is_ttt else None,
        'rounds_completed': turn_count, 'duration_seconds': duration, 'final_eval_input_summary': str(final_eval_input_summary), 'history': history,
    }
    return result


# --- Main Execution ---
# (main function remains unchanged)
def main():
    parser = argparse.ArgumentParser(description="ToyBench: LLM Agentic Benchmark Suite")
    parser.add_argument("-t", "--task", required=True, help="Task name (e.g., file_system, tic_tac_toe, solar_gen)")
    parser.add_argument("-p", "--provider", default="gemini", help="LLM Provider (e.g., gemini)")
    parser.add_argument("-m", "--model", default=None, help="Agent LLM model name. Defaults to provider default.")
    parser.add_argument("-n", "--attempts", type=int, default=5, help="Number of independent attempts to run")
    parser.add_argument("-r", "--rounds", type=int, default=5, help="Max rounds/steps per attempt (exact for solar_gen, max/safeguard for others, ignored for TTT)")
    parser.add_argument("--evaluator_model", default=None, help="Evaluator LLM model (defaults to provider default or flash). Used for LLM-based task eval & Solar intermediate eval.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--output_dir", default="results", help="Base directory for saving results")
    args = parser.parse_args()

    try: config = load_config()
    except ValueError as e: print(f"Configuration Error: {e}"); exit(1)
    agent_model = args.model; gemini_default = 'gemini-1.5-pro-latest'
    if not agent_model: default_model_key = f'default_{args.provider.lower()}_model'; agent_model = config.get(default_model_key, gemini_default if args.provider.lower() == 'gemini' else None)
    if not agent_model: print(f"Error: Agent model not specified via --model and no default found/set for provider '{args.provider}'."); exit(1)
    if not args.model: print(f"Warning: Agent model not specified. Using default: '{agent_model}'")
    evaluator_model_name = args.evaluator_model if args.evaluator_model else config.get('evaluator_model', 'gemini-1.5-flash-latest')
    base_output_dir = create_output_dir(args.output_dir, args.task, args.provider, agent_model)
    log_file = os.path.join(base_output_dir, "run.log"); setup_logging(log_level=getattr(logging, args.log_level.upper()), log_file=log_file)
    logger.info("--- ToyBench Run Initializing ---"); logger.info(f"Task: {args.task}, Provider: {args.provider}, Agent Model: {agent_model}, Attempts: {args.attempts}, Max Rounds/Steps: {args.rounds}"); logger.info(f"Evaluator Model: {evaluator_model_name} (Used for non-deterministic/Solar eval)"); logger.info(f"Log Level: {args.log_level}, Base Output Directory: {base_output_dir}")
    api_key_name = f'{args.provider.lower()}_api_key'; api_key = config.get(api_key_name); api_key_present = bool(api_key)
    logger.info(f"API Key ({api_key_name}): {'Loaded' if api_key_present else 'NOT FOUND'}")
    if not api_key_present: logger.error(f"API Key '{api_key_name}' is required but not found."); print(f"Error: API Key '{api_key_name}' not found."); exit(1)

    try:
        prompts = load_task_prompts(args.task, config.get('task_definitions_dir', 'tasks'))
        agent_llm = get_llm_interface(args.provider, api_key, agent_model)
        evaluator_llm = get_llm_interface(args.provider, api_key, evaluator_model_name)
        evaluator = Evaluator(evaluator_llm, prompts.get('finaleval_template', ''))
    except FileNotFoundError as e: logger.error(f"Init failed: Task files not found. {e}", exc_info=True); print(f"Init Error: {e}"); exit(1)
    except ValueError as e: logger.error(f"Init failed: Config/value error. {e}", exc_info=True); print(f"Init Error: {e}"); exit(1)
    except Exception as e: logger.error(f"Init failed: Unexpected error during LLM/prompt setup. {e}", exc_info=True); print(f"Init Error: {e}"); exit(1)

    logger.info("--- Starting Benchmark Attempts ---")
    all_results = []
    for i in range(args.attempts):
        attempt_start_time = time.time(); attempt_history = []
        attempt_output_dir = os.path.join(base_output_dir, f"attempt_{i+1}"); os.makedirs(attempt_output_dir, exist_ok=True)
        logger.info(f"Attempt {i+1} output dir: {attempt_output_dir}")
        try:
            environment = get_environment(task_name=args.task, goal=prompts.get('goal_description', 'No goal provided.'), prompts=prompts, evaluator_llm=evaluator_llm, output_dir=attempt_output_dir, max_rounds=args.rounds)
            attempt_result = run_attempt(attempt_id=i, env=environment, agent=agent_llm, evaluator=evaluator, max_rounds=args.rounds, prompts=prompts, task_name=args.task, output_dir=attempt_output_dir)
            attempt_history = attempt_result.get('history', [])
            all_results.append(attempt_result)
        except KeyboardInterrupt: logger.warning(f"--- Run interrupted by user during attempt {i+1} ---"); print("\nRun interrupted."); all_results.append({'attempt_id': i + 1, 'score': 1, 'failed': True, 'premature_failure': True, 'failure_reason': "User interrupt", 'success': False, 'final_outcome_description': "Interrupted by user", 'regressions_detected': False, 'rounds_completed': len(attempt_history), 'agent_made_invalid_moves': None, 'duration_seconds': time.time() - attempt_start_time, 'history': attempt_history, 'final_eval_input_summary': 'N/A'}); break
        except ImportError as e: logger.error(f"--- CRITICAL ERROR during attempt {i+1} setup: {e} ---", exc_info=True); print(f"Fatal Error: Missing dependency for task '{args.task}'. {e}"); all_results.append({'attempt_id': i + 1, 'score': 1, 'failed': True, 'premature_failure': True, 'failure_reason': f"Missing dependency: {e}", 'success': False, 'final_outcome_description': f"Missing dependency: {e}", 'regressions_detected': False, 'rounds_completed': 0, 'agent_made_invalid_moves': None, 'duration_seconds': time.time() - attempt_start_time, 'history': [], 'final_eval_input_summary': 'N/A'}); break
        except Exception as e: logger.error(f"--- CRITICAL ERROR during attempt {i+1}: {e} ---", exc_info=True); all_results.append({'attempt_id': i + 1, 'score': 1, 'failed': True, 'premature_failure': True, 'failure_reason': f"Unhandled exception in run_attempt: {e}", 'success': False, 'final_outcome_description': f"Unhandled exception: {e}", 'regressions_detected': False, 'rounds_completed': len(attempt_history), 'agent_made_invalid_moves': None, 'duration_seconds': time.time() - attempt_start_time, 'history': attempt_history, 'final_eval_input_summary': 'N/A'})

    logger.info("--- Benchmark Run Finished ---")
    if not all_results: logger.warning("No attempts completed."); print("No results to report."); exit(0)
    metrics = calculate_metrics(all_results, args.attempts); report = format_report(metrics, args.task, args.provider, agent_model, args.rounds)
    print("\n" + report + "\n"); logger.info(f"Final Report:\n{report}")
    run_config_args = vars(args).copy(); run_config_args['agent_model_used'] = agent_model; run_config_args['evaluator_model_used'] = evaluator_model_name; run_config_args[f'{args.provider.lower()}_api_key_loaded'] = api_key_present; run_config_args['base_output_directory'] = base_output_dir
    try: save_results(base_output_dir, all_results, report, run_config_args)
    except Exception as e: logger.error(f"Failed to save results: {e}", exc_info=True); print(f"Error saving results: {e}")
    logger.info(f"Results/logs saved in base directory: {base_output_dir}"); logger.info("--- ToyBench Run Complete ---")

if __name__ == "__main__":
    main()