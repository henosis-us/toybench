# <toybench_cli.py>
import argparse
import logging
import os
import json
import time
import re # Import regex for parsing completion signal/action

from config import load_config
from utils import setup_logging, create_output_dir
from llm_interface import LLMInterface, GeminiInterface # Import specific implementation + base
# Import other provider interfaces when added
from environments.base_env import BaseEnvironment
from environments.file_system_env import FileSystemEnv # Import specifically to check its type
from environments.tic_tac_toe_env import TicTacToeEnv # Import specifically to check its type
# --- Import other environments as needed ---
from evaluation import Evaluator
from reporting import calculate_metrics, format_report, save_results # Ensure reporting.py uses os

# Configure logging early
logger = logging.getLogger(__name__)

# --- Task Definitions (Load from files) ---
def load_task_prompts(task_name: str, task_dir: str) -> dict:
    """Loads goal description, generation prompt, and eval prompt for a task."""
    task_path = os.path.join(task_dir, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"Task directory not found: {task_path}")
    prompts = {}
    try:
        goal_file = os.path.join(task_path, f"{task_name}_goal.txt")
        if os.path.exists(goal_file):
             with open(goal_file, 'r', encoding='utf-8') as f: prompts['goal_description'] = f.read().strip()
        else:
             logger.warning(f"Goal file not found: {goal_file}. Ensure environment provides goal.")
             prompts['goal_description'] = "" # Env must provide goal if file missing

        generate_file = os.path.join(task_path, f"{task_name}_generate.txt")
        with open(generate_file, 'r', encoding='utf-8') as f:
            prompts['generate_template'] = f.read().strip()

        eval_file = os.path.join(task_path, f"{task_name}_finaleval.txt")
        if os.path.exists(eval_file):
             with open(eval_file, 'r', encoding='utf-8') as f:
                 prompts['finaleval_template'] = f.read().strip()
        else:
             # Provide a very basic default evaluation prompt if missing
             prompts['finaleval_template'] = "Evaluate the final state based on the goal. Goal: {goal} Final State: {final_outcome}. Rate 1-3 (3=Success): <rating>X</rating>"
             logger.warning(f"Final evaluation prompt not found for task {task_name}. Using basic default.")

    except FileNotFoundError as e: logger.error(f"Missing prompt file: {e}"); raise
    except Exception as e: logger.error(f"Error loading prompts: {e}"); raise

    logger.info(f"Loaded prompts for task: {task_name}")
    return prompts

# --- Environment Factory ---
def get_environment(task_name: str, goal: str) -> BaseEnvironment:
    """Factory function to instantiate the correct environment."""
    if task_name == "file_system": return FileSystemEnv(goal)
    elif task_name == "tic_tac_toe": return TicTacToeEnv(goal)
    # Add other environments here
    # elif task_name == "another_task": return AnotherTaskEnv(goal)
    else: raise ValueError(f"Unknown task name: {task_name}")

# --- LLM Provider Factory ---
def get_llm_interface(provider_name: str, api_key: str, model_name: str) -> LLMInterface:
    """Factory function to instantiate the correct LLM interface."""
    if provider_name.lower() == "gemini":
        if not api_key: raise ValueError("Gemini API Key is required.")
        return GeminiInterface(api_key=api_key, model_name=model_name)
    # Add other providers here
    # elif provider_name.lower() == "openai":
    #     if not api_key: raise ValueError("OpenAI API Key is required.")
    #     return OpenAIInterface(api_key=api_key, model_name=model_name)
    else: raise ValueError(f"Unsupported LLM provider: {provider_name}")

# --- Action Parsing Helper ---
def parse_agent_response(response_text: str) -> tuple[str | None, bool]:
    """
    Parses agent response for command (in ```action block or first likely line)
    and completion signal ('TASK_COMPLETE').
    Returns (command_or_None, task_complete_bool).
    """
    if response_text is None: return None, False

    # 1. Check for TASK_COMPLETE signal anywhere
    task_complete = "TASK_COMPLETE" in response_text.upper()

    # 2. Look for ```action block
    action_match = re.search(r"```action\n(.*?)\n```", response_text, re.DOTALL | re.IGNORECASE)
    if action_match:
        command = action_match.group(1).strip()
        logger.debug(f"Parsed command (block): '{command}'")
        # If the block *only* contains TASK_COMPLETE, treat it as signal only, no command
        if "TASK_COMPLETE" in command.upper() and len(command.split()) == 1:
             logger.debug("Action block contains only TASK_COMPLETE signal.")
             return None, task_complete # Signal found, no separate command
        return command, task_complete # Command found in block, signal possibly elsewhere

    # 3. Fallback: Check lines for likely commands or just the signal
    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        # Treat TASK_COMPLETE on its own line as signal only
        if "TASK_COMPLETE" in first_line.upper() and len(first_line.split()) == 1:
            logger.debug("Parsed signal (TASK_COMPLETE line).")
            # If there are other lines, one might be a command - this simplistic check doesn't handle that well.
            # Assume for now: if first line is *just* TASK_COMPLETE, it's primarily a signal.
            # We already checked for the signal anywhere, so task_complete is set.
            # Could potentially check lines[1] here if needed.
            return None, task_complete

        # Check if first line looks like a known command (heuristic)
        common_cmds = ["ls", "cd", "mkdir", "touch", "rm", "pwd", "mv", "place"] # Add other task cmds if needed
        first_word = first_line.split(maxsplit=1)[0].lower() if first_line else ""
        if first_word in common_cmds:
            logger.debug(f"Parsed command (first line heuristic): '{first_line}'")
            return first_line, task_complete

    # 4. No command found, return signal status only
    logger.debug(f"No command parsed from response. Signal status: {task_complete}. Response: {response_text[:100]}...")
    return None, task_complete


# --- Core Benchmark Logic ---
def run_attempt(attempt_id: int, env: BaseEnvironment, agent: LLMInterface, evaluator: Evaluator, max_steps: int, prompts: dict, task_name: str) -> dict:
    """Runs a single attempt, handling specific logic for TTT and FS."""
    logger.info(f"--- Starting Attempt {attempt_id + 1} ---")
    evaluator.reset_attempt_tracking(attempt_id)
    start_time = time.time()
    # Reset env. NOTE: initial_state_desc is captured but NOT directly shown to agent for FS.
    initial_state_desc = env.reset()
    logger.debug(f"Environment reset. Initial internal state description: {initial_state_desc}") # Log for debug

    history = []; conversation_history = []
    regressions_detected_in_attempt = False; attempt_failed_prematurely = False
    failure_reason = ""; final_eval_input = "N/A (Attempt Failed Prematurely)"
    agent_player_mark = env.get_agent_player_mark()
    MAX_INVALID_MOVES_PER_TURN = 3; agent_made_any_invalid_move_in_attempt = False
    turn_count = 0
    is_ttt = isinstance(env, TicTacToeEnv); is_fs = isinstance(env, FileSystemEnv)

    # Determine if max_steps should be ignored (tasks that run until natural completion)
    ignore_steps = is_ttt or is_fs # Add other tasks here if they have defined end states
    if ignore_steps: logger.info(f"Task '{task_name}' runs until completion/error, --steps ({max_steps}) ignored.")
    else: logger.info(f"Task '{task_name}' runs for max {max_steps} steps.")


    # --- Construct Initial Prompt & History ---
    initial_prompt = ""
    if is_fs:
        # **MODIFICATION**: Do not include initial_state in the first prompt for FS.
        # The new generate template encourages exploration.
        try:
            # Get available commands from env context (even after reset)
            fs_context = env.get_prompt_context()
            available_commands = fs_context.get("available_commands", "ls, cd, mkdir, touch, rm, pwd, mv") # Default fallback
            initial_prompt = prompts['generate_template'].format(
                goal=env.get_goal(),
                available_commands=available_commands
                # **REMOVED**: initial_state=initial_state_desc
            )
            # Ensure the template didn't accidentally require initial_state
            if "{initial_state}" in initial_prompt:
                 logger.warning("Detected {{initial_state}} in FS prompt, but it should be removed for agentic discovery.")
                 # Attempt to remove it if found, though template file should be fixed
                 initial_prompt = initial_prompt.replace("{initial_state}", "(Figure out the initial state using commands like 'ls')")

            logger.debug(f"Initial FS Prompt (State Hidden):\n{initial_prompt}")
            # Initialize conversation history for FS
            conversation_history.append({'role': 'user', 'parts': [initial_prompt]})
        except KeyError as e:
            logger.error(f"FS Initial Prompt key error: {e}. Check generate template.", exc_info=True)
            attempt_failed_prematurely = True; failure_reason = f"Init prompt key error: {e}"
        except Exception as e:
            logger.error(f"Error formatting FS initial prompt: {e}", exc_info=True)
            attempt_failed_prematurely = True; failure_reason = f"Init prompt format error: {e}"
    elif is_ttt:
        # TTT requires the initial board state
        try:
            # Get initial context after reset
            ttt_context = env.get_prompt_context()
            # Add placeholder for feedback if needed by template
            if 'last_invalid_action_feedback' not in ttt_context: ttt_context['last_invalid_action_feedback'] = ""
            initial_prompt = prompts['generate_template'].format(**ttt_context)
            logger.debug(f"Initial TTT Prompt:\n{initial_prompt}")
            # TTT likely uses non-conversational generate_action, history not used directly by LLM call
        except KeyError as e:
            logger.error(f"TTT Initial Prompt key error: {e}. Check generate template.", exc_info=True)
            attempt_failed_prematurely = True; failure_reason = f"Init prompt key error: {e}"
        except Exception as e:
            logger.error(f"Error formatting TTT initial prompt: {e}", exc_info=True)
            attempt_failed_prematurely = True; failure_reason = f"Init prompt format error: {e}"
    else:
        # Generic tasks - Assume they need initial state unless specified otherwise
        try:
            generic_context = env.get_prompt_context() # Get state after reset
            initial_prompt = prompts['generate_template'].format(**generic_context)
            logger.debug(f"Initial Generic Task Prompt:\n{initial_prompt}")
            # Generic tasks likely use non-conversational generate_action
        except KeyError as e:
            logger.error(f"Generic Initial Prompt key error: {e}. Check template.", exc_info=True)
            attempt_failed_prematurely = True; failure_reason = f"Init prompt key error: {e}"
        except Exception as e:
            logger.error(f"Error formatting Generic initial prompt: {e}", exc_info=True)
            attempt_failed_prematurely = True; failure_reason = f"Init prompt format error: {e}"

    # --- Main Interaction Loop ---
    while not attempt_failed_prematurely: # Exit loop if init failed
        # --- Check max_steps ONLY if NOT ignored ---
        if not ignore_steps and turn_count >= max_steps:
            logger.warning(f"Reached max_steps ({max_steps}) for task '{task_name}'. Terminating attempt.")
            failure_reason = f"Reached max_steps ({max_steps})"
            attempt_failed_prematurely = True # Treat reaching max steps as a failure for scoring
            break # Exit loop

        current_turn_number = turn_count + 1
        turn_log_msg = f"Attempt {attempt_id + 1}, Turn {current_turn_number}"
        if not ignore_steps: turn_log_msg += f"/{max_steps}"
        logger.info(turn_log_msg)

        is_terminal_this_turn = False; action_taken_this_turn = "N/A"; action_valid_this_turn = False; step_error_this_turn = None
        turn_player = "N/A"; new_state_desc_this_turn = "N/A"; env_result_string = "" # Specific to FS conversation

        # Get necessary context BEFORE the turn logic
        prompt_context_before_action = env.get_prompt_context() # Get current context dict
        state_before_action = prompt_context_before_action.get('current_state', 'State Unavailable') # Extract state string for logging/history
        new_state_desc_this_turn = state_before_action # Default if turn fails early

        # --- Determine whose turn (relevant for TTT) ---
        is_agent_turn = True; current_env_player = None
        if hasattr(env, 'current_player') and agent_player_mark is not None:
             current_env_player = getattr(env, 'current_player', agent_player_mark) # Get current player from env
             is_agent_turn = (current_env_player == agent_player_mark)
             turn_player = current_env_player
        else: turn_player = "Agent" # Default for non-player tasks

        logger.debug(f"Turn {current_turn_number}: Player = {turn_player} (Is Agent Turn = {is_agent_turn})")

        # ============================ FS Turn (Conversational) ============================
        if is_fs:
            if not is_agent_turn: # Should always be agent turn for FS
                 logger.error("Logic Error: FileSystem environment had non-agent turn?")
                 attempt_failed_prematurely = True; failure_reason = "Internal Logic Error: FS non-agent turn"; break

            turn_player = "Agent"
            # Agent generates action based on conversation history
            # Note: conversation_history contains the initial prompt asking agent to explore
            llm_response_text = agent.generate_action_conversational(conversation_history)

            if llm_response_text is None:
                logger.error(f"FS Agent failed to generate response (API Error?). Failing attempt.")
                attempt_failed_prematurely = True; failure_reason = "Agent LLM failed to respond"
                is_terminal_this_turn = True # Treat API failure as terminal for the loop
                action_taken_this_turn = 'AGENT_API_ERROR'
                env_result_string = "Error: Agent LLM failed to respond." # For history log
            else:
                logger.debug(f"Raw FS Response:\n{llm_response_text}")
                conversation_history.append({'role': 'model', 'parts': [llm_response_text]}) # Add agent response to history

                # Parse command AND completion signal from response
                command, agent_signaled_completion = parse_agent_response(llm_response_text)
                action_taken_this_turn = command if command else ("(Completion Signal)" if agent_signaled_completion else "(No Action Parsed)")
                action_valid_this_turn = False # Assume invalid until successful step or valid completion signal

                if agent_signaled_completion:
                    logger.info("Agent signaled TASK_COMPLETE.")
                    is_terminal_this_turn = True # End the interaction loop
                    action_valid_this_turn = True # Signaling completion is a valid final "action"

                if command:
                    # If agent provided a command, execute it (even if completion was also signaled)
                    logger.info(f"FS Agent proposed command: '{command}'")
                    try:
                        # Execute the command in the environment
                        env_result_string = env.step(command)
                        logger.info(f"Env Result: {env_result_string}")
                        action_valid_this_turn = not env_result_string.startswith("Error:") # Step successful if no error prefix
                        # Get the state description *after* the step
                        new_state_desc_this_turn = env.get_state() # Update state desc based on result
                    except Exception as e:
                        logger.error(f"Environment error during FS step execution: {e}", exc_info=True)
                        attempt_failed_prematurely = True; failure_reason = f"Environment error executing command '{command}': {e}"
                        step_error_this_turn = str(e)
                        is_terminal_this_turn = True # Error terminates loop
                        env_result_string = f"Internal Error executing command: {e}" # Update result string for history
                        action_valid_this_turn = False
                        new_state_desc_this_turn = state_before_action # Revert state desc on error? Or use env's state? Use before state for safety.
                elif not agent_signaled_completion:
                    # No command parsed and no completion signal -> Invalid response
                    logger.warning(f"FS Agent response lacked a command and did not signal completion. Raw response:\n{llm_response_text}")
                    env_result_string = "Error: Agent response did not contain a valid command or completion signal."
                    action_valid_this_turn = False
                    # Consider if this should immediately fail the attempt or allow retry? For now, just log and continue turn, history reflects error.

                # Add environment's result back to conversation history IF the loop isn't terminating yet
                # Avoid adding user message if agent just signaled completion or errored out.
                if not is_terminal_this_turn:
                    conversation_history.append({'role': 'user', 'parts': [env_result_string]})

        # ============================ TTT Turn (Non-Conversational) ============================
        elif is_ttt:
            if is_agent_turn:
                last_invalid_action_feedback = ""; consecutive_invalid_moves_this_turn = 0
                while True: # Inner loop for handling invalid moves within a single agent turn
                    # Prepare prompt context for this attempt
                    current_prompt_context = prompt_context_before_action.copy() # Use state before this turn attempt
                    current_prompt_context['last_invalid_action_feedback'] = last_invalid_action_feedback

                    # Ensure all required keys for the template are present
                    if 'goal' not in current_prompt_context: current_prompt_context['goal'] = env.get_goal()
                    if 'current_player' not in current_prompt_context: current_prompt_context['current_player'] = agent_player_mark

                    try:
                        # Remove feedback placeholder if template doesn't use it
                        template_uses_feedback = '{last_invalid_action_feedback}' in prompts['generate_template']
                        if not template_uses_feedback: current_prompt_context.pop('last_invalid_action_feedback', None)

                        generation_prompt = prompts['generate_template'].format(**current_prompt_context)
                        logger.debug(f"Agent Prompt (Turn {current_turn_number}, Try {consecutive_invalid_moves_this_turn + 1}):\n{generation_prompt}")
                    except KeyError as e:
                        logger.error(f"TTT Prompt formatting error: Missing key '{e}'. Check template.", exc_info=True)
                        attempt_failed_prematurely = True; failure_reason = f"Prompt key error: {e}"
                        step_error_this_turn = failure_reason; action_taken_this_turn = 'PROMPT_ERROR'; is_terminal_this_turn = True; break # Exit inner and outer loop
                    except Exception as e:
                         logger.error(f"TTT Prompt formatting error: {e}", exc_info=True)
                         attempt_failed_prematurely = True; failure_reason = f"Prompt format error: {e}"; step_error_this_turn = failure_reason; action_taken_this_turn = 'PROMPT_ERROR'; is_terminal_this_turn = True; break

                    # Generate action (non-conversational)
                    llm_action = agent.generate_action(generation_prompt)

                    if llm_action is None:
                        logger.error(f"TTT Agent failed to generate action (API Error?). Failing attempt.")
                        attempt_failed_prematurely = True; failure_reason = "Agent LLM failed to respond"
                        step_error_this_turn = failure_reason; action_taken_this_turn = 'AGENT_API_ERROR'; is_terminal_this_turn = True; break # Exit inner and outer loop

                    # Use the standard parser to extract the action (ignoring completion signal for TTT)
                    command, _ = parse_agent_response(llm_action) # TTT doesn't use TASK_COMPLETE signal

                    if command is None: # Handle case where parsing fails or only signal found
                         logger.warning(f"Could not parse command from TTT agent response: {llm_action[:100]}. Treating as invalid.")
                         action_taken_this_turn = llm_action.strip() # Log raw response as action attempt
                         action_valid_this_turn = False
                    else:
                         action_taken_this_turn = command # Log parsed command
                         logger.info(f"Agent proposed action: '{action_taken_this_turn}'")
                         action_valid_this_turn = env.validate_action(action_taken_this_turn) # Check validity

                    # Process valid/invalid action
                    if action_valid_this_turn:
                        try:
                            # Execute valid action
                            new_state_desc_this_turn, is_terminal_this_turn = env.step(action_taken_this_turn)
                            logger.debug(f"Agent action executed. New State:\n{new_state_desc_this_turn}\nTerminal: {is_terminal_this_turn}")
                        except Exception as e:
                            logger.error(f"Environment error during TTT agent step execution: {e}", exc_info=True)
                            attempt_failed_prematurely = True; failure_reason = f"Environment error during agent step: {e}"
                            step_error_this_turn = str(e); is_terminal_this_turn = True
                            new_state_desc_this_turn = state_before_action # Keep state before error
                        break # Exit inner loop (valid action processed or step failed)
                    else:
                        # Handle invalid action
                        consecutive_invalid_moves_this_turn += 1
                        agent_made_any_invalid_move_in_attempt = True # Flag that invalid move occurred
                        logger.warning(f"Invalid TTT action '{action_taken_this_turn}' by agent (Attempt {consecutive_invalid_moves_this_turn}/{MAX_INVALID_MOVES_PER_TURN})")
                        step_error_this_turn = f"Invalid action: {action_taken_this_turn}" # Record error for history
                        new_state_desc_this_turn = state_before_action # State doesn't change on invalid action

                        # Prepare feedback for the next try (if applicable)
                        if template_uses_feedback:
                             last_invalid_action_feedback = f"\nYour previous move ('{action_taken_this_turn}') was invalid. Please choose an empty cell (e.g., 'place X at row,col')."
                        else: last_invalid_action_feedback = "" # Don't include if template doesn't support it

                        # Check if max invalid retries reached
                        if consecutive_invalid_moves_this_turn >= MAX_INVALID_MOVES_PER_TURN:
                            logger.error(f"Failing TTT attempt after {MAX_INVALID_MOVES_PER_TURN} consecutive invalid agent moves.")
                            attempt_failed_prematurely = True; failure_reason = f"{MAX_INVALID_MOVES_PER_TURN} consecutive invalid moves"
                            is_terminal_this_turn = True # Terminate the attempt
                            break # Exit inner loop

                # --- End of Agent's Turn Inner Loop ---
                if is_terminal_this_turn and attempt_failed_prematurely: break # Exit outer loop if prompt error/API error/max invalid moves occurred

            # Opponent's Turn (TTT only, if game not over)
            elif not is_terminal_this_turn: # Check if game ended after agent's move
                 if hasattr(env, 'make_opponent_move') and callable(env.make_opponent_move):
                    logger.info(f"Opponent ({env.opponent_player_mark})'s turn (scripted).")
                    try:
                        opponent_action_str, new_state_desc_this_turn, is_terminal_this_turn = env.make_opponent_move()
                        if opponent_action_str is None:
                            logger.warning(f"Opponent ({env.opponent_player_mark}) failed to find a move (Game should be over?).")
                            # This might indicate a draw state wasn't caught, or an error.
                            # Check env state again just in case.
                            if not env.check_goal_achieved() and not getattr(env,'is_draw',False):
                                logger.error("Opponent failed move but game not terminal. Forcing failure.")
                                attempt_failed_prematurely = True; failure_reason = "Opponent failed to move in non-terminal state"
                                is_terminal_this_turn = True # Treat as terminal failure
                            action_taken_this_turn = "OPPONENT_ERROR"
                            action_valid_this_turn = False
                        else:
                            action_taken_this_turn = opponent_action_str # Record opponent's move
                            action_valid_this_turn = True # Assume scripted move is valid
                            logger.debug(f"Opponent action executed: '{opponent_action_str}'. New State:\n{new_state_desc_this_turn}\nTerminal: {is_terminal_this_turn}")
                    except Exception as e:
                        logger.error(f"Error during opponent's step execution: {e}", exc_info=True)
                        attempt_failed_prematurely = True; failure_reason = f"Error during opponent step: {e}"
                        step_error_this_turn = str(e); action_taken_this_turn = "OPPONENT_EXCEPT"; is_terminal_this_turn = True
                        new_state_desc_this_turn = state_before_action # Revert state description for safety
                 else:
                    # Should not happen if env is TTT
                    logger.error(f"Logic error: TTT Environment instance lacks make_opponent_move method.")
                    attempt_failed_prematurely = True; failure_reason = "Internal logic error: Missing opponent move function"
                    step_error_this_turn = failure_reason; action_taken_this_turn = "LOGIC_ERROR"; is_terminal_this_turn = True
                    new_state_desc_this_turn = state_before_action

        # ============================ Generic Task Turn (Non-Conversational) ============================
        else:
            # Handle other tasks using single-prompt generation (non-conversational)
            turn_player = "Agent"
            try:
                # Prepare context. Use state *before* this turn.
                current_prompt_context = prompt_context_before_action.copy()
                # Ensure required keys (goal, state) are present
                if 'goal' not in current_prompt_context: current_prompt_context['goal'] = env.get_goal()
                if 'current_state' not in current_prompt_context: current_prompt_context['current_state'] = state_before_action

                generation_prompt = prompts['generate_template'].format(**current_prompt_context)
                logger.debug(f"Generic Agent Prompt (Turn {current_turn_number}):\n{generation_prompt}")
            except KeyError as e:
                logger.error(f"Generic Prompt formatting error: Missing key '{e}'. Check template.", exc_info=True)
                attempt_failed_prematurely = True; failure_reason = f"Prompt key error: {e}"; step_error_this_turn = failure_reason; action_taken_this_turn = 'PROMPT_ERROR'; is_terminal_this_turn = True; break # Exit loop
            except Exception as e:
                 logger.error(f"Generic Prompt formatting error: {e}", exc_info=True)
                 attempt_failed_prematurely = True; failure_reason = f"Prompt format error: {e}"; step_error_this_turn = failure_reason; action_taken_this_turn = 'PROMPT_ERROR'; is_terminal_this_turn = True; break # Exit loop

            # Generate action
            llm_action = agent.generate_action(generation_prompt)

            if llm_action is None:
                logger.error(f"Generic Agent failed to generate action (API Error?). Failing attempt.")
                attempt_failed_prematurely = True; failure_reason = "Agent LLM failed to respond"
                step_error_this_turn = failure_reason; action_taken_this_turn = 'AGENT_API_ERROR'; is_terminal_this_turn = True; break # Exit loop

            # Parse action and completion signal
            command, agent_signaled_completion = parse_agent_response(llm_action)
            action_taken_this_turn = command if command else ("(Completion Signal)" if agent_signaled_completion else "(No Action Parsed)")
            action_valid_this_turn = False # Assume invalid initially

            if agent_signaled_completion:
                 logger.info("Agent signaled TASK_COMPLETE for generic task.")
                 is_terminal_this_turn = True # End interaction loop
                 action_valid_this_turn = True # Valid final action

            if command:
                logger.info(f"Agent proposed action: '{command}'")
                action_valid_this_turn = env.validate_action(command) # Validate command syntax/structure

                if not action_valid_this_turn:
                    logger.warning(f"Invalid action proposed by agent for generic task: '{command}'. Failing attempt.")
                    # For generic tasks, typically fail immediately on invalid action unlike TTT's retry.
                    attempt_failed_prematurely = True; failure_reason = f"Invalid action: {command}"; step_error_this_turn = failure_reason; is_terminal_this_turn = True
                    new_state_desc_this_turn = state_before_action # State doesn't change
                else:
                    # Execute valid action
                    try:
                        new_state_desc_this_turn, step_terminal = env.step(command) # Execute step
                        is_terminal_this_turn = is_terminal_this_turn or step_terminal # Become terminal if step says so OR agent signaled
                        logger.debug(f"Generic action executed. New State:\n{new_state_desc_this_turn}\nStep Terminal: {step_terminal}, Overall Terminal: {is_terminal_this_turn}")
                    except Exception as e:
                        logger.error(f"Environment error during generic step execution: {e}", exc_info=True)
                        attempt_failed_prematurely = True; failure_reason = f"Environment error executing action: {e}"
                        step_error_this_turn = str(e); is_terminal_this_turn = True
                        new_state_desc_this_turn = state_before_action # Revert state description
            elif not agent_signaled_completion:
                 # No command, no signal - invalid response
                 logger.warning("Generic Agent response lacked command/completion signal.")
                 # Consider failing attempt? For now, log and continue turn.
                 step_error_this_turn = "Invalid agent response (no command/signal)"
                 action_valid_this_turn = False


        # --- Post-Turn Processing & History Logging ---
        # Assess intermediate status for regression tracking (if implemented by env)
        intermediate_status = env.assess_intermediate_status()
        regression = evaluator.track_regression(attempt_id, intermediate_status, turn_count)
        if regression: regressions_detected_in_attempt = True

        # Log turn details to history list
        turn_data = {
            'turn': current_turn_number,
            'player': turn_player, # Who acted this turn
            'state_before_action': state_before_action, # State before the action was taken
            'action_taken': action_taken_this_turn, # The action string proposed/executed
            'action_valid': action_valid_this_turn, # Was the action valid syntactically/semantically?
            'state_after_action': new_state_desc_this_turn, # State after the action (or before if error/invalid)
            'intermediate_status': str(intermediate_status) if intermediate_status is not None else None, # Comparable status for regression
            'regression_detected_this_turn': regression, # Did status regress this turn?
            'is_terminal_after_turn': is_terminal_this_turn, # Did this turn result in a terminal state?
            'error_this_turn': step_error_this_turn # Any error message from this turn's execution/validation
        }
        # Add FS-specific conversational context if applicable
        if is_fs: turn_data['env_result_string'] = env_result_string # The string returned by FS env.step()

        history.append(turn_data)


        # --- Check for Terminal State to Exit Main Loop ---
        if is_terminal_this_turn:
            if not attempt_failed_prematurely: logger.info(f"Environment reached a terminal state or agent signaled completion.")
            else: logger.warning(f"Terminating attempt {attempt_id+1} due to failure: {failure_reason or step_error_this_turn or 'Unknown reason'}")
            break # Exit the main interaction loop

        # Increment turn counter only if loop continues
        turn_count += 1

    # --- Attempt End ---
    end_time = time.time(); duration = end_time - start_time
    logger.info(f"--- Attempt {attempt_id + 1} Finished (Duration: {duration:.2f}s, Turns: {len(history)}) ---")

    # --- Final Score Assignment ---
    final_score = 1 # Default to Fail
    final_eval_response = "" # Raw response or justification
    try:
        final_eval_input = env.get_final_eval_input() # Get final state representation for eval
    except Exception as e:
        logger.error(f"Failed to get final evaluation input from environment: {e}", exc_info=True)
        final_eval_input = f"Error getting final state: {e}"
        # Ensure failure state propagates if getting eval input fails
        # We check attempt_failed_prematurely later, so this just sets the reason
        if not attempt_failed_prematurely:
            failure_reason = f"Failed to get final eval input: {e}"
            # Avoid setting attempt_failed_prematurely = True here directly,
            # let the main check handle it to avoid potential double-flagging issues.
            # If getting eval input fails, the evaluation logic below will likely
            # result in score 1 anyway, or the premature flag check handles it.


    # --- Evaluation Logic ---
    if attempt_failed_prematurely:
        # Attempt failed due to error, max steps, invalid moves etc. Score is 1.
        final_score = 1
        final_eval_response = f"Fail (Attempt ended prematurely: {failure_reason or 'Unknown reason'})"
        logger.info(f"Final Score: 1 (Attempt Failed: {failure_reason or 'Unknown reason'})")
    else:
        # Attempt completed normally (reached terminal state or agent signaled completion)
        # Now evaluate the outcome based on the environment type.

        # --- TTT Deterministic Eval ---
        if isinstance(env, TicTacToeEnv):
            logger.info(f"Performing TTT deterministic evaluation.")
            agent_won = env.winner == env.agent_player_mark
            opponent_won = env.winner == env.opponent_player_mark
            is_draw = getattr(env, 'is_draw', False)

            if opponent_won:
                final_score = 1; final_eval_response = f"Fail (Opponent '{env.opponent_player_mark}' won)"
                logger.info(f"Score: 1 ({final_eval_response})")
            elif (agent_won or is_draw):
                 outcome = "Agent Win" if agent_won else "Draw"
                 if not agent_made_any_invalid_move_in_attempt:
                     final_score = 3; final_eval_response = f"Success ({outcome} with no invalid moves)"
                     logger.info(f"Score: 3 ({final_eval_response})")
                 else:
                     final_score = 2; final_eval_response = f"Partial ({outcome} achieved, but with invalid moves during attempt)"
                     logger.info(f"Score: 2 ({final_eval_response})")
            else:
                 # Should not happen if game terminated normally without winner/draw
                 logger.error(f"TTT evaluation inconclusive: Game ended but no winner/draw detected? State:\n{final_eval_input}")
                 final_score = 1; final_eval_response = "Fail (Inconclusive final state)"
                 logger.info(f"Score: 1 ({final_eval_response})")

        # --- FS Deterministic Eval ---
        elif isinstance(env, FileSystemEnv):
            logger.info(f"Performing FS deterministic evaluation.")
            try:
                # Use the environment's internal evaluation method
                final_score = env.evaluate_final_state() # Env returns 1, 2, or 3 directly
                if final_score == 3: final_eval_response = "Success (Final state matches goal criteria)"
                elif final_score == 2: final_eval_response = "Partial (Final state partially matches goal criteria)"
                else: final_eval_response = "Fail (Final state does not meet goal criteria)"
                logger.info(f"Score: {final_score} ({final_eval_response})")
            except Exception as e:
                logger.error(f"Error during FileSystem deterministic evaluation: {e}", exc_info=True)
                final_score = 1; final_eval_response = "Fail (Error during deterministic evaluation)"
                logger.info(f"Score: 1 ({final_eval_response})")

        # --- Other Tasks: LLM Eval ---
        else:
            logger.info(f"Performing LLM evaluation for task '{task_name}'.")
            # Ensure finaleval_template exists
            if not prompts.get('finaleval_template'):
                 logger.error(f"Cannot perform LLM evaluation: Missing finaleval_template for task '{task_name}'.")
                 final_score = 1; final_eval_response = "Fail (Missing evaluation prompt)"
            else:
                 logger.debug(f"LLM Eval Input:\n{final_eval_input}")
                 try:
                     # Check if evaluator is available (might not be needed for deterministic)
                     if evaluator and hasattr(evaluator, 'evaluate_final_outcome'):
                         final_score, final_eval_response_raw = evaluator.evaluate_final_outcome(final_eval_input)
                         final_eval_response = f"LLM Eval Raw: {final_eval_response_raw}" # Store raw response for inspection
                         logger.info(f"LLM evaluation completed. Parsed Score: {final_score}")
                         logger.debug(f"LLM Raw Eval Response: {final_eval_response_raw}")
                     else:
                         logger.error("LLM Evaluator not available or configured correctly.")
                         final_score = 1; final_eval_response = "Fail (LLM Evaluator unavailable)"

                 except Exception as e:
                     logger.error(f"Error during LLM evaluation call: {e}", exc_info=True)
                     final_score = 1; final_eval_response = f"Fail (LLM evaluation error: {e})"
                     logger.info(f"Score: 1 ({final_eval_response})")

    # --- Compile final result dictionary ---
    # FIX: Ensure 'failed' flag is False if score is 3, otherwise use attempt_failed_prematurely
    is_successful_run = (final_score == 3 and not attempt_failed_prematurely)
    final_failed_flag = False if is_successful_run else attempt_failed_prematurely
    # If the run wasn't flagged as failed prematurely, but the score is 1 or 2, it's still not a premature failure,
    # it's a completed run with a non-success score.
    # Ensure the failure_reason is only set if the final_failed_flag is True
    final_failure_reason = failure_reason if final_failed_flag else ""


    result = {
        'attempt_id': attempt_id + 1,
        'success': final_score == 3, # Boolean flag for strict success based on score only
        'score': final_score, # Numerical score (1, 2, or 3)
        # Use the corrected flag that respects the score for successful runs
        'failed': final_failed_flag,
        'failure_reason': final_failure_reason, # Reason only if failed=True
        'final_outcome_description': final_eval_response, # Justification or raw LLM eval response
        'regressions_detected': regressions_detected_in_attempt, # Were any regressions tracked?
        'agent_made_invalid_moves': agent_made_any_invalid_move_in_attempt if is_ttt else None, # Specific to TTT
        'turns_completed': len(history), # How many turns were actually executed
        'duration_seconds': duration, # Wall clock time for the attempt
        'final_eval_input': final_eval_input, # The state/data provided to the evaluator
        'history': history, # Detailed turn-by-turn log
    }
    return result


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="ToyBench: LLM Agentic Benchmark Suite")
    parser.add_argument("-t", "--task", required=True, help="Task name (e.g., file_system, tic_tac_toe)")
    parser.add_argument("-p", "--provider", default="gemini", help="LLM Provider (e.g., gemini)")
    parser.add_argument("-m", "--model", default=None, help="Agent LLM model name. Defaults to provider default.")
    parser.add_argument("-n", "--attempts", type=int, default=5, help="Number of independent attempts to run")
    parser.add_argument("-s", "--steps", type=int, default=10, help="Max steps/turns per attempt (ignored for tasks like TTT & FileSystem)")
    parser.add_argument("--evaluator_model", default=None, help="Evaluator LLM model (defaults to agent model or provider default). Not used for deterministic tasks like TTT/FS.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--output_dir", default="results", help="Base directory for saving results")
    args = parser.parse_args()

    # --- Config & Setup ---
    try:
        config = load_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        exit(1)

    # Determine agent model name
    agent_model = args.model
    if not agent_model:
        # Construct default model key based on provider (e.g., 'default_gemini_model')
        default_model_key = f'default_{args.provider.lower()}_model'
        agent_model = config.get(default_model_key)
        if not agent_model:
             # Fallback to a generic default if provider-specific default isn't set
             agent_model = 'gemini-1.5-flash-latest' # Or some other reasonable default
             print(f"Warning: Agent model not specified and '{default_model_key}' not found in config. Using fallback '{agent_model}'.")

    # Determine evaluator model name (only relevant for non-deterministic tasks)
    evaluator_model_name = args.evaluator_model if args.evaluator_model else config.get('evaluator_model', agent_model)


    # Create unique output directory for this run
    output_dir = create_output_dir(args.output_dir, args.task, args.provider, agent_model)
    log_file = os.path.join(output_dir, "run.log")
    setup_logging(log_level=getattr(logging, args.log_level.upper()), log_file=log_file)

    logger.info("--- ToyBench Run Initializing ---")
    logger.info(f"Task: {args.task}, Provider: {args.provider}, Agent Model: {agent_model}, Attempts: {args.attempts}")
    logger.info(f"Evaluator Model: {evaluator_model_name} (Used for non-deterministic tasks)")
    logger.info(f"Max Steps: {args.steps} (May be ignored by task)")
    logger.info(f"Log Level: {args.log_level}, Output Directory: {output_dir}")
    # Log API key status carefully
    api_key_name = f'{args.provider.lower()}_api_key'
    api_key_present = bool(config.get(api_key_name))
    logger.info(f"API Key ({api_key_name}): {'Loaded' if api_key_present else 'NOT FOUND'}")


    # --- Initialization ---
    try:
        prompts = load_task_prompts(args.task, config.get('task_definitions_dir', 'tasks'))
        api_key = config.get(api_key_name)
        if not api_key: raise ValueError(f"API Key '{api_key_name}' not found in config or environment variables.")

        agent_llm = get_llm_interface(args.provider, api_key, agent_model)
        # Initialize evaluator LLM interface only if needed (non-deterministic tasks)
        # For now, initialize always, but Evaluator class itself might check if needed.
        evaluator_llm = get_llm_interface(args.provider, api_key, evaluator_model_name)

        environment = get_environment(args.task, prompts.get('goal_description', '')) # Pass goal description
        evaluator = Evaluator(evaluator_llm, prompts.get('finaleval_template', '')) # Pass final eval template

    except FileNotFoundError as e: logger.error(f"Initialization failed: Task files not found. {e}", exc_info=True); print(f"Initialization Error: {e}"); exit(1)
    except ValueError as e: logger.error(f"Initialization failed: Configuration or value error. {e}", exc_info=True); print(f"Initialization Error: {e}"); exit(1)
    except Exception as e: logger.error(f"Initialization failed: An unexpected error occurred. {e}", exc_info=True); print(f"Initialization Error: {e}"); exit(1)

    # --- Run Attempts ---
    logger.info("--- Starting Benchmark Attempts ---")
    all_results = []
    start_time_attempt = 0 # Define outside loop for except block scope
    history_attempt = [] # Define outside loop for except block scope
    for i in range(args.attempts):
        try:
            start_time_attempt = time.time() # Capture start time for potential error duration calc
            history_attempt = [] # Reset history for this attempt's potential error log
            attempt_result = run_attempt(i, environment, agent_llm, evaluator, args.steps, prompts, args.task)
            all_results.append(attempt_result)
        except KeyboardInterrupt:
            logger.warning(f"--- Run interrupted by user during attempt {i+1} ---")
            print("\nRun interrupted.")
            # Append partial failure result for the interrupted attempt
            all_results.append({
                'attempt_id': i + 1, 'score': 1, 'failed': True, 'failure_reason': "User interrupt",
                'success': False, 'final_outcome_description': "Interrupted by user", 'regressions_detected': False,
                'turns_completed': len(history_attempt), # Use history captured within this loop iteration
                'agent_made_invalid_moves': None,
                'duration_seconds': time.time() - start_time_attempt,
                'history': history_attempt, 'final_eval_input': 'N/A'})
            break # Exit the loop
        except Exception as e:
            # Catch critical errors during the attempt run itself
            logger.error(f"--- CRITICAL ERROR during attempt {i+1}: {e} ---", exc_info=True)
            # Append a failure result
            all_results.append({
                'attempt_id': i + 1, 'score': 1, 'failed': True, 'failure_reason': f"Unhandled exception in run_attempt: {e}",
                'success': False, 'final_outcome_description': f"Unhandled exception: {e}", 'regressions_detected': False,
                'turns_completed': len(history_attempt),
                'agent_made_invalid_moves': None,
                'duration_seconds': time.time() - start_time_attempt,
                'history': history_attempt, 'final_eval_input': 'N/A'})
            # Optionally continue to next attempt or break? Let's continue for now.

    # --- Report & Save ---
    logger.info("--- Benchmark Run Finished ---")
    if not all_results:
        logger.warning("No attempts were completed.")
        print("No results to report.")
        exit(0)

    metrics = calculate_metrics(all_results, args.attempts) # Pass requested attempts for context
    report = format_report(metrics, args.task, args.provider, agent_model, args.steps)

    print("\n" + report + "\n")
    logger.info(f"Final Report:\n{report}")

    # Prepare config args for saving (include resolved model names)
    run_config_args = vars(args).copy() # Make a copy to modify
    run_config_args['agent_model_used'] = agent_model
    run_config_args['evaluator_model_used'] = evaluator_model_name
    # Add API key presence for record, but not the key itself
    run_config_args[f'{args.provider.lower()}_api_key_loaded'] = api_key_present

    try:
        save_results(output_dir, all_results, report, run_config_args)
    except Exception as e:
        logger.error(f"Failed to save results to {output_dir}: {e}", exc_info=True)
        print(f"Error saving results: {e}")

    logger.info(f"Results and logs saved in: {output_dir}")
    logger.info("--- ToyBench Run Complete ---")


if __name__ == "__main__":
    main()