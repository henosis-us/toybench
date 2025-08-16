# toybench_cli.py
# VERSION MODIFIED TO HANDLE SOLAR_GEN CONVERSATIONALLY + Specific Tag Error Handling + OpenAI provider
# VERSION MODIFIED TO CORRECTLY DETERMINE EVALUATOR PROVIDER INDEPENDENTLY
# VERSION MODIFIED TO ADD XAI GROK PROVIDER SUPPORT WITH REASONING EFFORT
# FIXED: API key name mismatch for Grok provider (mapped 'grok' to 'xai_api_key')
# ADDED: Quality Compute provider support with hardcoded URL in llm_interface.py
# ADDED: Anthropic Claude support
# UPDATED: Passing 'task_name' to calculate_metrics for task-specific metrics like Differentiated Score in SolarGen.
# ADDED: --thinking and --thinking_budget arguments for Anthropic extended thinking support
# UPDATED: Added --max_tokens argument and support for passing max_tokens to LLM interfaces and calls
# UPDATED: Modified get_llm_interface to pass thinking_enabled and thinking_budget to GeminiInterface for thinkingBudget support.
# ADDED: Kimi (Moonshot AI) provider support
# ### FINAL MAJOR UPDATE ### All Quality Compute inference style logic (Best-of-N, Collaborative)
# ### is now entirely encapsulated within the QualityComputeInterface itself.
# ### The CLI and run_attempt function remain simple and agnostic.
# ADDED: OpenRouter provider support with allow_fallbacks and sort options. Added reasoning_effort for OpenRouter compatibility. Fixed flag handling.
# UPDATED: Added --openai_reasoning_effort argument for configuring reasoning effort in OpenAI models.
# ADDED: --openai_background and --openai_bg_poll flags for OpenAI background (async) mode support with polling.
# UPDATED: Added --or_provider to force a specific upstream provider for OpenRouter (e.g., 'fireworks') and disable fallbacks.

import argparse
import logging
import os
import json
import time
import re

from config import load_config
from utils import setup_logging, create_output_dir, parse_llm_score

# Import Interfaces and Base Classes
from llm_interface import LLMInterface, GeminiInterface, OpenAIInterface, GrokInterface, QualityComputeInterface, AnthropicInterface, KimiInterface, OpenRouterInterface
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
            if key in ['goal_description', 'generate_template', 'finaleval_template'] and key != 'intermediate_eval_template':
                if not (key == 'finaleval_template' and task_name in ["tic_tac_toe", "file_system"]):
                    logger.warning(f"Potentially required prompt file not found: {file_path}")

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
        except ImportError as e:
            logger.error(f"Failed to initialize SolarSystemEnv due to missing browser utilities: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing SolarSystemEnv: {e}", exc_info=True)
            raise
    else:
        raise ValueError(f"Unknown task name: {task_name}")

# --- LLM Provider Factory ---
def get_llm_interface(provider_name: str, api_key: str, model_name: str, thinking_enabled: bool = False, thinking_budget: int = 16000, max_tokens: int = None, **kwargs) -> LLMInterface:
    """Factory function to instantiate the correct LLM interface."""
    p = provider_name.lower()
    if p == "gemini":
        return GeminiInterface(api_key=api_key, model_name=model_name, provider_name="gemini",
                               thinking_enabled=thinking_enabled, thinking_budget=thinking_budget)
    elif p == "openai":
        return OpenAIInterface(api_key=api_key, model_name=model_name, provider_name="openai",
                               reasoning_effort=kwargs.get('openai_reasoning_effort', 'high'),
                               background_enabled=kwargs.get('openai_background', False),
                               background_poll_interval=kwargs.get('openai_bg_poll', 2.0))
    elif p == "grok":
        return GrokInterface(api_key=api_key, model_name=model_name, provider_name="grok", reasoning_effort=kwargs.get('reasoning_effort', 'low'))
    elif p == "quality_compute":
        return QualityComputeInterface(api_key=api_key, model_name=model_name, provider_name="quality_compute", **kwargs)
    elif p == "anthropic":
        return AnthropicInterface(api_key=api_key, model_name=model_name, provider_name="anthropic", thinking_enabled=thinking_enabled, thinking_budget=thinking_budget)
    elif p == "kimi":
        return KimiInterface(api_key=api_key, model_name=model_name, provider_name="kimi")
    elif p == "openrouter":
        return OpenRouterInterface(api_key=api_key, model_name=model_name, provider_name="openrouter",
                                   allow_fallbacks=kwargs.get('or_allow_fallbacks', True),
                                   sort=kwargs.get('or_sort', 'price'),
                                   reasoning_effort=kwargs.get('or_reasoning_effort', 'low'),
                                   provider=kwargs.get('or_provider'))
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")

# --- Action Parsing Helper ---
def parse_agent_response(response_text: str, task_name: str) -> tuple[str | None, bool]:
    """Parses agent response based on the task type."""
    if response_text is None:
        return None, False

    if task_name == "solar_gen":
        match = re.search(r"<solar\.html>(.*?)</solar\.html>", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            code_content = match.group(1).strip()
            logger.debug("Parsed action (solar_gen): Extracted content from <solar.html> tags.")
            return code_content, False
        else:
            if "<solar.html>" in response_text and "</solar.html>" not in response_text:
                logger.error("Solar task specific error: Agent response missing closing </solar.html> tag.")
                return None, False
            elif "</solar.html>" in response_text and "<solar.html>" not in response_text:
                logger.error("Solar task specific error: Agent response missing opening <solar.html> tag.")
                return None, False
            else:
                logger.warning("Could not find or parse <solar.html>...</solar.html> tags in response for solar_gen.")
                return None, False

    task_complete = "TASK_COMPLETE" in response_text.upper()
    action_match = re.search(r"```action\n(.*?)\n```", response_text, re.DOTALL | re.IGNORECASE)
    if action_match:
        command = action_match.group(1).strip()
        logger.debug(f"Parsed command (block): '{command}'")
        if "TASK_COMPLETE" in command.upper() and len(command.split()) == 1:
            logger.debug("Parsed signal (TASK_COMPLETE in action block).")
            return None, True
        return command, task_complete

    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if "TASK_COMPLETE" in first_line.upper() and len(first_line.split()) == 1:
            logger.debug("Parsed signal (TASK_COMPLETE line).")
            return None, True
        
        common_cmds = ["ls", "cd", "mkdir", "touch", "rm", "pwd", "mv", "place", "cat", "cp", "echo"]
        first_word = first_line.split(maxsplit=1)[0].lower() if first_line else ""
        if first_word in common_cmds:
            logger.debug(f"Parsed command (first line heuristic): '{first_line}'")
            return first_line, task_complete

    logger.debug(f"No command parsed from response (Original Logic Path). Signal status: {task_complete}. Response: {response_text[:100]}...")
    return None, task_complete

# --- run_attempt function remains unchanged ---
def run_attempt(attempt_id: int,
                env: BaseEnvironment,
                agent: LLMInterface,
                evaluator: Evaluator,
                max_rounds: int,
                prompts: dict,
                task_name: str,
                output_dir: str,
                max_tokens: int = None) -> dict:
    """Runs a single attempt, handling specific logic for TTT, FS, and Solar."""
    logger.info(f"--- Starting Attempt {attempt_id + 1} (Task: {task_name}, Max Rounds/Steps: {max_rounds}) ---")
    evaluator.reset_attempt_tracking(attempt_id)

    start_time = time.time()
    initial_state_desc = env.reset()
    logger.debug(
        f"Environment reset. Initial state description (truncated): {str(initial_state_desc)[:200]}..."
    )

    history = []
    conversation_history = []
    regressions_detected_in_attempt = False
    premature_failure = False
    failure_reason = ""
    final_eval_input_summary = "N/A"
    agent_made_any_invalid_move_in_attempt = False
    
    agent_player_mark = env.get_agent_player_mark()

    turn_count = 0
    is_ttt = isinstance(env, TicTacToeEnv)
    is_fs = isinstance(env, FileSystemEnv)
    is_solar = isinstance(env, SolarSystemEnv)

    if is_ttt:
        logger.info(f"Task '{task_name}' runs until game completion or error. --rounds ({max_rounds}) ignored.")
        effective_max_rounds = 100
    elif is_fs:
        logger.info(f"Task '{task_name}' runs until completion signal, error, or max_rounds. --rounds ({max_rounds}) acts as step limit.")
        effective_max_rounds = max_rounds
    elif is_solar:
        logger.info(f"Task '{task_name}' runs for exactly {max_rounds} refinement rounds (steps).")
        effective_max_rounds = max_rounds
    else:
        logger.info(f"Task '{task_name}' runs for max {max_rounds} steps or until completion signal/error.")
        effective_max_rounds = max_rounds

    use_conversational_api = is_fs or is_solar
    if use_conversational_api:
        logger.info(f"Using conversational API path for agent in task '{task_name}'.")
    else:
        logger.info(f"Using single-shot API path for agent in task '{task_name}'.")

    try:
        initial_context = env.get_prompt_context()
        if 'goal' not in initial_context:
            initial_context['goal'] = prompts.get('goal_description', 'Goal not provided.')
        
        template = prompts.get('generate_template')
        if not template:
            raise ValueError("Generate template prompt is missing.")

        required_keys = set(re.findall(r"\{(\w+)\}", template))
        missing_keys = required_keys - set(initial_context.keys())
        if missing_keys:
            logger.warning(f"Initial prompt context missing keys required by template: {missing_keys}. Attempting to proceed with placeholders.")
            for key in missing_keys:
                initial_context[key] = f"<{key}_placeholder>"

        generation_prompt_text = template.format(**initial_context)

        if use_conversational_api:
            conversation_history.append({'role': 'user', 'parts': [generation_prompt_text]})
            logger.debug(f"Initialized conversational history for {task_name}. First user message created.")
        else:
            logger.debug(f"Initial Non-Conversational Prompt prepared for task {task_name}.")
    except Exception as e:
        logger.error(f"Error during initial prompt setup: {e}", exc_info=True)
        premature_failure = True
        failure_reason = f"Initial prompt error: {e}"

    env_feedback_or_result = ""

    while not premature_failure:
        if turn_count >= effective_max_rounds:
            logger.warning(f"Reached maximum allowed rounds/steps ({effective_max_rounds}) for task '{task_name}'. Terminating attempt.")
            if not is_solar:
                failure_reason = f"Reached max_rounds ({effective_max_rounds}) without success signal"
            break

        current_turn_number = turn_count + 1
        log_prefix = f"Attempt {attempt_id + 1}, Round/Turn {current_turn_number}"
        if not is_ttt:
            log_prefix += f"/{effective_max_rounds}"
        logger.info(f"--- {log_prefix} ---")

        is_terminal_this_turn = False
        action_taken_this_turn = "N/A"
        action_valid_this_turn = False
        step_error_this_turn = None
        new_state_desc_this_turn = "N/A"
        turn_player = "Agent"
        current_turn_feedback = ""
        
        token_usage_this_turn = None
        raw_api_response_this_turn = None

        prompt_context_before_action = env.get_prompt_context()
        if 'goal' not in prompt_context_before_action:
            prompt_context_before_action['goal'] = prompts.get('goal_description', '')
        state_before_action_desc = str(prompt_context_before_action.get('current_state', 'State Unavailable'))
        new_state_desc_this_turn = state_before_action_desc

        if is_ttt:
            current_env_player = getattr(env, 'current_player', agent_player_mark)
            is_agent_turn = (current_env_player == agent_player_mark)
            turn_player = current_env_player
            if not is_agent_turn:
                logger.debug(f"Turn {current_turn_number}: Skipping agent action, it's Opponent ({turn_player})'s turn.")
                turn_count += 1
                continue
            else:
                logger.debug(f"Turn {current_turn_number}: Agent ({turn_player})'s turn.")

        agent_response_text = None
        try:
            if use_conversational_api:
                if turn_count > 0 and env_feedback_or_result:
                    if conversation_history and conversation_history[-1]['role'] == 'model':
                        conversation_history.append({'role': 'user', 'parts': [env_feedback_or_result]})
                        logger.debug("Appended environment feedback from previous turn to conversation history.")
                    else:
                        logger.warning("Last message in history was not from 'model'. Skipping append of env feedback to avoid user->user sequence.")
                if not conversation_history:
                    logger.error(f"Conversational history is unexpectedly empty before calling agent for {task_name}.")
                    raise ValueError("Cannot generate action with empty conversational history.")
                logger.debug(f"Calling conversational agent ({agent.model_name}) for {task_name}. History length: {len(conversation_history)}")
                agent_response_text, token_usage_this_turn, raw_api_response_this_turn = agent.generate_action_conversational(conversation_history, max_tokens=max_tokens)
            else:
                current_context = prompt_context_before_action
                template = prompts['generate_template']
                required_keys = set(re.findall(r"\{(\w+)\}", template))
                missing_keys = required_keys - set(current_context.keys())
                if missing_keys:
                    logger.warning(f"Turn {current_turn_number}: Prompt context missing keys {missing_keys}. Using defaults.")
                    if is_ttt:
                        current_context.setdefault('last_invalid_action_feedback', '')
                    for key in missing_keys:
                        current_context.setdefault(key, f"<{key}_placeholder>")
                generation_prompt = template.format(**current_context)
                logger.debug(f"Agent Prompt (Turn {current_turn_number}, Task: {task_name}, Non-Conversational).")
                agent_response_text, token_usage_this_turn, raw_api_response_this_turn = agent.generate_action(generation_prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Agent generation failed during turn {current_turn_number}: {e}", exc_info=True)
            step_error_this_turn = f"Agent generation API error: {e}"
            is_terminal_this_turn = True
            premature_failure = True
            failure_reason = step_error_this_turn

        if agent_response_text is None and not premature_failure:
            logger.error(f"Agent ({agent.model_name}) returned None response for turn {current_turn_number}.")
            logger.error(f"Raw API Response when text was None: {raw_api_response_this_turn}")
            step_error_this_turn = "Agent LLM returned no response"
            is_terminal_this_turn = True
            premature_failure = True
            failure_reason = step_error_this_turn
            action_taken_this_turn = 'AGENT_API_ERROR'
        if premature_failure:
            break

        logger.debug(f"Raw Agent Response (Turn {current_turn_number}): {str(agent_response_text)}")
        if use_conversational_api:
            if isinstance(agent_response_text, str):
                conversation_history.append({'role': 'model', 'parts': [agent_response_text]})
            else:
                logger.warning(f"Cannot append non-string agent response to conversational history (Type: {type(agent_response_text)}). Skipping append for turn {current_turn_number}.")

        action_content, agent_signaled_completion = parse_agent_response(agent_response_text, task_name)
        improperly_closed_tags = False
        if action_content is not None:
            log_action = action_content[:100] + ('...' if len(action_content) > 100 else '')
            action_taken_this_turn = log_action.replace('\n', '\\n')
            logger.info(f"Agent proposed action/output: '{action_taken_this_turn}'")
        elif agent_signaled_completion:
            action_taken_this_turn = "(Completion Signal: TASK_COMPLETE)"
            logger.info("Agent signaled TASK_COMPLETE.")
        else:
            action_taken_this_turn = "(No Action Parsed)"
            logger.warning(f"Could not parse valid action/output from agent response for turn {current_turn_number}.")
            action_valid_this_turn = False
            step_error_this_turn = "Failed to parse action/output from agent response"
            if is_solar and isinstance(agent_response_text, str):
                if "<solar.html>" in agent_response_text and "</solar.html>" not in agent_response_text:
                    improperly_closed_tags = True
                    step_error_this_turn = "Agent output missing closing </solar.html> tag"
                elif "</solar.html>" in agent_response_text and "<solar.html>" not in agent_response_text:
                    improperly_closed_tags = True
                    step_error_this_turn = "Agent output missing opening <solar.html> tag"
            if is_ttt or is_fs:
                agent_made_any_invalid_move_in_attempt = True

        should_execute_step = False
        current_turn_feedback = ""
        if is_solar:
            if action_content is not None and not improperly_closed_tags:
                should_execute_step = True
            else:
                current_turn_feedback = step_error_this_turn or "Skipping step due to parsing error/invalid output format."
                action_valid_this_turn = False
        elif action_content is not None:
            should_execute_step = True
        else:
            current_turn_feedback = step_error_this_turn or "Skipping step: No action parsed from agent response."
            action_valid_this_turn = False

        if should_execute_step:
            try:
                if not is_solar:
                    action_valid_this_turn = env.validate_action(action_content)
                    if not action_valid_this_turn:
                        logger.warning(f"Action '{action_taken_this_turn}' failed environment validation.")
                        step_error_this_turn = f"Invalid action based on environment rules: {action_content}"
                        current_turn_feedback = step_error_this_turn
                        if is_ttt:
                            agent_made_any_invalid_move_in_attempt = True
                            logger.error("TTT: Agent made invalid move. Failing attempt.")
                            premature_failure = True
                            failure_reason = step_error_this_turn
                            is_terminal_this_turn = True
                        elif is_fs:
                            logger.error(f"FS: Agent action '{action_taken_this_turn}' invalid. Failing attempt.")
                            premature_failure = True
                            failure_reason = step_error_this_turn
                            is_terminal_this_turn = True
                        should_execute_step = False

                if should_execute_step and not premature_failure:
                    logger.debug(f"Executing env.step with action for task {task_name}...")
                    if is_fs:
                        step_result_output = env.step(action_content)
                        is_terminal_from_step = False
                        action_valid_this_turn = not str(step_result_output).startswith("Error:")
                        current_turn_feedback = step_result_output
                        new_state_desc_this_turn = env.get_state()
                    else:
                        step_result_output, is_terminal_from_step = env.step(action_content if action_content is not None else "")
                        action_valid_this_turn = True
                        current_turn_feedback = step_result_output
                        new_state_desc_this_turn = step_result_output
                    
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
                action_valid_this_turn = False
                new_state_desc_this_turn = state_before_action_desc
                current_turn_feedback = step_error_this_turn
        else:
            logger.warning(f"Skipping environment step execution for Turn {current_turn_number} due to previous error/invalid action.")
            action_valid_this_turn = False
            if not current_turn_feedback:
                current_turn_feedback = step_error_this_turn or "Step skipped due to error or invalid action."
            new_state_desc_this_turn = state_before_action_desc

        if agent_signaled_completion and not is_terminal_this_turn:
            logger.info("Agent signaled TASK_COMPLETE, marking turn as terminal.")
            is_terminal_this_turn = True

        intermediate_status = None
        if not is_ttt and not is_solar:
            intermediate_status = env.assess_intermediate_status()
            regression = evaluator.track_regression(attempt_id, intermediate_status, turn_count)
            if regression:
                regressions_detected_in_attempt = True
        else:
            regression = False
                
        turn_data = {
            'turn': current_turn_number,
            'player': turn_player,
            'state_before_action': state_before_action_desc,
            'agent_raw_response': agent_response_text if turn_player == "Agent" else None,
            'agent_raw_api_response': raw_api_response_this_turn if turn_player == "Agent" else None,
            'action_parsed': action_content if action_content is not None else None,
            'action_taken_for_log': action_taken_this_turn,
            'action_valid': action_valid_this_turn,
            'agent_signaled_completion': agent_signaled_completion,
            'env_feedback_or_result': current_turn_feedback,
            'state_after_action': new_state_desc_this_turn,
            'intermediate_status': str(intermediate_status) if intermediate_status is not None else None,
            'regression_detected_this_turn': regression,
            'is_terminal_after_turn': is_terminal_this_turn,
            'error_this_turn': step_error_this_turn,
            'token_usage': token_usage_this_turn,
        }
        history.append(turn_data)

        env_feedback_or_result = current_turn_feedback
        if is_terminal_this_turn:
            if not premature_failure:
                logger.info(f"Environment reached terminal state or agent signaled completion at turn {current_turn_number}.")
            else:
                logger.warning(f"Terminating attempt {attempt_id+1} early due to failure: {failure_reason or step_error_this_turn or 'Unknown reason'}")
            break

        turn_count += 1
        
        if is_ttt and not is_terminal_this_turn:
            opponent_player = getattr(env, 'opponent_player_mark', 'O')
            if getattr(env, 'current_player', None) == opponent_player:
                logger.info(f"--- Opponent ({opponent_player}) Turn {current_turn_number}.5 ---")
                try:
                    if hasattr(env, 'make_opponent_move'):
                        opp_action, opp_new_state, opp_terminal = env.make_opponent_move()
                        opp_turn_data = {
                            'turn': current_turn_number + 0.5,
                            'player': opponent_player,
                            'state_before_action': new_state_desc_this_turn,
                            'agent_raw_response': None,
                            'action_parsed': opp_action or "N/A",
                            'action_taken_for_log': opp_action or "(Opponent Failed Move)",
                            'action_valid': opp_action is not None,
                            'agent_signaled_completion': False,
                            'env_feedback_or_result': "N/A",
                            'state_after_action': opp_new_state,
                            'intermediate_status': None,
                            'regression_detected_this_turn': False,
                            'is_terminal_after_turn': opp_terminal,
                            'error_this_turn': None if opp_action else "Opponent failed to make a move"
                        }
                        history.append(opp_turn_data)
                        new_state_desc_this_turn = opp_new_state
                        env_feedback_or_result = opp_new_state
                        if opp_terminal:
                            logger.info("Game ended after opponent's move.")
                            is_terminal_this_turn = True
                            break
                        
                        if opp_action is None:
                            logger.error("Opponent failed to make a move (logic error?). Failing attempt.")
                            premature_failure = True
                            failure_reason = "Opponent move error (returned None)"
                            is_terminal_this_turn = True
                            break
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
                    break

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- Attempt {attempt_id + 1} Finished (Duration: {duration:.2f}s, Rounds/Turns Ran: {turn_count}) ---")

    final_score = 1
    final_eval_response = "Evaluation not performed."
    final_eval_input = None
    final_premature_failure = premature_failure

    if final_premature_failure:
        final_score = 1
        final_eval_response = f"Fail (Premature Failure: {failure_reason or 'Unknown'})"
        logger.info(f"Final Score: 1 ({final_eval_response})")
        final_eval_input_summary = failure_reason
    else:
        logger.info("Performing final evaluation...")
        try:
            final_eval_input = env.get_final_eval_input()
            final_eval_input_summary = str(final_eval_input)[:200] + ('...' if len(str(final_eval_input)) > 200 else '')

            if is_ttt:
                agent_won    = getattr(env, "winner", None) == env.agent_player_mark
                opponent_won = getattr(env, "winner", None) == getattr(env, "opponent_player_mark", "O")
                is_draw      = getattr(env, "is_draw", False)
                made_invalid = agent_made_any_invalid_move_in_attempt

                if opponent_won:
                    final_score = 2 if not made_invalid else 1
                    final_eval_response = ("Partial (Player X lost but made no invalid moves)" if final_score == 2 else f"Fail (Opponent '{getattr(env, 'opponent_player_mark', 'O')}' won and X made invalid moves)")
                elif agent_won or is_draw:
                    final_score = 3 if not made_invalid else 2
                    outcome_str = "won" if agent_won else "tied"
                    final_eval_response = (f"{'Success' if final_score == 3 else 'Partial'} (Player X {outcome_str}, {'no' if final_score == 3 else 'some'} invalid moves)")
                else:
                    final_score = 1
                    final_eval_response = "Fail (Game ended inconclusively)"
                logger.info(f"Deterministic TTT Score: {final_score} ({final_eval_response})")

            elif is_fs:
                if hasattr(env, 'evaluate_final_state'):
                    final_score = env.evaluate_final_state()
                    score_map = {3: "Success", 2: "Partial", 1: "Fail"}
                    final_eval_response = f"{score_map.get(final_score, 'Unknown')} (FS Deterministic Eval)"
                    logger.info(f"Deterministic FS Score: {final_score} ({final_eval_response})")
                else:
                    logger.error("FileSystemEnv missing 'evaluate_final_state' method for deterministic eval.")
                    final_score = 1
                    final_eval_response = "Fail (Environment evaluation method missing)"

            elif is_solar:
                template = prompts.get('finaleval_template')
                if not template:
                    logger.error("Final evaluation prompt template (finaleval.txt) missing for solar_gen.")
                    final_score = 1
                    final_eval_response = "Fail (Missing final evaluation prompt)"
                else:
                    image_path = final_eval_input
                    if not isinstance(image_path, str):
                        logger.error(f"Solar final evaluation input is not a string path: {type(image_path)}")
                        final_score = 1
                        final_eval_response = "Fail (Invalid image path type for final eval)"
                    else:
                        evaluator.eval_prompt_template = template
                        if hasattr(evaluator, 'evaluate_final_image_outcome'):
                            final_score, raw_resp = evaluator.evaluate_final_image_outcome(image_path, max_tokens=max_tokens)
                            final_eval_response = f"Multimodal Eval Raw Response: {raw_resp}"
                            logger.info(f"Multimodal Final Score: {final_score} (Raw response logged)")
                        else:
                            logger.error("Evaluator instance is missing the 'evaluate_final_image_outcome' method.")
                            final_score = 1
                            final_eval_response = "Fail (Evaluator configuration error - missing image eval method)"
            else:
                template = prompts.get('finaleval_template')
                if not template:
                    logger.error(f"Final evaluation prompt template (finaleval.txt) missing for task {task_name}.")
                    final_score = 1
                    final_eval_response = "Fail (Missing final evaluation prompt)"
                else:
                    evaluator.eval_prompt_template = template
                    if hasattr(evaluator, 'evaluate_final_outcome'):
                        final_score, raw_resp = evaluator.evaluate_final_outcome(str(final_eval_input), max_tokens=max_tokens)
                        final_eval_response = f"LLM Text Eval Raw Response: {raw_resp}"
                        logger.info(f"LLM Text Final Score: {final_score} (Raw response logged)")
                    else:
                        logger.error("Evaluator instance is missing the 'evaluate_final_outcome' method.")
                        final_score = 1
                        final_eval_response = "Fail (Evaluator configuration error - missing text eval method)"

        except Exception as e:
            logger.error(f"Error during final evaluation phase: {e}", exc_info=True)
            final_score = 1
            final_eval_response = f"Fail (Final evaluation phase error: {e})"
            final_eval_input_summary = final_eval_input_summary if final_eval_input_summary != "N/A" else f"Error during eval: {e}"

    final_failed_flag = final_premature_failure or (final_score == 1)
    final_failure_reason_str = ""
    if final_failed_flag:
        final_failure_reason_str = failure_reason if final_premature_failure else "Final evaluation resulted in Fail score (1)"

    is_successful_run = (final_score == 3 and not final_premature_failure)

    result = {
        'attempt_id': attempt_id + 1,
        'task_name': task_name,
        'success': is_successful_run,
        'score': final_score,
        'failed': final_failed_flag,
        'premature_failure': final_premature_failure,
        'failure_reason': final_failure_reason_str,
        'final_outcome_description': final_eval_response,
        'regressions_detected': regressions_detected_in_attempt,
        'agent_made_invalid_moves': agent_made_any_invalid_move_in_attempt if is_ttt else None,
        'rounds_completed': turn_count,
        'duration_seconds': duration,
        'final_eval_input_summary': str(final_eval_input_summary),
        'history': history,
    }
    return result

# --- Helper to Infer Provider from Model Name ---
def infer_provider_from_model(model_name: str) -> str | None:
    """Infers the provider from model name prefix."""
    if not model_name: return None
    name_lower = model_name.lower()
    # Special-case OpenRouter first: gpt-oss models are routed via OpenRouter
    if name_lower.startswith('openai/gpt-oss-') or '/gpt-oss-' in name_lower:
        return "openrouter"
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
    elif name_lower.startswith('moonshot-'):
        return "kimi"
    logger.debug(f"Could not infer provider from model name: {model_name}")
    return None

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="ToyBench: LLM Agentic Benchmark Suite"
    )
    # --- Standard Arguments ---
    parser.add_argument("-t", "--task", required=True,
                        help="Task name (e.g., file_system, tic_tac_toe, solar_gen)")
    parser.add_argument("-p", "--provider", default="gemini",
                         choices=["gemini", "openai", "grok", "quality_compute", "anthropic", "kimi", "openrouter"],
                        help="LLM Provider for the AGENT")
    parser.add_argument("-m", "--model", default=None,
                        help="Agent LLM model name. For QualityCompute, this is the base model for Best-of-N, or ignored if collaborative.")
    parser.add_argument("-n", "--attempts", type=int, default=1,
                        help="Number of independent attempts to run")
    parser.add_argument("-r", "--rounds", type=int, default=35,
                        help="Max rounds/steps per attempt")
    parser.add_argument("--evaluator_model", default=None,
                        help="Evaluator LLM model name. Defaults to config default (gemini-1.5-flash). Provider inferred.")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens for LLM responses (passed to agent and evaluator if supported).")
    parser.add_argument("--output_dir", default="results",
                        help="Base directory for saving results")
    # --- Provider-Specific Argument Groups ---
    openai_group = parser.add_argument_group('OpenAI Options')
    openai_group.add_argument("--openai_reasoning_effort", default="high", choices=["auto", "low", "medium", "high"],
                              help="Reasoning effort for compatible OpenAI models (e.g., gpt-5). Default is 'high'.")
    openai_group.add_argument("--openai_background", action='store_true', help="Run OpenAI responses in background mode (async) and poll until completion.")
    openai_group.add_argument("--openai_bg_poll", type=float, default=2.0, help="Polling interval (seconds) for background responses. Default 2.0s.")
    
    anthropic_group = parser.add_argument_group('Anthropic Options')
    anthropic_group.add_argument("--thinking", action='store_true', help="Enable extended thinking for Anthropic models")
    anthropic_group.add_argument("--thinking_budget", type=int, default=None, help="Token budget for extended thinking.")

    grok_group = parser.add_argument_group('Grok (xAI) Options')
    grok_group.add_argument("--reasoning_effort", default="low", choices=["low", "high"], help="Reasoning effort for Grok models.")
    
    # --- Quality Compute Specific Options ---
    qc_group = parser.add_argument_group('Quality Compute - General Options')
    qc_group.add_argument("--judge_model", default=None,
                          help="Model for judging. Used in QC 'selection' mode and required for 'collaborative' mode. If not provided for 'selection', uses default.")

    # Quality Compute - Best-of-N / Selection Mode
    qc_standard_mode_group = parser.add_argument_group('Quality Compute - Best-of-N / Selection Mode')
    qc_standard_mode_group.add_argument("--ensemble_mode", default="synthesis", choices=["synthesis", "selection"],
                          help="[QC Only] Strategy for Best-of-N: 'synthesis' (default) or 'selection' (judge model). Requires -m to have a -B<N> suffix.")

    # Quality Compute - Collaborative Agent Mode
    qc_collab_mode_group = parser.add_argument_group('Quality Compute - Collaborative Agent Mode')
    qc_collab_mode_group.add_argument("--use_collaborative_agent", action='store_true',
                              help="[QC Only] Use the collaborative agent inference style. Overrides Best-of-N settings.")
    qc_collab_mode_group.add_argument("--team_leader_model", type=str, default=None,
                              help="[Collaborative] The model to use for the Team Leader.")
    qc_collab_mode_group.add_argument("--student_model", type=str, default=None,
                              help="[Collaborative] The model to use for the Students.")
    qc_collab_mode_group.add_argument("--num_students", type=int, default=6,
                              help="[Collaborative] Number of student agents to run in parallel.")
    qc_collab_mode_group.add_argument("--num_turns", type=int, default=2,
                              help="[Collaborative] Number of internal refinement turns for the team.")

    # --- OpenRouter Specific Options (updated with reasoning_effort and provider)
    openrouter_group = parser.add_argument_group('OpenRouter Options')
    openrouter_group.add_argument("--or_no_fallbacks", action='store_false', dest='or_allow_fallbacks', default=True,
                                  help="Disable fallbacks for OpenRouter models. By default, fallbacks are allowed.")
    openrouter_group.add_argument("--or_sort", default="price", choices=["price", "speed"],
                                  help="Sorting method for OpenRouter provider (e.g., 'price' or 'speed'). Default is 'price'.")
    openrouter_group.add_argument("--or_reasoning_effort", default="low", choices=["low", "medium", "high"],
                                  help="Reasoning effort for OpenRouter models (low, medium, high). Default is 'low'.")
    openrouter_group.add_argument("--or_provider", default="fireworks",
                                  help="Force a specific upstream provider for OpenRouter (e.g., 'fireworks'). "
                                       "When set, fallbacks are disabled automatically.")

    args = parser.parse_args()

    try:
        config = load_config()
    except Exception as e:
        print(f"FATAL: Error loading configuration (.env or defaults): {e}")
        exit(1)

    # --- Determine Agent Configuration (Provider, Model, Keys) ---
    agent_provider = args.provider.lower()
    agent_model = args.model
    if not agent_model and not args.use_collaborative_agent:
        default_key = f"default_{agent_provider}_model"
        agent_model = config.get(default_key)
        if not agent_model:
            print(f"Error: Agent model not specified via --model and no default found for provider '{agent_provider}'.")
            exit(1)
        else:
            logger.info(f"Info: Agent model not specified. Using default for '{agent_provider}': '{agent_model}'")
    
    # Determine Evaluator Configuration
    evaluator_model_name = args.evaluator_model or config.get('evaluator_model', 'gemini-1.5-flash')
    evaluator_provider = infer_provider_from_model(evaluator_model_name)
    if evaluator_provider is None:
        logger.warning(f"Could not infer provider for evaluator model '{evaluator_model_name}'. Defaulting to agent's provider '{agent_provider}'.")
        evaluator_provider = agent_provider

    # --- Setup Output and Logging ---
    output_model_name = "collaborative_agent" if args.use_collaborative_agent else agent_model if agent_model else "unknown_model"
    base_output_dir = create_output_dir(args.output_dir, args.task, agent_provider, output_model_name)
    log_file = os.path.join(base_output_dir, "run.log")
    log_level_attr = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level=log_level_attr, log_file=log_file)

    logger.info("--- ToyBench Run Initializing ---")
    logger.info(f"Task: {args.task}, Provider: {agent_provider}, Agent Model: {agent_model or 'N/A (Collaborative)'}")
    logger.info(f"Evaluator: {evaluator_provider}, Evaluator Model: {evaluator_model_name}")
    logger.info(f"Attempts: {args.attempts}, Max Rounds/Steps: {args.rounds}")
    logger.info(f"Log Level: {args.log_level}, Base Output Directory: {base_output_dir}")
    logger.info(f"Max Tokens (if set): {args.max_tokens}")
    
    if agent_provider == 'quality_compute':
        if args.use_collaborative_agent:
            logger.info("Quality Compute Mode: Collaborative Agent.")
            logger.info(f"  Team Leader Model: {args.team_leader_model}, Student Model: {args.student_model}, Judge Model: {args.judge_model}")
            logger.info(f"  Num Students: {args.num_students}, Num Turns: {args.num_turns}")
        else:
            logger.info(f"Quality Compute Mode: Best-of-N / Selection. Ensemble Mode: {args.ensemble_mode}")
            if args.ensemble_mode == 'selection':
                logger.info(f"  Judge Model: {args.judge_model}")
    else:
        if agent_provider == 'anthropic':
            logger.info(f"Anthropic Thinking Enabled: {args.thinking}, Thinking Budget: {args.thinking_budget}")
        elif agent_provider == 'grok':
            logger.info(f"Grok Reasoning Effort: {args.reasoning_effort}")
        elif agent_provider == 'openai':
            logger.info(f"OpenAI Reasoning Effort: {args.openai_reasoning_effort}")
            logger.info(f"OpenAI Background Mode: {args.openai_background}, Background Poll Interval: {args.openai_bg_poll}s")
        elif agent_provider == 'openrouter':
            logger.info(f"OpenRouter Options: allow_fallbacks={args.or_allow_fallbacks}, sort={args.or_sort}, "
                        f"reasoning_effort={args.or_reasoning_effort}, provider={args.or_provider}")

    # --- API Key Loading ---
    api_key_map = {'gemini': 'gemini_api_key', 'openai': 'openai_api_key', 'grok': 'xai_api_key', 'quality_compute': 'quality_compute_api_key', 'anthropic': 'anthropic_api_key', 'kimi': 'kimi_api_key', 'openrouter': 'openrouter_api_key'}
    agent_api_key = config.get(api_key_map.get(agent_provider))
    evaluator_api_key = config.get(api_key_map.get(evaluator_provider))
    
    if not agent_api_key:
        logger.error(f"Agent API Key for provider '{agent_provider}' is required but not found in config/env.")
        print(f"Error: Agent API Key for '{agent_provider}' not found. Set the appropriate environment variable (e.g., GOOGLE_API_KEY, OPENAI_API_KEY, XAI_API_KEY, QUALITY_COMPUTE_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY, OPENROUTER_API_KEY).")
        exit(1)
    else:
        logger.info(f"Agent API Key loaded successfully for provider '{agent_provider}'.")

    if not evaluator_api_key:
        logger.error(f"Evaluator API Key for provider '{evaluator_provider}' is required but not found in config/env.")
        print(f"Error: Evaluator API Key for '{evaluator_provider}' not found. Ensure the necessary key is set.")
        exit(1)
    else:
        if evaluator_provider != agent_provider:
            logger.info(f"Evaluator API Key loaded successfully for provider '{evaluator_provider}'.")


    # --- Build the kwargs dictionary for the Agent LLM Interface ---
    agent_interface_kwargs = {
        'max_tokens': args.max_tokens,
        'thinking_enabled': args.thinking,
        'thinking_budget': args.thinking_budget,
        'reasoning_effort': args.reasoning_effort,  # For Grok
        'openai_reasoning_effort': args.openai_reasoning_effort,  # For OpenAI
        'openai_background': args.openai_background,            # NEW
        'openai_bg_poll': args.openai_bg_poll,                  # NEW
        'or_allow_fallbacks': args.or_allow_fallbacks,  # For OpenRouter
        'or_sort': args.or_sort,  # For OpenRouter
        'or_reasoning_effort': args.or_reasoning_effort,  # For OpenRouter
        'or_provider': args.or_provider,  # For OpenRouter
    }

    if agent_provider == 'quality_compute':
        agent_interface_kwargs['use_collaborative_agent'] = args.use_collaborative_agent
        agent_interface_kwargs['ensemble_mode'] = args.ensemble_mode
        agent_interface_kwargs['judge_model'] = args.judge_model
        if args.use_collaborative_agent:
            agent_interface_kwargs['team_leader_model'] = args.team_leader_model
            agent_interface_kwargs['student_model'] = args.student_model
            agent_interface_kwargs['num_students'] = args.num_students
            agent_interface_kwargs['num_turns'] = args.num_turns

    # --- Initialize Components ---
    try:
        prompts = load_task_prompts(args.task, config.get('task_definitions_dir', 'tasks'))

        # Initialize the Agent LLM Interface with all its specific configuration
        agent_llm = get_llm_interface(
            provider_name=agent_provider,
            api_key=agent_api_key,
            model_name=agent_model,
            **agent_interface_kwargs
        )
        logger.info(f"Agent LLM interface ({type(agent_llm).__name__}) initialized for model {agent_llm.model_name}.")

        # Initialize the Final Evaluator LLM Interface (generally a standard LLM)
        final_evaluator_llm = get_llm_interface(
            provider_name=evaluator_provider,
            api_key=evaluator_api_key,
            model_name=evaluator_model_name,
            max_tokens=args.max_tokens
        )
        logger.info(f"Final Evaluator LLM interface ({type(final_evaluator_llm).__name__}) initialized for model {final_evaluator_llm.model_name}.")
        
        # Initialize the Final Evaluator class (uses the final_evaluator_llm for scoring)
        final_evaluator = Evaluator(final_evaluator_llm, prompts.get('finaleval_template', ''))
        logger.info("Final Evaluator class initialized.")

    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
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
        attempt_history = []
        
        # Create a unique subdirectory for each attempt's artifacts
        attempt_output_dir = os.path.join(base_output_dir, f"attempt_{i+1}")
        try:
            os.makedirs(attempt_output_dir, exist_ok=True)
            logger.info(f"Attempt {i+1} output directory: {attempt_output_dir}")

            # Get environment for the attempt
            environment = get_environment(
                task_name=args.task,
                goal=prompts.get('goal_description', 'No goal provided.'),
                prompts=prompts,
                evaluator_llm=final_evaluator_llm,
                output_dir=attempt_output_dir,
                max_rounds=args.rounds
            )
            logger.info(f"Environment instance ({type(environment).__name__}) created for attempt {i+1}.")

            # Run the single attempt - the agent_llm object itself holds the configuration
            attempt_result = run_attempt(
                attempt_id=i,
                env=environment,
                agent=agent_llm,
                evaluator=final_evaluator,
                max_rounds=args.rounds,
                prompts=prompts,
                task_name=args.task,
                output_dir=attempt_output_dir,
                max_tokens=args.max_tokens
            )
            attempt_history = attempt_result.get('history', [])
            all_results.append(attempt_result)

        except KeyboardInterrupt:
            logger.warning(f"--- Run interrupted by user during attempt {i+1} ---")
            print("\nRun interrupted by user.")
            all_results.append({
                'attempt_id': i + 1, 'task_name': args.task, 'score': 1, 'failed': True, 'premature_failure': True,
                'failure_reason': "User interrupt", 'success': False,
                'final_outcome_description': "Interrupted by user",
                'regressions_detected': False, 'rounds_completed': len(attempt_history),
                'agent_made_invalid_moves': None,
                'duration_seconds': time.time() - attempt_start_time,
                'history': attempt_history, 'final_eval_input_summary': 'N/A'
            })
            break

        except ImportError as e:
            logger.error(f"--- CRITICAL ERROR during attempt {i+1} setup: Missing dependency. {e} ---", exc_info=True)
            print(f"\nFATAL Error: Missing dependency required for task '{args.task}'. {e}")
            print("Please install the required libraries (e.g., 'pip install selenium webdriver-manager') and ensure WebDriver is set up.")
            all_results.append({
                'attempt_id': i + 1, 'task_name': args.task, 'score': 1, 'failed': True, 'premature_failure': True,
                'failure_reason': f"Missing dependency: {e}", 'success': False,
                'final_outcome_description': f"Missing dependency: {e}",
                'regressions_detected': False, 'rounds_completed': 0,
                'agent_made_invalid_moves': None,
                'duration_seconds': time.time() - attempt_start_time,
                'history': [], 'final_eval_input_summary': 'N/A'
            })
            break

        except Exception as e:
            logger.error(f"--- CRITICAL UNHANDLED ERROR during attempt {i+1}: {e} ---", exc_info=True)
            print(f"\nFATAL Error: An unexpected error occurred during attempt {i+1}: {e}")
            all_results.append({
                'attempt_id': i + 1, 'task_name': args.task, 'score': 1, 'failed': True, 'premature_failure': True,
                'failure_reason': f"Unhandled exception in run_attempt or setup: {e}", 'success': False,
                'final_outcome_description': f"Unhandled exception: {e}",
                'regressions_detected': False,
                'rounds_completed': len(attempt_history),
                'agent_made_invalid_moves': None,
                'duration_seconds': time.time() - attempt_start_time,
                'history': attempt_history,
                'final_eval_input_summary': 'N/A'
            })
            break

    logger.info("--- Benchmark Run Finished ---")

    # --- Reporting and Saving ---
    if not all_results:
        logger.warning("No attempts were completed or recorded.")
        print("No results to report.")
        exit(0)

    try:
        metrics = calculate_metrics(all_results, args.attempts, task_name=args.task)
        report = format_report(metrics, args.task, agent_provider, output_model_name, args.rounds)
        print("\n" + report + "\n")
        logger.info(f"Final Report:\n{report}")
    except Exception as e:
        logger.error(f"Failed to calculate or format metrics/report: {e}", exc_info=True)
        print(f"\nError generating final report: {e}\n")

    run_config_args = vars(args).copy()
    run_config_args['agent_provider_used'] = agent_provider
    run_config_args['agent_model_used'] = output_model_name
    run_config_args['evaluator_provider_used'] = evaluator_provider
    run_config_args['evaluator_model_used'] = evaluator_model_name
    run_config_args[f'{agent_provider}_api_key_loaded'] = (agent_api_key is not None)
    if evaluator_provider != agent_provider:
        run_config_args[f'{evaluator_provider}_api_key_loaded'] = (evaluator_api_key is not None)
    run_config_args['base_output_directory'] = base_output_dir
    run_config_args['task_definitions_dir_used'] = config.get('task_definitions_dir', 'tasks')

    try:
        save_results(base_output_dir, all_results, report if 'report' in locals() else "Report generation failed.", run_config_args)
    except Exception as e:
        logger.error(f"Failed to save results to {base_output_dir}: {e}", exc_info=True)
        print(f"Error saving results: {e}")

    logger.info(f"Results, logs, and artifacts saved in base directory: {base_output_dir}")
    logger.info("--- ToyBench Run Complete ---")


if __name__ == "__main__":
    main()