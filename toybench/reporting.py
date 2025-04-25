# reporting.py
import logging
import os
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# Define type alias for token usage dictionary for clarity within this module
TokenUsage = dict[str, int | None] | None

def calculate_metrics(attempt_results: list[dict], num_attempts: int, k_values: list[int] = [1, 5, 10, 20]) -> dict:
    """Calculates metrics based on a list of attempt results.
    Focuses on accurate distinction between Score 1 failures and Error failures using the 'premature_failure' flag.
    Also calculates aggregate token usage, including reasoning tokens for supported models.
    """
    if not attempt_results:
        logger.warning("No attempt results to calculate metrics from.")
        return {}

    num_attempts_run = len(attempt_results)
    num_successful = 0
    num_partial = 0
    num_failed_score = 0
    num_failed_error = 0
    num_regressions = 0

    # --- Token Usage Aggregation ---
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0  # Added for reasoning token aggregation
    total_tokens_overall = 0
    attempts_with_token_data = 0  # Count attempts that had at least one turn with token data

    logger.debug("--- Categorizing Attempt Results and Aggregating Tokens ---")
    for i, r in enumerate(attempt_results):
        score = r.get('score')
        is_premature = r.get('premature_failure', False)  # Default to False if missing
        attempt_id = r.get('attempt_id', i+1)
        has_regression = r.get('regressions_detected', False)
        log_msg = f"Attempt {attempt_id}: Score={score}, Premature={is_premature}"

        # --- Outcome Categorization ---
        if is_premature:
            # Any premature failure is counted as an Error failure
            num_failed_error += 1
            log_msg += " -> Categorized as: Error (Premature)"
        elif score == 1:
            # Completed but scored 1 -> Score 1 failure
            num_failed_score += 1
            log_msg += " -> Categorized as: Failed (Score 1)"
        elif score == 2:
            num_partial += 1
            log_msg += " -> Categorized as: Partial"
        elif score == 3:
            num_successful += 1
            log_msg += " -> Categorized as: Success"
        else:
            # Handle unexpected scores or missing scores if not premature
            logger.warning(f"Attempt {attempt_id}: Unexpected score ({score}) and not premature. Categorizing as Error.")
            num_failed_error += 1  # Treat unexpected as error
            log_msg += " -> Categorized as: Error (Unexpected Score)"

        if has_regression:
            num_regressions += 1

        # --- Token Aggregation per Attempt ---
        attempt_had_tokens = False
        history = r.get('history', [])
        if history:  # Check if history exists
            for turn_data in history:
                # Look for 'token_usage' key, which should contain a dict or None
                token_usage: TokenUsage = turn_data.get('token_usage')  # Returns None if key missing
                if isinstance(token_usage, dict):
                    # Safely get values, defaulting to 0 if key missing or value is None
                    input_t = token_usage.get('input_tokens') or 0
                    output_t = token_usage.get('output_tokens') or 0
                    reasoning_t = token_usage.get('reasoning_tokens') or 0  # Added for reasoning tokens, default to 0
                    total_t = token_usage.get('total_tokens') or 0  # Use reported total if available

                    # If total_tokens is missing or 0, sum input and output as fallback
                    if not total_t and (input_t > 0 or output_t > 0):
                        total_t = input_t + output_t

                    # Add to overall totals
                    total_input_tokens += input_t
                    total_output_tokens += output_t
                    total_reasoning_tokens += reasoning_t  # Aggregate reasoning tokens
                    total_tokens_overall += total_t
                    attempt_had_tokens = True  # Mark that this attempt contributed token data

        if attempt_had_tokens:
            attempts_with_token_data += 1

        logger.debug(log_msg)  # Log categorization result

    total_failures = num_failed_score + num_failed_error

    # --- Calculate Final Metrics ---
    metrics = {
        'num_attempts_requested': num_attempts,
        'num_attempts_completed': num_attempts_run,
        'num_successful': num_successful,
        'num_partial': num_partial,
        'num_failed_score': num_failed_score,
        'num_failed_error': num_failed_error,
        'num_total_failed': total_failures,
        'pass_rate_strict': num_successful / num_attempts_run if num_attempts_run > 0 else 0,
        'pass_rate_partial': (num_successful + num_partial) / num_attempts_run if num_attempts_run > 0 else 0,
        'regression_count': num_regressions,
        'regression_frequency': num_regressions / num_attempts_run if num_attempts_run > 0 else 0,
        'pass@1': 0,  # Default value
        'pass@20': 0.0,  # Default value
        'at_least_one_success': 1.0 if num_successful > 0 else 0.0,

        # --- Token Metrics ---
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_reasoning_tokens': total_reasoning_tokens,  # Added for total reasoning tokens
        'total_tokens_all_attempts': total_tokens_overall,
        'attempts_with_token_data': attempts_with_token_data,
        # Calculate averages only if data is available to avoid division by zero
        'avg_input_tokens_per_attempt': total_input_tokens / attempts_with_token_data if attempts_with_token_data > 0 else 0,
        'avg_output_tokens_per_attempt': total_output_tokens / attempts_with_token_data if attempts_with_token_data > 0 else 0,
        'avg_reasoning_tokens_per_attempt': total_reasoning_tokens / attempts_with_token_data if attempts_with_token_data > 0 else 0,  # Added for average reasoning tokens
        'avg_total_tokens_per_attempt': total_tokens_overall / attempts_with_token_data if attempts_with_token_data > 0 else 0,
    }

    # Calculate Pass@K (handle edge case where k > num_attempts_run)
    metrics['pass@1'] = metrics['pass_rate_strict']  # Pass@1 is just the strict pass rate
    k_pass_check = min(num_attempts_run, 20)
    metrics['pass@20'] = 1.0 if any(r.get('score') == 3 and not r.get('premature_failure', False) for r in attempt_results[:k_pass_check]) else 0.0

    # Log the final calculated metrics before returning
    logger.info(f"Metrics calculated: Success={metrics['num_successful']}, Partial={metrics['num_partial']}, Failed (Score 1)={metrics['num_failed_score']}, Failed (Error)={metrics['num_failed_error']}, Regressions={metrics['regression_count']}")
    logger.info(f"Token Metrics: TotalInput={metrics['total_input_tokens']}, TotalOutput={metrics['total_output_tokens']}, TotalReasoning={metrics['total_reasoning_tokens']}, TotalOverall={metrics['total_tokens_all_attempts']} (from {metrics['attempts_with_token_data']} attempts)")  # Updated log to include reasoning tokens
    logger.debug(f"Final metrics dictionary: {metrics}")
    return metrics

def format_report(metrics_dict: dict, task_name: str, provider: str, model: str, turn_horizon: int) -> str:
    """Formats the calculated metrics into a human-readable string."""
    # Use direct key access with .get() for safety
    num_success = metrics_dict.get('num_successful', 0)
    num_partial = metrics_dict.get('num_partial', 0)
    num_failed_score = metrics_dict.get('num_failed_score', 0)
    num_failed_error = metrics_dict.get('num_failed_error', 0)
    num_total_failed = metrics_dict.get('num_total_failed', 0)
    pass_1_rate = metrics_dict.get('pass@1', 0)
    pass_20_rate = metrics_dict.get('pass@20', 0)
    any_success_rate = metrics_dict.get('at_least_one_success', 0)
    regression_freq = metrics_dict.get('regression_frequency', 0)
    regression_count = metrics_dict.get('regression_count', 0)
    attempts_completed = metrics_dict.get('num_attempts_completed', 0)
    attempts_requested = metrics_dict.get('num_attempts_requested', 0)

    # Token metrics
    total_input = metrics_dict.get('total_input_tokens', 0)
    total_output = metrics_dict.get('total_output_tokens', 0)
    total_reasoning = metrics_dict.get('total_reasoning_tokens', 0)  # Added for reasoning tokens
    total_overall = metrics_dict.get('total_tokens_all_attempts', 0)
    avg_input = metrics_dict.get('avg_input_tokens_per_attempt', 0)
    avg_output = metrics_dict.get('avg_output_tokens_per_attempt', 0)
    avg_reasoning = metrics_dict.get('avg_reasoning_tokens_per_attempt', 0)  # Added for average reasoning tokens
    avg_total = metrics_dict.get('avg_total_tokens_per_attempt', 0)
    token_attempts = metrics_dict.get('attempts_with_token_data', 0)

    logger.debug(f"Formatting report with metrics: Success={num_success}, Partial={num_partial}, FailedScore={num_failed_score}, FailedError={num_failed_error}")
    report = [
        f"--- ToyBench Report ---",
        f"Task          : {task_name}",
        f"Provider      : {provider}",
        f"Model         : {model}",
        f"Turn Horizon  : {turn_horizon}",  # Max Rounds/Steps setting
        f"Attempts Run  : {attempts_completed} / {attempts_requested}",
        f"-------------------------",
        f"Success (Score 3) : {num_success}",
        f"Partial (Score 2) : {num_partial}",
        f"Failed (Score 1)  : {num_failed_score}",
        f"Failed (Error)    : {num_failed_error}",
        f"Total Failed      : {num_total_failed}",
        f"-------------------------",
        f"Pass Rate (Pass@1): {pass_1_rate:.2%}",
        f"Pass@20 (Any success in first 20): {pass_20_rate:.0%}",  # Display as percentage rounded to whole number
        f"Success Rate (Any in Run): {any_success_rate:.0%}",  # Display as percentage rounded to whole number
        f"Regression Freq.  : {regression_freq:.2%} ({regression_count} attempts)",
    ]

    # Add token section only if data was found
    if token_attempts > 0:
        report.extend([
            f"--- Token Usage (Based on {token_attempts} attempts) ---",
            f"Total Input     : {total_input:,}",
            f"Total Output    : {total_output:,}",
            f"Total Reasoning : {total_reasoning:,}",  # Added line for reasoning tokens
            f"Total Overall   : {total_overall:,}",
            f"Avg Input/Attempt : {avg_input:,.1f}",
            f"Avg Output/Attempt: {avg_output:,.1f}",
            f"Avg Reasoning/Attempt : {avg_reasoning:,.1f}",  # Added line for average reasoning tokens
            f"Avg Total/Attempt : {avg_total:,.1f}",
        ])
    else:
        report.append("--- Token Usage: No token data recorded ---")

    report.append("--- End Report ---")
    return "\n".join(report)

def save_results(output_dir: str, results: list[dict], report: str, config_args: dict):
    """
    Saves detailed attempt results (including per-turn token usage if present in history)
    and the summary report.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Ensure base dir exists

    # Save summary report
    summary_path = os.path.join(output_dir, "summary_report.txt")
    try:
        with open(summary_path, "w", encoding='utf-8') as f:  # Ensure utf-8 encoding
            f.write(report)
    except Exception as e:
        logger.error(f"Failed to write summary report to {summary_path}: {e}")

    # Save configuration used
    config_path = os.path.join(output_dir, "run_config.json")
    try:
        # Convert Path objects or other non-serializable types if necessary
        serializable_config = {}
        for k, v in config_args.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)  # Convert unknown types to string

        with open(config_path, "w", encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to write config file to {config_path}: {e}")

    # Save detailed attempt results to JSONL
    # This now implicitly saves the token_usage if it was added to the turn history
    # within the 'results' list by the benchmark runner (toybench_cli.py).
    results_path = os.path.join(output_dir, "attempt_results.jsonl")
    try:
        with open(results_path, "w", encoding='utf-8') as f:
            for result in results:
                try:
                    # Attempt to serialize each result individually
                    f.write(json.dumps(result) + "\n")
                except TypeError as json_e:
                    logger.error(f"Failed to serialize result for attempt {result.get('attempt_id', 'N/A')} to JSON: {json_e}. Skipping result.")
                    # Optionally write a placeholder or error message to the file
                    f.write(json.dumps({"error": "Serialization failed", "attempt_id": result.get('attempt_id', 'N/A')}) + "\n")
                except Exception as write_e:
                    # Catch other potential errors during write
                    logger.error(f"Failed to write result for attempt {result.get('attempt_id', 'N/A')} to JSONL: {write_e}. Skipping.")
                    f.write(json.dumps({"error": "Write failed", "attempt_id": result.get('attempt_id', 'N/A')}) + "\n")

    except Exception as e:
        logger.error(f"Failed to open or write detailed results file {results_path}: {e}")

    logger.info(f"Results saved to: {output_dir} (Summary: {summary_path}, Config: {config_path}, Details: {results_path})")