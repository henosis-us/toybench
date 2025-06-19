# reporting.py
import logging
import os
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# Define type alias for token usage dictionary for clarity within this module
TokenUsage = dict[str, int | None] | None

def calculate_metrics(attempt_results: list[dict], num_attempts: int, task_name: str, k_values: list[int] = [1, 5, 10, 20]) -> dict:
    """Calculates metrics based on a list of attempt results.

    Focuses on accurate distinction between Score 1 failures and Error failures using the 'premature_failure' flag.
    Also calculates aggregate token usage, including reasoning tokens for supported models.
    Added task_name to support task-specific metrics, such as the Differentiated Score for solar_gen.
    MODIFIED: Now calculates reasoning tokens from the discrepancy if not explicitly provided, ensuring totals add up.
    ADDED: Debug logging to show raw token_usage before reconciliation.
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
    total_reasoning_tokens = 0  # For reasoning/thinking token aggregation
    total_tokens_overall = 0
    attempts_with_token_data = 0  # Count attempts that had at least one turn with token data

    logger.debug("--- Categorizing Attempt Results and Aggregating Tokens ---")
    for i, r in enumerate(attempt_results):
        score = r.get('score')
        is_premature = r.get('premature_failure', False)  # Default to False if missing
        attempt_id = r.get('attempt_id', i + 1)
        has_regression = r.get('regressions_detected', False)
        log_msg = f"Attempt {attempt_id}: Score={score}, Premature={is_premature}"

        # --- Outcome Categorization ---
        if is_premature:
            num_failed_error += 1
            log_msg += " -> Categorized as: Error (Premature)"
        elif score == 1:
            num_failed_score += 1
            log_msg += " -> Categorized as: Failed (Score 1)"
        elif score == 2:
            num_partial += 1
            log_msg += " -> Categorized as: Partial"
        elif score == 3:
            num_successful += 1
            log_msg += " -> Categorized as: Success"
        else:
            logger.warning(f"Attempt {attempt_id}: Unexpected score ({score}) and not premature. Categorizing as Error.")
            num_failed_error += 1
            log_msg += " -> Categorized as: Error (Unexpected Score)"

        if has_regression:
            num_regressions += 1

        # --- Token Aggregation per Attempt ---
        attempt_had_tokens = False
        history = r.get('history', [])
        if history:
            for turn_data in history:
                token_usage: TokenUsage = turn_data.get('token_usage')
                if isinstance(token_usage, dict):
                    input_t = token_usage.get('input_tokens') or 0
                    output_t = token_usage.get('output_tokens') or 0
                    reasoning_t = token_usage.get('reasoning_tokens') or 0
                    total_t = token_usage.get('total_tokens') or 0

                    # NEW LOG LINE HERE: Log the raw token_usage before any reconciliation
                    logger.debug(f"Turn {turn_data.get('turn', 'N/A')} raw token_usage from LLM interface: {token_usage}")

                    # If total_tokens is missing or 0, sum input and output as fallback
                    if not total_t and (input_t > 0 or output_t > 0 or reasoning_t > 0): # Include reasoning_t in fallback check
                        total_t = input_t + output_t + reasoning_t # Ensure reasoning is included in fallback calculation of total
                        logger.debug(f"Calculated total_tokens as sum of input, output, and reasoning for turn {turn_data.get('turn', 'N/A')}: {total_t}")

                    # --- FIX: Calculate reasoning tokens from discrepancy if not explicitly provided or incorrect ---
                    # This handles cases where the API includes thinking tokens in the total
                    # but doesn't provide a separate `thoughts_token_count` field, or the provided one is incorrect.
                    # Only reconcile if there's a positive discrepancy and reasoning_t isn't already the source.
                    if total_t > (input_t + output_t + (reasoning_t if reasoning_t is not None else 0)): # Add reasoning_t to the sum for comparison
                        calculated_discrepancy = total_t - (input_t + output_t + (reasoning_t if reasoning_t is not None else 0))
                        # Only update if the calculated discrepancy is positive
                        if calculated_discrepancy > 0:
                            reasoning_t += calculated_discrepancy # Add discrepancy to existing reasoning_t
                            logger.debug(
                                f"Turn {turn_data.get('turn', 'N/A')} had a discrepancy in total tokens. "
                                f"Attributing the {calculated_discrepancy} token difference to reasoning. "
                                f"New Reasoning_t: {reasoning_t} (Total: {total_t}, Input: {input_t}, Output: {output_t})"
                            )

                    # Add to overall totals
                    total_input_tokens += input_t
                    total_output_tokens += output_t
                    total_reasoning_tokens += reasoning_t
                    total_tokens_overall += total_t
                    attempt_had_tokens = True

        if attempt_had_tokens:
            attempts_with_token_data += 1
        logger.debug(log_msg)

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
        'pass@1': 0,
        'pass@20': 0.0,
        'at_least_one_success': 1.0 if num_successful > 0 else 0.0,
        'solar_differentiated_score': None,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_reasoning_tokens': total_reasoning_tokens,
        'total_tokens_all_attempts': total_tokens_overall,
        'attempts_with_token_data': attempts_with_token_data,
        'avg_input_tokens_per_attempt': total_input_tokens / attempts_with_token_data if attempts_with_token_data > 0 else 0,
        'avg_output_tokens_per_attempt': total_output_tokens / attempts_with_token_data if attempts_with_token_data > 0 else 0,
        'avg_reasoning_tokens_per_attempt': total_reasoning_tokens / attempts_with_token_data if attempts_with_token_data > 0 else 0,
        'avg_total_tokens_per_attempt': total_tokens_overall / attempts_with_token_data if attempts_with_token_data > 0 else 0,
    }

    if task_name == "solar_gen" and num_attempts_run > 0:
        differentiated_score = ((metrics['num_successful'] * 1.0) + (metrics['num_partial'] * 0.3)) / num_attempts_run * 100
        metrics['solar_differentiated_score'] = differentiated_score
        logger.info(f"Differentiated Score for SolarGen calculated: {differentiated_score:.2f}%")

    metrics['pass@1'] = metrics['pass_rate_strict']
    k_pass_check = min(num_attempts_run, 20)
    metrics['pass@20'] = 1.0 if any(r.get('score') == 3 and not r.get('premature_failure', False) for r in attempt_results[:k_pass_check]) else 0.0

    logger.info(f"Metrics calculated: Success={metrics['num_successful']}, Partial={metrics['num_partial']}, Failed (Score 1)={metrics['num_failed_score']}, Failed (Error)={metrics['num_failed_error']}, Solar Differentiated Score={metrics.get('solar_differentiated_score', 'N/A')}")
    logger.info(f"Token Metrics: TotalInput={metrics['total_input_tokens']}, TotalOutput={metrics['total_output_tokens']}, TotalReasoning={metrics['total_reasoning_tokens']}, TotalOverall={metrics['total_tokens_all_attempts']} (from {metrics['attempts_with_token_data']} attempts)")
    logger.debug(f"Final metrics dictionary: {metrics}")
    return metrics

def format_report(metrics_dict: dict, task_name: str, provider: str, model: str, turn_horizon: int) -> str:
    """Formats the calculated metrics into a human-readable string."""
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

    total_input = metrics_dict.get('total_input_tokens', 0)
    total_output = metrics_dict.get('total_output_tokens', 0)
    total_reasoning = metrics_dict.get('total_reasoning_tokens', 0)
    total_overall = metrics_dict.get('total_tokens_all_attempts', 0)
    avg_input = metrics_dict.get('avg_input_tokens_per_attempt', 0)
    avg_output = metrics_dict.get('avg_output_tokens_per_attempt', 0)
    avg_reasoning = metrics_dict.get('avg_reasoning_tokens_per_attempt', 0)
    avg_total = metrics_dict.get('avg_total_tokens_per_attempt', 0)
    token_attempts = metrics_dict.get('attempts_with_token_data', 0)

    logger.debug(f"Formatting report with metrics: Success={num_success}, Partial={num_partial}, FailedScore={num_failed_score}, FailedError={num_failed_error}")

    report = [
        f"--- ToyBench Report ---",
        f"Task          : {task_name}",
        f"Provider      : {provider}",
        f"Model         : {model}",
        f"Turn Horizon  : {turn_horizon}",
        f"Attempts Run  : {attempts_completed} / {attempts_requested}",
        f"-------------------------",
        f"Success (Score 3) : {num_success}",
        f"Partial (Score 2) : {num_partial}",
        f"Failed (Score 1)  : {num_failed_score}",
        f"Failed (Error)    : {num_failed_error}",
        f"Total Failed      : {num_total_failed}",
        f"-------------------------",
        f"Pass Rate (Pass@1): {pass_1_rate:.2%}",
        f"Pass@20 (Any success in first 20): {pass_20_rate:.0%}",
        f"Success Rate (Any in Run): {any_success_rate:.0%}",
        f"Regression Freq.  : {regression_freq:.2%} ({regression_count} attempts)",
    ]

    if task_name == "solar_gen":
        differentiated_score = metrics_dict.get('solar_differentiated_score', 0)
        report.append(f"Differentiated Solar Score (S3=1pt, S2=0.3pt): {differentiated_score:.2f}%")

    if token_attempts > 0:
        report.extend([
            f"--- Token Usage (Based on {token_attempts} attempts) ---",
            f"Total Input     : {total_input:,}",
            f"Total Output    : {total_output:,}",
            f"Total Reasoning : {total_reasoning:,}",
            f"Total Overall   : {total_overall:,}",
            f"Avg Input/Attempt : {avg_input:,.1f}",
            f"Avg Output/Attempt: {avg_output:,.1f}",
            f"Avg Reasoning/Attempt : {avg_reasoning:,.1f}",
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
        os.makedirs(output_dir)

    summary_path = os.path.join(output_dir, "summary_report.txt")
    try:
        with open(summary_path, "w", encoding='utf-8') as f:
            f.write(report)
    except Exception as e:
        logger.error(f"Failed to write summary report to {summary_path}: {e}")

    config_path = os.path.join(output_dir, "run_config.json")
    try:
        serializable_config = {}
        for k, v in config_args.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
        with open(config_path, "w", encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to write config file to {config_path}: {e}")

    results_path = os.path.join(output_dir, "attempt_results.jsonl")
    try:
        with open(results_path, "w", encoding='utf-8') as f:
            for result in results:
                try:
                    f.write(json.dumps(result) + "\n")
                except TypeError as json_e:
                    logger.error(f"Failed to serialize result for attempt {result.get('attempt_id', 'N/A')} to JSON: {json_e}. Skipping result.")
                    f.write(json.dumps({"error": "Serialization failed", "attempt_id": result.get('attempt_id', 'N/A')}) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write result for attempt {result.get('attempt_id', 'N/A')} to JSONL: {write_e}. Skipping.")
                    f.write(json.dumps({"error": "Write failed", "attempt_id": result.get('attempt_id', 'N/A')}) + "\n")
    except Exception as e:
        logger.error(f"Failed to open or write detailed results file {results_path}: {e}")

    logger.info(f"Results saved to: {output_dir} (Summary: {summary_path}, Config: {config_path}, Details: {results_path}")