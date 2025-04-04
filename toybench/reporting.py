# reporting.py
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

def calculate_metrics(attempt_results: list[dict], num_attempts: int, k_values: list[int] = [1, 5, 10, 20]) -> dict:
    """
    Calculates metrics based on a list of attempt results.
    Focuses on accurate distinction between Score 1 failures and Error failures using the 'premature_failure' flag.
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

    logger.debug("--- Categorizing Attempt Results ---")
    for i, r in enumerate(attempt_results):
        score = r.get('score')
        is_premature = r.get('premature_failure', False) # Default to False if missing
        attempt_id = r.get('attempt_id', i+1)
        has_regression = r.get('regressions_detected', False)

        log_msg = f"Attempt {attempt_id}: Score={score}, Premature={is_premature}"

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
            num_failed_error += 1 # Treat unexpected as error
            log_msg += " -> Categorized as: Error (Unexpected Score)"

        if has_regression:
            num_regressions += 1

        logger.debug(log_msg)

    total_failures = num_failed_score + num_failed_error

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
    }
    metrics['regression_frequency'] = metrics['regression_count'] / num_attempts_run if num_attempts_run > 0 else 0
    metrics['pass@1'] = metrics['pass_rate_strict']
    k_pass_check = min(num_attempts_run, 20)
    # Pass@k success requires score 3 AND not premature
    metrics['pass@20'] = 1.0 if any(r.get('score') == 3 and not r.get('premature_failure', False) for r in attempt_results[:k_pass_check]) else 0.0
    metrics['at_least_one_success'] = 1.0 if num_successful > 0 else 0.0

    # Log the final calculated metrics before returning
    logger.info(f"Metrics calculated: Success={metrics['num_successful']}, Partial={metrics['num_partial']}, Failed (Score 1)={metrics['num_failed_score']}, Failed (Error)={metrics['num_failed_error']}, Regressions={metrics['regression_count']}")
    logger.debug(f"Final metrics dictionary: {metrics}")
    return metrics


# MODIFIED: Explicitly access dictionary keys in format_report
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

    logger.debug(f"Formatting report with metrics: Success={num_success}, Partial={num_partial}, FailedScore={num_failed_score}, FailedError={num_failed_error}")

    report = [
        f"--- ToyBench Report ---",
        f"Task          : {task_name}",
        f"Provider      : {provider}",
        f"Model         : {model}",
        f"Turn Horizon  : {turn_horizon}", # Max Rounds/Steps setting
        f"Attempts Run  : {attempts_completed} / {attempts_requested}",
        f"-------------------------",
        f"Success (Score 3) : {num_success}",
        f"Partial (Score 2) : {num_partial}",
        f"Failed (Score 1)  : {num_failed_score}", # Uses explicit variable
        f"Failed (Error)    : {num_failed_error}", # Uses explicit variable
        f"Total Failed      : {num_total_failed}",
        f"-------------------------",
        f"Pass Rate (Pass@1): {pass_1_rate:.2%}",
        f"Pass@20 (Any success in first 20): {pass_20_rate:.0%}",
        f"Success Rate (Any in Run): {any_success_rate:.0%}",
        f"Regression Freq.  : {regression_freq:.2%} ({regression_count} attempts)",
        f"--- End Report ---"
    ]
    return "\n".join(report)

# (save_results remains unchanged)
def save_results(output_dir: str, results: list[dict], report: str, config_args: dict):
    """Saves detailed attempt results and the summary report."""
    import json
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # Ensure base dir exists

    # Save summary report
    summary_path = os.path.join(output_dir, "summary_report.txt")
    try:
        with open(summary_path, "w") as f: f.write(report)
    except Exception as e: logger.error(f"Failed to write summary report to {summary_path}: {e}")

    # Save configuration used
    config_path = os.path.join(output_dir, "run_config.json")
    try:
        with open(config_path, "w") as f: json.dump(config_args, f, indent=4)
    except Exception as e: logger.error(f"Failed to write config file to {config_path}: {e}")

    # Save detailed attempt results
    results_path = os.path.join(output_dir, "attempt_results.jsonl")
    try:
        with open(results_path, "w") as f:
            for result in results: f.write(json.dumps(result) + "\n")
    except Exception as e: logger.error(f"Failed to write detailed results to {results_path}: {e}")

    logger.info(f"Results saved to: {output_dir}")
