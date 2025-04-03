import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

def calculate_metrics(attempt_results: list[dict], num_attempts: int, k_values: list[int] = [1, 5, 10, 20]) -> dict:
    """
    Calculates metrics based on a list of attempt results.
    Each result dict should contain: 'score' (1, 2, or 3), 'regressions_detected' (bool), 'failed' (bool).
    """
    if not attempt_results:
        logger.warning("No attempt results to calculate metrics from.")
        return {}

    successful_attempts = [r for r in attempt_results if not r.get('failed', False) and r.get('score') == 3]
    partial_attempts = [r for r in attempt_results if not r.get('failed', False) and r.get('score') == 2]
    failed_attempts_score = [r for r in attempt_results if not r.get('failed', False) and r.get('score') == 1]
    failed_attempts_error = [r for r in attempt_results if r.get('failed', True)] # Includes errors/invalid actions

    total_evaluated = len(successful_attempts) + len(partial_attempts) + len(failed_attempts_score)
    total_failures = len(failed_attempts_score) + len(failed_attempts_error)
    num_attempts_run = len(attempt_results) # Could be less than requested if errors occurred early

    metrics = {
        'num_attempts_requested': num_attempts,
        'num_attempts_completed': num_attempts_run,
        'num_successful': len(successful_attempts),
        'num_partial': len(partial_attempts),
        'num_failed_score': len(failed_attempts_score),
        'num_failed_error': len(failed_attempts_error),
        'num_total_failed': total_failures,
        'pass_rate_strict': len(successful_attempts) / num_attempts_run if num_attempts_run > 0 else 0,
        'pass_rate_partial': (len(successful_attempts) + len(partial_attempts)) / num_attempts_run if num_attempts_run > 0 else 0,
        'regression_count': sum(1 for r in attempt_results if r.get('regressions_detected')),
    }
    metrics['regression_frequency'] = metrics['regression_count'] / num_attempts_run if num_attempts_run > 0 else 0

    # Calculate pass@k (assuming k=num_attempts here as per PRD FR5.2)
    # The PRD seems to conflate pass@k (success within k attempts) with success rate over N attempts.
    # Let's report pass_rate_strict as pass@1 (average success rate) and calculate pass@N based on the total run.
    metrics['pass@1'] = metrics['pass_rate_strict'] # Average success rate

    # This interprets pass@k as "what fraction of runs achieved success within k *attempts*".
    # For a single run of N attempts, pass@N is just checking if at least one succeeded.
    # Let's rename to be clearer: success_in_run (at least one success)
    metrics['at_least_one_success'] = 1.0 if len(successful_attempts) > 0 else 0.0

    # The PRD's pass@k might mean running the *entire benchmark* k times and checking consistency,
    # or more likely, it refers to the success rate over the N attempts run (which we called pass@1).
    # Let's stick to pass@1 = average success rate. FR5.2 might be misinterpreted or needs clarification.
    # We will report pass rate over the N attempts (`pass_rate_strict`).

    logger.info(f"Metrics calculated: Success={metrics['num_successful']}, Partial={metrics['num_partial']}, Failed={metrics['num_total_failed']}, Regressions={metrics['regression_count']}")
    return metrics


def format_report(metrics: dict, task_name: str, provider: str, model: str, turn_horizon: int) -> str:
    """Formats the calculated metrics into a human-readable string."""
    report = [
        f"--- ToyBench Report ---",
        f"Task          : {task_name}",
        f"Provider      : {provider}",
        f"Model         : {model}",
        f"Turn Horizon  : {turn_horizon}",
        f"Attempts Run  : {metrics.get('num_attempts_completed', 0)} / {metrics.get('num_attempts_requested', 0)}",
        f"-------------------------",
        f"Success (Score 3) : {metrics.get('num_successful', 0)}",
        f"Partial (Score 2) : {metrics.get('num_partial', 0)}",
        f"Failed (Score 1)  : {metrics.get('num_failed_score', 0)}",
        f"Failed (Error)    : {metrics.get('num_failed_error', 0)}",
        f"Total Failed      : {metrics.get('num_total_failed', 0)}",
        f"-------------------------",
        f"Pass Rate (Pass@1): {metrics.get('pass@1', 0):.2%}",
        f"Success Rate (Any): {metrics.get('at_least_one_success', 0):.2%}", # Renamed from pass@k
        f"Regression Freq.  : {metrics.get('regression_frequency', 0):.2%} ({metrics.get('regression_count', 0)} attempts)",
        f"--- End Report ---"
    ]
    return "\n".join(report)

def save_results(output_dir: str, results: list[dict], report: str, config_args: dict):
    """Saves detailed attempt results and the summary report."""
    import json
    # Save summary report
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write(report)

    # Save configuration used
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
         # Convert args namespace to dict if necessary
        json.dump(config_args, f, indent=4)

    # Save detailed attempt results
    with open(os.path.join(output_dir, "attempt_results.jsonl"), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Results saved to: {output_dir}")
