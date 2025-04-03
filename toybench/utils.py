import logging
import datetime
import os
import re

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configures logging."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

def create_output_dir(base_dir="results", task_name="unknown", provider="gemini", model="default"):
    """Creates a unique output directory for a benchmark run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for directory path
    safe_model_name = re.sub(r'[^\w\-.]', '_', model)
    dir_name = f"{task_name}_{provider}_{safe_model_name}_{timestamp}"
    output_path = os.path.join(base_dir, dir_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def parse_llm_score(response_text: str) -> int | None:
    """Extracts a score like <rating>X</rating> from LLM response."""
    match = re.search(r"<rating>([1-3])</rating>", response_text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None # Return None if pattern not found or invalid number