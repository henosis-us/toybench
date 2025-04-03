import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def load_config():
    """Loads configuration, prioritizing environment variables."""
    load_dotenv() # Load .env file if present

    config = {
        'gemini_api_key': os.getenv('GOOGLE_API_KEY'),
        # Add placeholders for other providers later
        # 'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'default_gemini_model': 'gemini-2.5-pro-exp-03-25', # Or your preferred default # 'gemini-2.5-pro-exp-03-25'
        'evaluator_model': 'gemini-1.5-flash-8b', # Default evaluator model
        'task_definitions_dir': 'tasks' # Directory containing task prompts
    }

    if not config['gemini_api_key']:
        logger.warning("GOOGLE_API_KEY environment variable not set.")
        # You might want to raise an error here or handle it differently
        # raise ValueError("GOOGLE_API_KEY must be set in environment variables or .env file")

    return config

# Load config globally on import? Or pass it around? Passing is cleaner.
# CONFIG = load_config()