# config.py

import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def load_config():
    """Loads configuration, prioritizing environment variables."""
    load_dotenv()  # Load .env file if present

    config = {
        # Google Gemini
        'gemini_api_key':      os.getenv('GOOGLE_API_KEY'),
        # OpenAI
        'openai_api_key':      os.getenv('OPENAI_API_KEY'),

        # Default models
        'default_gemini_model': 'gemini-2.0-flash',
        'default_openai_model': 'o4-mini',

        # Evaluator (text‑based by default)
        'evaluator_model':     'gemini-1.5-flash-8b',

        # Where task prompt files live
        'task_definitions_dir':'tasks'
    }

    if not config['gemini_api_key']:
        logger.warning("GOOGLE_API_KEY environment variable not set.")
    if not config['openai_api_key']:
        logger.warning("OPENAI_API_KEY environment variable not set.")

    return config