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
        
        # xAI Grok
        'xai_api_key':         os.getenv('XAI_API_KEY'),  # Added for xAI Grok support
        
        # Anthropic Claude
        'anthropic_api_key':   os.getenv('ANTHROPIC_API_KEY2'),  # New: Anthropic API key
        
        # Quality Compute Simulator
        'quality_compute_api_key': os.getenv('QC_API_KEY'),  # API key for Quality Compute simulator
        'QUALITY_COMPUTE_URL': os.getenv('QUALITY_COMPUTE_URL'),  # URL for Quality Compute simulator
        
        # Kimi (Moonshot AI)
        'kimi_api_key':        os.getenv('KIMI_API_KEY2'),  # Added for Kimi suppor
        
        # OpenRouter (added per integration plan)
        'openrouter_api_key':  os.getenv('OPENROUTER_API_KEY'),  # API key for OpenRouter
        'default_openrouter_model': 'openai/gpt-oss-120b',      # Default model for OpenRouter provider
        
        # Default models
        'default_gemini_model': 'gemini-2.0-flash',
        'default_openai_model': 'o4-mini',
        'default_grok_model':   'grok-3-mini-beta',  # Added default model for Grok provider
        'default_anthropic_model': 'claude-3-5-sonnet-20241022',  # New: Default Anthropic model
        
        # Evaluator (text-based by default)
        'evaluator_model':     'gemini-1.5-flash-8b',
        
        # Where task prompt files live
        'task_definitions_dir': 'tasks'
    }
    
    # Warnings for missing API keys
    if not config['gemini_api_key']:
        logger.warning("GOOGLE_API_KEY environment variable not set.")
    if not config['openai_api_key']:
        logger.warning("OPENAI_API_KEY environment variable not set.")
    if not config['xai_api_key']:
        logger.warning("XAI_API_KEY environment variable not set.")
    if not config['anthropic_api_key']:
        logger.warning("ANTHROPIC_API_KEY environment variable not set.")
    if not config['quality_compute_api_key']:
        logger.warning("QC_API_KEY environment variable not set.")
    if not config['QUALITY_COMPUTE_URL']:
        logger.warning("QUALITY_COMPUTE_URL environment variable not set.")
    if not config['kimi_api_key']:
        logger.warning("KIMI_API_KEY environment variable not set.")  # Added warning for Kimi
    if not config['openrouter_api_key']:  # Added warning for OpenRouter
        logger.warning("OPENROUTER_API_KEY environment variable not set.")
    
    return config