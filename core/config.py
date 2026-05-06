"""Configuration for PromptGen V2"""
import os
from google import genai

def get_api_key():
    """Get API key from environment variable (for non-Streamlit usage)"""
    return os.getenv('GOOGLE_API_KEY', None)

# Model configurations
PROMPT_GENERATOR_MODEL = 'gemini-2.5-flash'
PROMPT_TESTER_MODEL = 'gemini-2.5-pro'
ANALYZER_MODEL = 'gemini-2.5-pro'

# Workflow settings
MAX_ITERATIONS = 5
TARGET_SCORE = 100

def create_client(api_key=None):
    """Create a google-genai Client with the given or env API key."""
    key = api_key or get_api_key()
    if key:
        return genai.Client(api_key=key)
    return None

# Module-level client (configured later via configure_client)
_client = None

def configure_client(api_key):
    """Configure the global client with an API key."""
    global _client
    _client = genai.Client(api_key=api_key)

def get_client():
    """Get the configured client."""
    global _client
    if _client is None:
        key = get_api_key()
        if key:
            _client = genai.Client(api_key=key)
    return _client
