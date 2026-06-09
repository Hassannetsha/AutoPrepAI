import os

# Legacy single key (kept for backward compatibility).
# Set GROQ_API_KEY in your environment instead of committing secrets.
API_KEY = os.getenv("GROQ_API_KEY", "")

# For multi-key rotation support, use api_key_manager instead
# from api_key_manager import get_api_key
# api_key = get_api_key()  # Gets current key
# rotate key if needed: from api_key_manager import rotate_api_key
