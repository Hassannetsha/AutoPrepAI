"""
API Key Manager - Handles rotation and management of multiple Groq API keys
to avoid rate limiting issues.
"""

import os
from typing import List, Optional
from pathlib import Path


class APIKeyManager:
    """Manages multiple API keys with rotation and retry logic."""
    
    def __init__(self, keys_file: str = "ApiKeys.txt"):
        """
        Initialize the API Key Manager.
        
        Args:
            keys_file: Path to the file containing API keys (one per line)
        """
        self.keys_file = keys_file
        self.api_keys: List[str] = []
        self.current_index = 0
        self.failed_keys = set()  # Track keys that have hit rate limits
        
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load API keys from file."""
        if not os.path.exists(self.keys_file):
            raise FileNotFoundError(f"API keys file not found: {self.keys_file}")
        
        with open(self.keys_file, 'r') as f:
            lines = f.readlines()
            self.api_keys = [line.strip() for line in lines if line.strip()]
        
        if not self.api_keys:
            raise ValueError("No API keys found in the keys file.")
        
        print(f"[OK] Loaded {len(self.api_keys)} API keys")
    
    def get_current_key(self) -> str:
        """Get the current active API key."""
        if not self.api_keys:
            raise RuntimeError("No API keys available.")
        
        key = self.api_keys[self.current_index]
        return key
    
    def rotate_key(self) -> str:
        """
        Rotate to the next available API key.
        Skips keys that have previously failed.
        
        Returns:
            The next API key to use
            
        Raises:
            RuntimeError: If all keys have been exhausted
        """
        available_keys = len(self.api_keys) - len(self.failed_keys)
        
        if available_keys <= 0:
            raise RuntimeError(
                f"❌ All {len(self.api_keys)} API keys have reached rate limits. "
                "Please wait before retrying or add more keys."
            )
        
        # Move to next key
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        
        # Skip failed keys
        attempts = 0
        while self.api_keys[self.current_index] in self.failed_keys and attempts < len(self.api_keys):
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            attempts += 1
        
        key = self.api_keys[self.current_index]
        print(f"🔄 Rotated to API key #{self.current_index + 1}")
        return key
    
    def mark_key_failed(self, key: Optional[str] = None) -> None:
        """
        Mark the current (or specified) key as having hit rate limits.
        
        Args:
            key: Optional specific key to mark as failed. If None, marks current key.
        """
        if key is None:
            key = self.get_current_key()
        
        self.failed_keys.add(key)
        print(f"[WARN] Key marked as rate-limited: {key[:20]}...")
    
    def mark_key_recovered(self, key: Optional[str] = None) -> None:
        """
        Mark a key as recovered (useful if you wait and retry later).
        
        Args:
            key: Optional specific key to mark as recovered. If None, marks current key.
        """
        if key is None:
            key = self.get_current_key()
        
        if key in self.failed_keys:
            self.failed_keys.discard(key)
            print(f"[OK] Key marked as recovered: {key[:20]}...")
    
    def get_available_keys_count(self) -> int:
        """Get the number of available (non-failed) keys."""
        return len(self.api_keys) - len(self.failed_keys)
    
    def get_total_keys_count(self) -> int:
        """Get the total number of loaded keys."""
        return len(self.api_keys)
    
    def reset_failed_keys(self) -> None:
        """Reset all failed keys (useful after a timeout/wait period)."""
        self.failed_keys.clear()
        print("[RESET] All keys reset and available for use")
    
    def set_active_key(self, key_index: int) -> str:
        """
        Manually set the active key by index.
        
        Args:
            key_index: Index of the key (0-based)
            
        Returns:
            The selected API key
        """
        if not (0 <= key_index < len(self.api_keys)):
            raise IndexError(f"Key index {key_index} out of range (0-{len(self.api_keys)-1})")
        
        self.current_index = key_index
        return self.get_current_key()


# Create a global instance for easy access
_key_manager: Optional[APIKeyManager] = None


def get_key_manager(keys_file: str = "ApiKeys.txt") -> APIKeyManager:
    """Get or create the global API Key Manager instance."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager(keys_file)
    return _key_manager


def get_api_key() -> str:
    """Convenience function to get the current API key."""
    return get_key_manager().get_current_key()


def rotate_api_key() -> str:
    """Convenience function to rotate to the next API key."""
    return get_key_manager().rotate_key()
