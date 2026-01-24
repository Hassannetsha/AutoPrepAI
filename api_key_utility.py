#!/usr/bin/env python3
"""
Utility script for managing API keys - Test and monitor key status
"""

import sys
import time
from api_key_manager import get_key_manager


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def show_key_status():
    """Display current key manager status"""
    manager = get_key_manager()
    
    print_header("API KEY STATUS")
    
    print(f"Total Keys Loaded: {manager.get_total_keys_count()}")
    print(f"Available Keys: {manager.get_available_keys_count()}")
    print(f"Current Key Index: {manager.current_index + 1}")
    print(f"\nCurrent Key: {manager.get_current_key()[:30]}...")
    
    if manager.failed_keys:
        print(f"\n[WARN] Failed Keys ({len(manager.failed_keys)}):")
        for i, key in enumerate(manager.failed_keys, 1):
            print(f"   {i}. {key[:30]}...")
    else:
        print("\n[OK] No failed keys")


def rotate_key():
    """Rotate to next available key"""
    manager = get_key_manager()
    
    print_header("ROTATING API KEY")
    
    try:
        new_key = manager.rotate_key()
        print(f"[OK] Successfully rotated to key #{manager.current_index + 1}")
        print(f"New Key: {new_key[:30]}...")
    except RuntimeError as e:
        print(f"[ERROR] Rotation failed: {e}")


def reset_keys():
    """Reset all failed keys"""
    manager = get_key_manager()
    
    print_header("RESETTING FAILED KEYS")
    
    if not manager.failed_keys:
        print("[OK] No failed keys to reset")
    else:
        manager.reset_failed_keys()
        print(f"[OK] Reset {len(manager.failed_keys)} failed keys")
        show_key_status()


def mark_failed():
    """Manually mark current key as failed (for testing)"""
    manager = get_key_manager()
    
    print_header("MARKING KEY AS FAILED")
    
    current_key = manager.get_current_key()
    manager.mark_key_failed()
    print(f"[OK] Key #{manager.current_index + 1} marked as failed")
    show_key_status()


def test_key_rotation():
    """Test key rotation cycle"""
    manager = get_key_manager()
    
    print_header("TESTING KEY ROTATION")
    
    initial_available = manager.get_available_keys_count()
    print(f"Starting with {initial_available} available keys\n")
    
    for i in range(min(3, initial_available)):
        print(f"Step {i + 1}:")
        print(f"  Current Key #{manager.current_index + 1}")
        
        try:
            manager.mark_key_failed()
            next_key = manager.rotate_key()
            print(f"  [OK] Rotated to Key #{manager.current_index + 1}")
            print(f"  Available keys: {manager.get_available_keys_count()}\n")
        except RuntimeError as e:
            print(f"  [WARN] {e}\n")
            break
        
        time.sleep(0.5)
    
    # Reset for clean state
    manager.reset_failed_keys()
    print("\n[OK] Test rotation complete - all keys reset")


def list_all_keys():
    """List all loaded API keys (first 20 chars)"""
    manager = get_key_manager()
    
    print_header("ALL LOADED API KEYS")
    
    for i, key in enumerate(manager.api_keys, 1):
        status = "[FAIL]" if key in manager.failed_keys else "[OK]"
        marker = " <- Current" if i - 1 == manager.current_index else ""
        print(f"{i:2d}. {key[:30]}... {status}{marker}")


def main():
    """CLI interface for key management"""
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("  API Key Manager - Utility Tool")
        print("="*60)
        print("\nUsage: python api_key_utility.py <command>\n")
        print("Commands:")
        print("  status     - Show current key status")
        print("  list       - List all loaded API keys")
        print("  rotate     - Rotate to next available key")
        print("  reset      - Reset all failed keys")
        print("  test       - Test key rotation cycle")
        print("  mark_failed - Mark current key as failed (testing)")
        print("\nExample:")
        print("  python api_key_utility.py status\n")
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "status":
            show_key_status()
        elif command == "list":
            list_all_keys()
        elif command == "rotate":
            rotate_key()
        elif command == "reset":
            reset_keys()
        elif command == "test":
            test_key_rotation()
        elif command == "mark_failed":
            mark_failed()
        else:
            print(f"[ERROR] Unknown command: {command}")
            print("Use 'python api_key_utility.py' to see available commands")
    except Exception as e:
        print(f"[ERROR] Error: {e}")


if __name__ == "__main__":
    main()
