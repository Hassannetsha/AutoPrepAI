#!/usr/bin/env python3
"""
Test script to demonstrate API key rotation in action
"""

import time
import sys
from api_key_manager import get_key_manager


def demo_status():
    """Demo 1: Check current status"""
    print("\n" + "="*70)
    print("DEMO 1: Current Key Status")
    print("="*70)
    
    manager = get_key_manager()
    
    print(f"\nTotal Keys Available: {manager.get_total_keys_count()}")
    print(f"Currently Using: Key #{manager.current_index + 1}")
    print(f"Active Keys: {manager.get_available_keys_count()}/{manager.get_total_keys_count()}")
    print(f"\nCurrent Key: {manager.get_current_key()[:40]}...")


def demo_rotation():
    """Demo 2: Simulate key rotation"""
    print("\n" + "="*70)
    print("DEMO 2: Simulating Key Rotation")
    print("="*70)
    
    manager = get_key_manager()
    
    print("\nScenario: Rate limit hit on key #1, rotating to #2...")
    print(f"\nStep 1: Currently using Key #{manager.current_index + 1}")
    print(f"        {manager.get_current_key()[:40]}...")
    
    time.sleep(1)
    
    print("\nStep 2: Rate limit error detected!")
    print("        Marking key #1 as rate-limited...")
    manager.mark_key_failed()
    print(f"        ⚠️  Key marked. Available keys: {manager.get_available_keys_count()}/12")
    
    time.sleep(1)
    
    print("\nStep 3: Rotating to next available key...")
    new_key = manager.rotate_key()
    print(f"        ✅ Rotated to Key #{manager.current_index + 1}")
    print(f"        {new_key[:40]}...")
    
    print("\nStep 4: Retrying request with new key...")
    time.sleep(1)
    print("        ✅ Request successful!")


def demo_multi_rotation():
    """Demo 3: Multiple rotations"""
    print("\n" + "="*70)
    print("DEMO 3: Multiple Key Rotations")
    print("="*70)
    
    manager = get_key_manager()
    manager.reset_failed_keys()  # Start fresh
    
    print("\nScenario: Experiencing multiple rate limits, rotating through keys...\n")
    
    for rotation in range(1, 4):
        print(f"Rotation #{rotation}:")
        print(f"  Current Key: #{manager.current_index + 1}/12")
        
        if rotation > 1:
            time.sleep(0.5)
            manager.mark_key_failed()
            manager.rotate_key()
            print(f"  After rotation: Key #{manager.current_index + 1}/12")
        
        print(f"  Available keys: {manager.get_available_keys_count()}/12")
        print()
    
    # Reset for clean state
    manager.reset_failed_keys()
    print("Demo complete - keys reset to initial state\n")


def demo_recovery():
    """Demo 4: Key recovery"""
    print("\n" + "="*70)
    print("DEMO 4: Key Recovery After Rate Limit Window")
    print("="*70)
    
    manager = get_key_manager()
    manager.reset_failed_keys()  # Start fresh
    
    print("\nScenario: All keys hit rate limit, then window expires...\n")
    
    # Simulate all keys hitting limit
    print("Marking all keys as rate-limited...")
    for key in manager.api_keys:
        manager.mark_key_failed()
    
    print(f"Available keys: {manager.get_available_keys_count()}/12")
    print("❌ System would now show: 'All keys rate-limited'\n")
    
    time.sleep(1)
    
    print("Waiting 60 minutes (simulated)...")
    print("Rate limit window expires on Groq API...\n")
    
    print("Running: python api_key_utility.py reset")
    manager.reset_failed_keys()
    
    print(f"✅ Keys recovered!")
    print(f"Available keys: {manager.get_available_keys_count()}/12")
    print("Ready to use again!\n")


def demo_key_list():
    """Demo 5: View all keys"""
    print("\n" + "="*70)
    print("DEMO 5: All Loaded Keys")
    print("="*70 + "\n")
    
    manager = get_key_manager()
    
    print(f"Loaded {manager.get_total_keys_count()} keys:\n")
    
    for i, key in enumerate(manager.api_keys, 1):
        status = "❌ Failed" if key in manager.failed_keys else "✅ Active"
        current = " ← Current" if (i - 1) == manager.current_index else ""
        print(f"{i:2d}. {key[:35]}... {status}{current}")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("API KEY ROTATION SYSTEM - DEMONSTRATION")
    print("="*70)
    
    demos = [
        ("1", "Current Status", demo_status),
        ("2", "Key Rotation on Rate Limit", demo_rotation),
        ("3", "Multiple Rotations", demo_multi_rotation),
        ("4", "Key Recovery", demo_recovery),
        ("5", "All Keys List", demo_key_list),
        ("all", "Run All Demos", None),
    ]
    
    if len(sys.argv) < 2:
        print("\nAvailable Demos:\n")
        for code, name, _ in demos[:-1]:
            print(f"  python test_key_rotation.py {code}  - {name}")
        print(f"  python test_key_rotation.py all  - {demos[-1][1]}")
        print("\nUsage example:")
        print("  python test_key_rotation.py 1")
        return
    
    choice = sys.argv[1].lower()
    
    try:
        if choice == "all":
            for code, name, func in demos[:-1]:
                if func:
                    func()
                    time.sleep(1)
        else:
            found = False
            for code, name, func in demos:
                if code == choice and func:
                    func()
                    found = True
                    break
            if not found:
                print(f"❌ Unknown demo: {choice}")
                print("Valid options: 1, 2, 3, 4, 5, all")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
