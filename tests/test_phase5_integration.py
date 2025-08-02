"""
Test script for Phase 5 ChatBot integration with Ebbinghaus Memory

This script tests the basic functionality of the updated ChatBot class
with Ebbinghaus memory integration and command handling.
"""

import sys
import time
from chatbot import ChatBot


def test_chatbot_initialization():
    """Test chatbot initialization in different modes."""
    print("=== Testing ChatBot Initialization ===")
    
    # Test standard mode
    try:
        print("\n1. Testing standard mode initialization...")
        chatbot_standard = ChatBot(memory_mode="standard", config_mode="default")
        print("✅ Standard mode initialization successful")
        chatbot_standard.shutdown()
    except Exception as e:
        print(f"❌ Standard mode initialization failed: {e}")
        return False
    
    # Test ebbinghaus mode with testing config
    try:
        print("\n2. Testing ebbinghaus mode (testing config) initialization...")
        chatbot_ebbinghaus = ChatBot(memory_mode="ebbinghaus", config_mode="testing")
        print("✅ Ebbinghaus mode (testing) initialization successful")
        chatbot_ebbinghaus.shutdown()
    except Exception as e:
        print(f"❌ Ebbinghaus mode (testing) initialization failed: {e}")
        return False
    
    # Test ebbinghaus mode with production config
    try:
        print("\n3. Testing ebbinghaus mode (production config) initialization...")
        chatbot_production = ChatBot(memory_mode="ebbinghaus", config_mode="production")
        print("✅ Ebbinghaus mode (production) initialization successful")
        chatbot_production.shutdown()
    except Exception as e:
        print(f"❌ Ebbinghaus mode (production) initialization failed: {e}")
        return False
    
    return True


def test_command_handling():
    """Test command handling functionality."""
    print("\n=== Testing Command Handling ===")
    
    try:
        chatbot = ChatBot(memory_mode="standard", config_mode="testing")
        
        # Test help command
        print("\n1. Testing /help command...")
        chatbot.handle_command('/help')
        
        # Test memory status command
        print("\n2. Testing /memory_status command...")
        chatbot.handle_command('/memory_status')
        
        # Test maintenance status command
        print("\n3. Testing /memory_maintenance command...")
        chatbot.handle_command('/memory_maintenance')
        
        # Test invalid command
        print("\n4. Testing invalid command...")
        chatbot.handle_command('/invalid_command')
        
        chatbot.shutdown()
        print("✅ Command handling tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Command handling tests failed: {e}")
        if 'chatbot' in locals():
            chatbot.shutdown()
        return False


def test_memory_integration():
    """Test memory integration with chat functionality."""
    print("\n=== Testing Memory Integration ===")
    
    try:
        chatbot = ChatBot(memory_mode="ebbinghaus", config_mode="testing")
        
        # Test basic chat with memory
        print("\n1. Testing chat with memory...")
        response1 = chatbot.chat("My name is Alice", user_id="test_user")
        print(f"Response 1: {response1}")
        
        response2 = chatbot.chat("What is my name?", user_id="test_user")
        print(f"Response 2: {response2}")
        
        # Test memory retrieval 
        print("\n2. Testing memory retrieval...")
        memories = chatbot.get_memories("name", user_id="test_user")
        print(f"Retrieved memories: {len(memories) if memories else 0}")
        
        chatbot.shutdown()
        print("✅ Memory integration tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Memory integration tests failed: {e}")
        if 'chatbot' in locals():
            chatbot.shutdown()
        return False


def run_all_tests():
    """Run all test functions."""
    print("Starting Phase 5 Integration Tests...")
    print("=" * 50)
    
    tests = [
        test_chatbot_initialization,
        test_command_handling,
        test_memory_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            time.sleep(1)  # Brief pause between tests
        except Exception as e:
            print(f"❌ Test {test_func.__name__} encountered an error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 5 implementation appears to be working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
