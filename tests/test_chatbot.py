import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to the Python path to import chatbot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import ChatBot


class TestChatBot(unittest.TestCase):
    """Simple test cases for the ChatBot class - mirroring the original main function tests."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the model loading, memory setup, and scheduler setup to avoid loading actual models
        with patch.object(ChatBot, '_load_model'), \
             patch.object(ChatBot, '_setup_memory'), \
             patch.object(ChatBot, '_setup_scheduler'):
            self.chatbot = ChatBot()
            
        # Mock the model, tokenizer, memory, and scheduler
        self.chatbot.model = Mock()
        self.chatbot.tokenizer = Mock()
        self.chatbot.memory = Mock()
        self.chatbot.scheduler = Mock()
        
        # Set up tokenizer mocks
        self.chatbot.tokenizer.pad_token = "[PAD]"
        self.chatbot.tokenizer.eos_token = "[EOS]"
        self.chatbot.tokenizer.pad_token_id = 0
        self.chatbot.tokenizer.eos_token_id = 1
        
        # Set up memory mode for any scheduler-related checks
        self.chatbot.memory_mode = "standard"
        self.chatbot.config_mode = "default"

    def tearDown(self):
        """Clean up after each test method."""
        # Ensure scheduler is properly mocked and won't cause issues
        if hasattr(self.chatbot, 'scheduler') and self.chatbot.scheduler:
            if hasattr(self.chatbot.scheduler, 'stop'):
                self.chatbot.scheduler.stop = Mock()  # Mock the stop method to avoid real shutdown

    def test_first_chat_hello_i_like_coffee(self):
        """Test: response1 = chatbot.chat("Hello, I like coffee")"""
        # Mock memory search to return empty results (first conversation)
        self.chatbot.memory.search.return_value = {'results': []}
        
        # Mock tokenizer and model response
        mock_inputs = Mock()
        mock_inputs.input_ids = Mock()
        mock_inputs.input_ids.shape = (1, 8)
        mock_inputs.attention_mask = Mock()
        
        self.chatbot.tokenizer.return_value = mock_inputs
        self.chatbot.model.generate.return_value = [[0] * 12]
        self.chatbot.tokenizer.decode.return_value = "Nice to meet you! I'll remember that you like coffee."
        
        # Mock random for periodic forgetting to avoid randomness in tests
        with patch('random.random', return_value=0.5):  # Return value > 0.1 to avoid forgetting
            # Test the chat method
            response = self.chatbot.chat("Hello, I like coffee")
        
        # Assertions
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.chatbot.memory.add.assert_called_once()
        print(f"✓ Test 1 passed: Bot responded to 'Hello, I like coffee'")

    def test_second_chat_what_do_i_like_to_drink(self):
        """Test: response2 = chatbot.chat("What do I like to drink?")"""
        # Mock memory search to return previous conversation
        mock_memory_response = {
            'results': [
                {'memory': 'User: Hello, I like coffee, Assistant: Nice to meet you! I\'ll remember that you like coffee.'}
            ]
        }
        self.chatbot.memory.search.return_value = mock_memory_response
        
        # Mock tokenizer and model response
        mock_inputs = Mock()
        mock_inputs.input_ids = Mock()
        mock_inputs.input_ids.shape = (1, 10)
        mock_inputs.attention_mask = Mock()
        
        self.chatbot.tokenizer.return_value = mock_inputs
        self.chatbot.model.generate.return_value = [[0] * 15]
        self.chatbot.tokenizer.decode.return_value = "Based on our conversation, you like coffee!"
        
        # Mock random for periodic forgetting to avoid randomness in tests
        with patch('random.random', return_value=0.5):  # Return value > 0.1 to avoid forgetting
            # Test the chat method
            response = self.chatbot.chat("What do I like to drink?")
        
        # Assertions
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.chatbot.memory.search.assert_called_with("What do I like to drink?", user_id="default_user")
        print(f"✓ Test 2 passed: Bot responded to 'What do I like to drink?'")

    def test_get_memories_drink(self):
        """Test: memories = chatbot.get_memories("drink", limit=3)"""
        # Mock memory search response
        mock_memory_response = {
            'results': [
                {'memory': 'User: Hello, I like coffee, Assistant: Nice to meet you!'},
                {'memory': 'User: What do I like to drink?, Assistant: Based on our conversation, you like coffee!'},
                {'memory': 'User: I also enjoy tea, Assistant: Tea is great too!'}
            ]
        }
        self.chatbot.memory.search.return_value = mock_memory_response
        
        # Test get_memories method
        memories = self.chatbot.get_memories("drink", limit=3)
        
        # Assertions
        self.assertIsInstance(memories, list)
        self.assertEqual(len(memories), 3)
        self.chatbot.memory.search.assert_called_with("drink", user_id="default_user")
        
        # Verify each memory has the expected structure
        for memory in memories:
            self.assertIn('memory', memory)
            
        print(f"✓ Test 3 passed: Retrieved {len(memories)} memories for 'drink'")

    def test_chatbot_initialization_with_modes(self):
        """Test ChatBot initialization with different memory modes."""
        # Test standard mode initialization
        with patch.object(ChatBot, '_load_model'), \
             patch.object(ChatBot, '_setup_memory'), \
             patch.object(ChatBot, '_setup_scheduler'):
            chatbot_standard = ChatBot(memory_mode="standard", config_mode="default")
            self.assertEqual(chatbot_standard.memory_mode, "standard")
            self.assertEqual(chatbot_standard.config_mode, "default")
        
        # Test ebbinghaus mode initialization
        with patch.object(ChatBot, '_load_model'), \
             patch.object(ChatBot, '_setup_memory'), \
             patch.object(ChatBot, '_setup_scheduler'):
            chatbot_ebbinghaus = ChatBot(memory_mode="ebbinghaus", config_mode="testing")
            self.assertEqual(chatbot_ebbinghaus.memory_mode, "ebbinghaus")
            self.assertEqual(chatbot_ebbinghaus.config_mode, "testing")
        
        print("✓ Test 4 passed: ChatBot initialization with different modes")

    def test_command_handling(self):
        """Test command handling functionality."""
        # Mock the command handler methods
        self.chatbot._show_help = Mock()
        self.chatbot._show_memory_status = Mock()
        self.chatbot._show_maintenance_status = Mock()
        self.chatbot._force_maintenance = Mock()
        
        # Test help command
        self.chatbot.handle_command('/help')
        self.chatbot._show_help.assert_called_once()
        
        # Test memory status command
        self.chatbot.handle_command('/memory_status')
        self.chatbot._show_memory_status.assert_called_once()
        
        # Test maintenance status command
        self.chatbot.handle_command('/memory_maintenance')
        self.chatbot._show_maintenance_status.assert_called_once()
        
        # Test force maintenance command
        self.chatbot.handle_command('/force_maintenance')
        self.chatbot._force_maintenance.assert_called_once()
        
        print("✓ Test 5 passed: Command handling works correctly")

    def test_shutdown(self):
        """Test graceful shutdown functionality."""
        # Mock the scheduler
        self.chatbot.scheduler = Mock()
        
        # Test shutdown
        with patch('builtins.print'):  # Suppress print output during test
            self.chatbot.shutdown()
        
        # Verify scheduler was stopped
        self.chatbot.scheduler.stop.assert_called_once()
        
        print("✓ Test 6 passed: Graceful shutdown works correctly")


if __name__ == '__main__':
    unittest.main(verbosity=2)
