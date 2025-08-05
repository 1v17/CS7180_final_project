"""
Main runner for the Ebbinghaus Memory Chatbot

This script provides the main entry point for running the chatbot with
Ebbinghaus memory capabilities. It handles initialization and provides
an interactive command-line interface.
"""

import traceback
from chatbot import ChatBot


def get_user_preferences():
    """Get user preferences for memory mode and configuration."""
    print("=== Ebbinghaus Memory ChatBot Configuration ===")
    
    # Get memory mode
    while True:
        print("\nChoose memory mode:")
        print("  1. Standard - Traditional perfect memory (default Mem0 behavior)")
        print("  2. Ebbinghaus - Memory with forgetting curve and decay over time")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            memory_mode = "standard"
            config_mode = "default"
            break
        elif choice == "2":
            memory_mode = "ebbinghaus"
            
            # Get scheduler configuration for ebbinghaus mode
            while True:
                print("\nChoose scheduler configuration for Ebbinghaus mode:")
                print("  1. Testing - Fast decay, 1-minute maintenance interval (for development)")
                print("  2. Production - Slower decay, 1-hour maintenance interval (for real use)")
                print("  3. Standard - Default settings with moderate decay")
                
                scheduler_choice = input("Enter your choice (1, 2, or 3): ").strip()
                
                if scheduler_choice == "1":
                    config_mode = "testing"
                    break
                elif scheduler_choice == "2":
                    config_mode = "production"
                    break
                elif scheduler_choice == "3":
                    config_mode = "default"
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print(f"\nSelected: {memory_mode.title()} memory mode with {config_mode} configuration")
    return memory_mode, config_mode


def main():
    """Main function for interactive chatbot usage."""
    try:
        # Get user preferences
        memory_mode, config_mode = get_user_preferences()
        
        # Initialize chatbot with user preferences
        print(f"\nInitializing Ebbinghaus Memory ChatBot...")
        print(f"Memory Mode: {memory_mode}")
        print(f"Configuration: {config_mode}")
        
        # Test model loading and memory setup: using ./models/test_local_model
        # chatbot = ChatBot(memory_mode=memory_mode, config_mode=config_mode)
        # Uncomment the line below to specify a custom model path
        chatbot = ChatBot(model_path="./models/fine-tuned-model", 
                          memory_mode=memory_mode, config_mode=config_mode)
        
        print("\nChatBot ready! Type 'quit' to exit.")
        print("\nAvailable commands:")
        print("  /help - Show available commands")
        print("  /memory_status - Show memory mode and statistics")
        print("  /quit or quit - Exit the chatbot\n")
        
        # Interactive chat loop
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', '/quit']:
                chatbot.shutdown()
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                chatbot.handle_command(user_input)
            else:
                # Get bot response
                response = chatbot.chat(user_input)
                print(f"Bot: {response}\n")
            
    except KeyboardInterrupt:
        if 'chatbot' in locals():
            chatbot.shutdown()
        print("\nGoodbye!")
    except Exception as e:
        if 'chatbot' in locals():
            chatbot.shutdown()
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
