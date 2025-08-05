from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from ebbinghaus_memory import EbbinghausMemory
from memory_config import MemoryConfig

DEFAULT_MAX_NEW_TOKENS = 50
RELEVANT_MEMORIES_LIMIT = 3
INPUT_MAX_LENGTH = 512
OUTPUT_TEMPORARY = 0.7
OUTPUT_TOP_P = 0.9
FORGETTING_PROBABILITY = 0.1  # 10% chance to trigger forgetting process

class ChatBot:
    """A chatbot that combines local LLM with Ebbinghaus memory capabilities."""
    
    def __init__(self, model_path="./models/test_local_model", memory_mode="standard", config_mode="default"):
        """
        Initialize the chatbot with model and Ebbinghaus memory.
        
        Args:
            model_path (str): Path to the local model directory
            memory_mode (str): Memory mode - "standard" or "ebbinghaus"
            config_mode (str): Configuration preset - "default", "testing", or "production"
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.memory = None
        self.memory_mode = memory_mode
        self.config_mode = config_mode
        
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self._load_model()
        self._setup_memory()
    
    def _load_model(self):
        """Load the local model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            print("Tokenizer loaded successfully!")
        except Exception as e:
            print(f"Tokenizer loading failed: {e}")
            return
        
        # Load model
        try:
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype="auto",
                device_map="auto"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            return
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("\nModel loaded successfully!")
    
    def _setup_memory(self):
        """Set up Ebbinghaus memory with configuration."""
        config = MemoryConfig.get_config(self.config_mode)
        config["memory_mode"] = self.memory_mode  # Override with user preference
        
        self.memory = EbbinghausMemory(config=config, memory_mode=self.memory_mode)
        print(f"\nEbbinghaus Memory system initialized in '{self.memory_mode}' mode!")
    
    def chat(self, message, user_id="default_user", max_new_tokens=50):
        """
        Generate a response to the user's message with memory context.
        
        Args:
            message (str): User's input message
            user_id (str): Unique identifier for the user
            max_new_tokens (int): Maximum number of new tokens to generate
            
        Returns:
            str: The chatbot's response
        """
        # Get relevant memories
        relevant_memories = self.memory.search(message, user_id=user_id)
        
        # Handle the dictionary response from memory.search
        if isinstance(relevant_memories, dict) and 'results' in relevant_memories:
            memory_list = relevant_memories['results'][:RELEVANT_MEMORIES_LIMIT]
        else:
            memory_list = list(relevant_memories)[:RELEVANT_MEMORIES_LIMIT] if relevant_memories else []
        
        context = "\n".join([mem["memory"] for mem in memory_list])
        
        # Create prompt with context
        if context:
            prompt = f"Context: {context}\nUser: {message}\nAssistant:"
        else:
            prompt = f"User: {message}\nAssistant:"
        
        # Generate response with proper tokenization
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=False,  # No padding needed for single sequence
            truncation=True,
            max_length=INPUT_MAX_LENGTH
        )
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=inputs.input_ids.shape[1] + max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=OUTPUT_TEMPORARY,
            top_p=OUTPUT_TOP_P
        )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Store conversation in memory
        self.memory.add(
            f"User: {message}, Assistant: {response}", 
            user_id=user_id
        )
        
        # Periodically trigger forgetting process for this user (Ebbinghaus mode only)
        if self.memory_mode == "ebbinghaus":
            try:
                # Trigger forgetting occasionally (every ~10 interactions per user)
                import random
                if random.random() < FORGETTING_PROBABILITY:  # Configurable chance
                    forgetting_results = self.memory.forget_weak_memories(user_id=user_id)
                    if forgetting_results.get('forgotten', 0) > 0:
                        print(f"[Memory] Forgot {forgetting_results['forgotten']} weak memories")
            except Exception as e:
                # Don't let forgetting errors break the chat
                pass
        
        return response.strip()
    
    def get_memories(self, query, user_id="default_user", limit=RELEVANT_MEMORIES_LIMIT):
        """
        Retrieve memories for a user based on a query.
        
        Args:
            query (str): Query to search memories
            user_id (str): User identifier
            limit (int): Maximum number of memories to return
            
        Returns:
            list: List of relevant memories
        """
        try:
            memories = self.memory.search(query, user_id=user_id)
            # print(f"Retrieved memories: {memories}, type: {type(memories)}")
            
            # Handle the dictionary response from memory.search (same as in chat method)
            if isinstance(memories, dict) and 'results' in memories:
                memory_list = memories['results'][:limit]
            else:
                memory_list = list(memories)[:limit] if memories else []
            
            return memory_list
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []
    
    def handle_command(self, command: str) -> None:
        """
        Handle special commands from the user.
        
        Args:
            command (str): Command string starting with '/'
        """
        command = command.lower().strip()
        
        if command == '/help':
            self._show_help()
        elif command == '/memory_status':
            self._show_memory_status()
        else:
            print(f"Unknown command: {command}. Type /help for available commands.")
    
    def _show_help(self) -> None:
        """Show available commands."""
        print("\nAvailable commands:")
        print("  /help - Show this help message")
        print("  /memory_status - Show current memory mode and statistics")
        print("  /quit - Exit the chatbot")
        print()
    
    def _show_memory_status(self) -> None:
        """Show current memory mode and statistics."""
        print(f"\n=== Memory Status ===")
        print(f"Current Mode: {self.memory_mode}")
        print(f"Config Mode: {self.config_mode}")
        
        try:
            if hasattr(self.memory, 'get_memory_statistics'):
                # Use default_user for statistics to avoid the user_id requirement
                stats = self.memory.get_memory_statistics(user_id="default_user")
                
                # Check if stats is actually a dictionary
                if isinstance(stats, dict):
                    # Handle the case where total_memories might be a string (error message)
                    total_memories = stats.get('total_memories', 'N/A')
                    print(f"Total Memories: {total_memories}")
                    
                    if self.memory_mode == "ebbinghaus":
                        strong_count = stats.get('strong_memories', 'N/A')
                        weak_count = stats.get('weak_memories', 'N/A') 
                        archived_count = stats.get('archived_memories', 'N/A')
                        avg_strength = stats.get('average_strength', 0.0)
                        oldest_age = stats.get('oldest_memory_age', 'N/A')
                        
                        # Get the actual weak memory threshold from configuration
                        weak_threshold = self.memory.fc_config.get("min_retention_threshold", 0.1)
                        strong_threshold = self.memory.STRONG_MEMORY_THRESHOLD
                        
                        print(f"Strong Memories (>{strong_threshold}): {strong_count}")
                        print(f"Weak Memories (<{weak_threshold}): {weak_count}")
                        print(f"Archived Memories: {archived_count}")
                        
                        if isinstance(avg_strength, (int, float)):
                            print(f"Average Strength: {avg_strength:.3f}")
                        else:
                            print(f"Average Strength: {avg_strength}")
                            
                        print(f"Oldest Memory: {oldest_age}")
                    else:
                        print("Memory strength tracking disabled in standard mode")
                else:
                    # stats is not a dictionary (might be an error string)
                    print(f"Memory statistics error: {stats}")
            else:
                print("Memory statistics not available")
        except Exception as e:
            print(f"Error retrieving memory statistics: {e}")
        
        print()
    
    def shutdown(self) -> None:
        """Gracefully shutdown the chatbot."""
        print("Shutting down chatbot...")
        print("Shutdown complete")
    

# Main function removed - use main.py to run the chatbot
