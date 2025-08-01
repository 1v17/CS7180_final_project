from transformers import AutoModelForCausalLM, AutoTokenizer
from mem0 import Memory
from dotenv import load_dotenv
import traceback


class ChatBot:
    """A chatbot that combines local LLM with memory capabilities."""
    
    def __init__(self, model_path="./models/test_local_model"):
        """
        Initialize the chatbot with model and memory.
        
        Args:
            model_path (str): Path to the local model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.memory = None
        
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self._load_model()
        self._setup_memory()
    
    def _load_model(self):
        """Load the local model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("\nModel loaded successfully!")
    
    def _setup_memory(self):
        """Set up mem0 with OpenRouter configuration."""
        memory_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-3.5-turbo",
                }
            }
        }
        
        self.memory = Memory.from_config(memory_config)
        print("\nMemory system initialized!")
    
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
            memory_list = relevant_memories['results'][:3]
        else:
            memory_list = list(relevant_memories)[:3] if relevant_memories else []
        
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
            max_length=512  # Adjust as needed
        )
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=inputs.input_ids.shape[1] + max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
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
        
        return response.strip()
    
    def get_memories(self, query, user_id="default_user", limit=5):
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
    



def main():
    """Main function for interactive chatbot usage."""
    try:
        # Initialize chatbot
        print("Initializing ChatBot...")
        chatbot = ChatBot()
        print("ChatBot ready! Type 'quit' to exit.\n")
        
        # Interactive chat loop
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
                
            # Get bot response
            response = chatbot.chat(user_input)
            print(f"Bot: {response}\n")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
