"""
Test script for memory_evaluator.py

This script tests the memory evaluation functionality
to ensure it correctly integrates with the existing ChatBot.
"""

import sys
import os
import logging

# Add the project root to the path so we can import evaluation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.evaluation_config import EvaluationConfig
from evaluation.memory_evaluator import MemoryEvaluator

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_chatbot_initialization():
    """Test ChatBot initialization for different memory modes."""
    print("🧪 Testing ChatBot Initialization...")
    
    try:
        config = EvaluationConfig(
            max_conversations=1,
            memory_modes=["standard"],  # Test one mode at a time to avoid file conflicts
            local_model_path="./models/TinyLlama-1.1B-Chat-v1.0"  # Use smaller test model
        )
        evaluator = MemoryEvaluator(config)
        
        # Test initialization with just standard mode first
        evaluator.initialize_chatbots()
        
        # Check that chatbots were created
        assert "standard" in evaluator.chatbots, "Standard ChatBot not initialized"
        
        print(f"✅ Successfully initialized {len(evaluator.chatbots)} chatbots:")
        for mode in evaluator.chatbots:
            print(f"  - {mode} mode chatbot")
        
        # Cleanup
        evaluator.cleanup_chatbots()
        
        # Now test ebbinghaus mode separately
        config.memory_modes = ["ebbinghaus"]
        config.local_model_path = "./models/TinyLlama-1.1B-Chat-v1.0"  # Use smaller test model
        evaluator = MemoryEvaluator(config)
        evaluator.initialize_chatbots()
        
        assert "ebbinghaus" in evaluator.chatbots, "Ebbinghaus ChatBot not initialized"
        print(f"  - ebbinghaus mode chatbot")
        
        # Cleanup
        evaluator.cleanup_chatbots()
        
        print("✅ Both memory modes tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ChatBot initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_population():
    """Test memory population with a sample conversation."""
    print("\n🧪 Testing Memory Population...")
    
    try:
        config = EvaluationConfig(
            max_conversations=1,
            memory_modes=["standard"],  # Test one mode to avoid conflicts
            local_model_path="./models/TinyLlama-1.1B-Chat-v1.0"  # Use smaller test model
        )
        evaluator = MemoryEvaluator(config)
        
        # Initialize chatbots
        evaluator.initialize_chatbots()
        
        # Load a sample conversation
        conversations = evaluator.dataset_loader.load_conversations("locomo10_sample.json")
        if not conversations:
            print("❌ No conversations loaded")
            return False
        
        sample_conversation = conversations[0]
        print(f"Using conversation: {sample_conversation.conversation_id}")
        print(f"Messages: {len(sample_conversation.messages)}")
        print(f"Questions: {len(sample_conversation.questions)}")
        
        # Test memory population with standard mode
        standard_chatbot = evaluator.chatbots["standard"]
        population_stats = evaluator.populate_chatbot_memory(standard_chatbot, sample_conversation)
        
        print(f"Population results:")
        print(f"  Messages processed: {population_stats['messages_processed']}")
        print(f"  Messages added: {population_stats['messages_added']}")
        print(f"  Population time: {population_stats['population_time']:.2f}s")
        print(f"  User ID: {population_stats['user_id']}")
        
        # Test memory retrieval
        if population_stats['messages_added'] > 0:
            test_query = "What did they talk about?"
            memories = standard_chatbot.get_memories(test_query, user_id=population_stats['user_id'])
            print(f"  Retrieved {len(memories)} memories for query: '{test_query}'")
            
            if memories:
                print(f"  First memory: {str(memories[0])[:100]}...")
        
        # Cleanup
        evaluator.cleanup_chatbots()
        
        print("✅ Memory population test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Memory population test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_question_evaluation():
    """Test evaluation of a single question."""
    print("\n🧪 Testing Question Evaluation...")
    
    try:
        config = EvaluationConfig(
            max_conversations=1, 
            use_llm_judge=False,  # Disable judge for faster testing
            memory_modes=["standard"],  # Test one mode to avoid conflicts
            local_model_path="./models/TinyLlama-1.1B-Chat-v1.0"  # Use smaller test model
        )
        evaluator = MemoryEvaluator(config)
        
        # Initialize chatbots
        evaluator.initialize_chatbots()
        
        # Load a sample conversation
        conversations = evaluator.dataset_loader.load_conversations("locomo10_sample.json")
        sample_conversation = conversations[0]
        
        # Populate memory
        standard_chatbot = evaluator.chatbots["standard"]
        population_stats = evaluator.populate_chatbot_memory(standard_chatbot, sample_conversation)
        
        # Test question evaluation
        if sample_conversation.questions:
            first_question = sample_conversation.questions[0]
            print(f"Evaluating question: {first_question.question[:60]}...")
            print(f"Ground truth: {first_question.answer[:60]}...")
            
            result = evaluator.evaluate_question_with_chatbot(
                standard_chatbot, 
                sample_conversation, 
                first_question,
                population_stats['user_id'],
                "standard"
            )
            
            if result:
                print(f"Evaluation result:")
                print(f"  Generated answer: {result.generated_answer[:100]}...")
                print(f"  F1 Score: {result.f1_score:.3f}")
                print(f"  BLEU-1 Score: {result.bleu_1_score:.3f}")
                print(f"  Generation time: {result.generation_time:.2f}s")
                print(f"  Memory search time: {result.memory_search_time:.3f}s")
                print(f"  Memory count: {result.metadata.get('memory_count', 'N/A')}")
            else:
                print("❌ Question evaluation returned None")
                return False
        else:
            print("❌ No questions found in conversation")
            return False
        
        # Cleanup
        evaluator.cleanup_chatbots()
        
        print("✅ Question evaluation test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Question evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_evaluation():
    """Test complete evaluation of a single conversation."""
    print("\n🧪 Testing Complete Conversation Evaluation...")
    
    try:
        config = EvaluationConfig(
            max_conversations=1, 
            use_llm_judge=False,  # Disable judge for faster testing
            memory_modes=["standard"],  # Test one mode to avoid conflicts
            local_model_path="./models/TinyLlama-1.1B-Chat-v1.0"  # Use smaller test model
        )
        evaluator = MemoryEvaluator(config)
        
        # Initialize chatbots
        evaluator.initialize_chatbots()
        
        # Load a sample conversation
        conversations = evaluator.dataset_loader.load_conversations("locomo10_sample.json")
        sample_conversation = conversations[0]
        
        # Evaluate conversation with standard memory
        summary = evaluator.evaluate_conversation(sample_conversation, "standard")
        
        print(f"Conversation evaluation summary:")
        print(f"  Conversation ID: {summary.conversation_id}")
        print(f"  Memory mode: {summary.memory_mode}")
        print(f"  Total questions: {summary.total_questions}")  
        print(f"  Successful evaluations: {summary.successful_evaluations}")
        print(f"  Failed evaluations: {summary.failed_evaluations}")
        print(f"  Average F1 score: {summary.avg_f1_score:.3f}")
        print(f"  Average BLEU-1 score: {summary.avg_bleu_1_score:.3f}")
        print(f"  Average generation time: {summary.avg_generation_time:.2f}s")
        print(f"  Individual results: {len(summary.results)}")
        
        # Show a sample result
        if summary.results:
            first_result = summary.results[0]
            print(f"  Sample result:")
            print(f"    Question: {first_result.question[:50]}...")
            print(f"    Answer: {first_result.generated_answer[:50]}...")
            print(f"    F1: {first_result.f1_score:.3f}")
        
        # Cleanup
        evaluator.cleanup_chatbots()
        
        print("✅ Conversation evaluation test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Conversation evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_pipeline():
    """Test the complete evaluation pipeline with minimal data."""
    print("\n🧪 Testing Complete Evaluation Pipeline...")
    
    try:
        # Use very limited config for testing
        config = EvaluationConfig(
            max_conversations=1,  # Just one conversation
            use_llm_judge=False,  # Disable judge for speed
            memory_modes=["standard"],  # Just one mode for testing
            local_model_path="./models/TinyLlama-1.1B-Chat-v1.0"  # Use smaller test model
        )
        evaluator = MemoryEvaluator(config)
        
        # Run the evaluation
        results = evaluator.run_evaluation("locomo10_sample.json")
        
        print(f"Pipeline results:")
        print(f"  Memory modes evaluated: {list(results.keys())}")
        
        for mode, summaries in results.items():
            print(f"  {mode} mode:")
            print(f"    Conversations processed: {len(summaries)}")
            if summaries:
                total_questions = sum(s.total_questions for s in summaries)
                successful = sum(s.successful_evaluations for s in summaries)
                avg_f1 = sum(s.avg_f1_score for s in summaries) / len(summaries)
                print(f"    Total questions: {total_questions}")
                print(f"    Successful evaluations: {successful}")
                print(f"    Average F1: {avg_f1:.3f}")
        
        # Test saving results
        output_path = evaluator.save_results(results)
        print(f"  Results saved to: {output_path}")
        
        # Verify file exists
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  File size: {file_size} bytes")
        
        # Cleanup
        evaluator.cleanup_chatbots()
        
        print("✅ Complete evaluation pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation pipeline test failed: {e}")
        return False

def test_llm_judge():
    """Test LLM judge evaluation using GPT-4o-mini."""
    print("\n🧪 Testing LLM Judge Evaluation...")
    
    try:
        config = EvaluationConfig(
            max_conversations=1,
            use_llm_judge=True,  # Enable LLM judge
            memory_modes=["standard"],
            local_model_path="./models/TinyLlama-1.1B-Chat-v1.0"  # Use smaller test model
        )
        evaluator = MemoryEvaluator(config)
        
        # Initialize chatbots
        evaluator.initialize_chatbots()
        
        # Load a sample conversation
        conversations = evaluator.dataset_loader.load_conversations("locomo10_sample.json")
        sample_conversation = conversations[0]
        
        # Populate memory
        standard_chatbot = evaluator.chatbots["standard"]
        population_stats = evaluator.populate_chatbot_memory(standard_chatbot, sample_conversation)
        
        # Test question evaluation with LLM judge
        if sample_conversation.questions:
            first_question = sample_conversation.questions[0]
            print(f"Evaluating question with LLM judge: {first_question.question[:60]}...")
            print(f"Ground truth: {first_question.answer[:60]}...")
            
            result = evaluator.evaluate_question_with_chatbot(
                standard_chatbot, 
                sample_conversation, 
                first_question,
                population_stats['user_id'],
                "standard"
            )
            
            if result:
                print(f"Evaluation result:")
                print(f"  Generated answer: {result.generated_answer[:100]}...")
                print(f"  F1 Score: {result.f1_score:.3f}")
                print(f"  BLEU-1 Score: {result.bleu_1_score:.3f}")
                print(f"  LLM Judge Score: {result.llm_judge_score:.3f}")
                print(f"  Generation time: {result.generation_time:.2f}s")
                print(f"  Memory search time: {result.memory_search_time:.3f}s")
                print(f"  Memory count: {result.metadata.get('memory_count', 'N/A')}")
                
                # Verify LLM judge was actually used
                if hasattr(result, 'llm_judge_score') and result.llm_judge_score is not None:
                    print("✅ LLM judge successfully provided score!")
                else:
                    print("❌ LLM judge score is missing or None")
                    return False
                    
            else:
                print("❌ Question evaluation with LLM judge returned None")
                return False
        else:
            print("❌ No questions found in conversation")
            return False
        
        # Cleanup
        evaluator.cleanup_chatbots()
        
        print("✅ LLM judge test successful!")
        return True
        
    except Exception as e:
        print(f"❌ LLM judge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all memory evaluator tests."""
    print("🚀 Testing memory_evaluator.py components...\n")
    
    tests = [
        test_chatbot_initialization,
        test_memory_population,
        test_question_evaluation,
        test_conversation_evaluation,
        test_evaluation_pipeline,
        test_llm_judge
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All memory evaluator tests passed! Ready to implement File 4.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Fix issues before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    main()
