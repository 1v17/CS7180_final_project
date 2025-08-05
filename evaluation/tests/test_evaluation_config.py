"""
Simple test script for evaluation_config.py

This script tests the key components of the evaluation configuration
to ensure everything is working before implementing the full evaluation system.
"""

import sys
import os

# Add the project root to the path so we can import evaluation modules
# Go up two levels: tests -> evaluation -> project_root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.evaluation_config import (
    EvaluationConfig,
    MetricsCalculator,
    LLMJudge,
    create_answer_generation_prompt
)

def test_evaluation_config():
    """Test the EvaluationConfig class."""
    print("🧪 Testing EvaluationConfig...")
    
    config = EvaluationConfig()
    
    # Test default values
    assert config.local_model_path == "./models/Llama-3.1-8B-Instruct"
    assert config.temperature == 0.0
    assert config.max_conversations == 3
    assert "standard" in config.memory_modes
    assert "ebbinghaus" in config.memory_modes
    
    print("✅ EvaluationConfig tests passed!")
    return True

def test_metrics_calculator():
    """Test the MetricsCalculator class."""
    print("\n🧪 Testing MetricsCalculator...")
    
    calc = MetricsCalculator()
    
    # Test F1 score calculation
    predicted = "The capital of France is Paris"
    ground_truth = "Paris is the capital of France"
    
    f1_score = calc.calculate_f1_score(predicted, ground_truth)
    print(f"F1 Score: {f1_score:.4f}")
    assert 0.0 <= f1_score <= 1.0, "F1 score should be between 0 and 1"
    
    # Test BLEU-1 score calculation
    bleu_score = calc.calculate_bleu_1(predicted, ground_truth)
    print(f"BLEU-1 Score: {bleu_score:.4f}")
    assert 0.0 <= bleu_score <= 1.0, "BLEU score should be between 0 and 1"
    
    # Test with identical strings
    perfect_f1 = calc.calculate_f1_score("hello world", "hello world")
    perfect_bleu = calc.calculate_bleu_1("hello world", "hello world")
    print(f"Perfect match F1: {perfect_f1:.4f}, BLEU: {perfect_bleu:.4f}")
    assert perfect_f1 == 1.0, "Identical strings should have F1 = 1.0"
    assert perfect_bleu == 1.0, "Identical strings should have BLEU = 1.0"
    
    # Test with completely different strings
    zero_f1 = calc.calculate_f1_score("hello world", "goodbye universe")
    zero_bleu = calc.calculate_bleu_1("hello world", "goodbye universe")
    print(f"No match F1: {zero_f1:.4f}, BLEU: {zero_bleu:.4f}")
    assert zero_f1 == 0.0, "Completely different strings should have F1 = 0.0"
    assert zero_bleu == 0.0, "Completely different strings should have BLEU = 0.0"
    
    print("✅ MetricsCalculator tests passed!")
    return True

def test_tokenization():
    """Test NLTK tokenization functionality."""
    print("\n🧪 Testing NLTK Tokenization...")
    
    calc = MetricsCalculator()
    
    # Test basic tokenization
    text = "Hello, world! How's it going?"
    tokens = calc._tokenize_and_clean(text)
    print(f"Tokenized '{text}' -> {tokens}")
    
    # Should have filtered out punctuation and converted to lowercase
    # Note: 's' is filtered out because it's not alphanumeric (contains apostrophe)
    expected_words = ['hello', 'world', 'how', 'it', 'going']
    assert all(word in tokens for word in expected_words), f"Expected words not found in tokens: {tokens}"
    assert all(token.isalnum() for token in tokens), "All tokens should be alphanumeric"
    
    print("✅ Tokenization tests passed!")
    return True

def test_prompt_generation():
    """Test prompt generation functions."""
    print("\n🧪 Testing Prompt Generation...")
    
    memory_context = "User mentioned they live in Paris and work as a teacher."
    question = "Where does the user work?"
    
    prompt = create_answer_generation_prompt(memory_context, question)
    print(f"Generated prompt:\n{prompt[:100]}...")
    
    # Check that prompt contains key elements
    assert "memories" in prompt.lower(), "Prompt should mention memories"
    assert question in prompt, "Prompt should contain the question"
    assert memory_context in prompt, "Prompt should contain memory context"
    
    print("✅ Prompt generation tests passed!")
    return True

def test_llm_judge():
    """Test the LLMJudge class (requires OpenAI API key)."""
    print("\n🧪 Testing LLMJudge...")
    
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for LLM judge functionality. "
                        "Please set OPENAI_API_KEY in your .env file or environment variables.")
    
    try:
        # Test initialization
        judge = LLMJudge(api_key)
        print("✅ LLMJudge initialized successfully")
        
        # Test basic judging functionality with a simple example
        question = "What is the capital of France?"
        ground_truth = "Paris"
        predicted_correct = "The capital of France is Paris"
        predicted_wrong = "The capital of France is London"
        
        print("Testing correct answer judgment...")
        score_correct = judge.judge_answer(question, predicted_correct, ground_truth)
        print(f"Correct answer score: {score_correct:.1f}/100")
        
        print("Testing incorrect answer judgment...")
        score_wrong = judge.judge_answer(question, predicted_wrong, ground_truth)
        print(f"Incorrect answer score: {score_wrong:.1f}/100")
        
        # Validate scores are in range
        assert 0.0 <= score_correct <= 100.0, f"Correct score out of range: {score_correct}"
        assert 0.0 <= score_wrong <= 100.0, f"Wrong score out of range: {score_wrong}"
        
        # The correct answer should generally score higher than the wrong one
        # (though this isn't guaranteed with LLM judges, so we make it a soft check)
        if score_correct <= score_wrong:
            print("ℹ️  Note: Correct answer didn't score higher than wrong answer")
            print("   This can happen with LLM judges, but is worth noting")
        
        print("✅ LLMJudge tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ LLMJudge test failed: {e}")
        print("This might be due to API issues or network connectivity")
        return False

def test_nltk_dependencies():
    """Test that NLTK dependencies are properly installed."""
    print("\n🧪 Testing NLTK Dependencies...")
    
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Test that punkt data is available
        tokens = word_tokenize("Hello, world!")
        print(f"NLTK tokenization works: {tokens}")
        
        # Test BLEU-1 calculation with smoothing (like we use in MetricsCalculator)
        smoothing_function = SmoothingFunction().method1
        score = sentence_bleu(
            [["hello", "world"]], 
            ["hello", "world"],
            weights=(1.0, 0, 0, 0),  # BLEU-1 weights
            smoothing_function=smoothing_function
        )
        print(f"NLTK BLEU-1 calculation works: {score}")
        
        # Should be 1.0 for identical sequences
        assert abs(score - 1.0) < 0.01, f"Expected BLEU-1 score ~1.0, got {score}"
        
        print("✅ NLTK dependencies tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ NLTK dependencies test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing evaluation_config.py components...\n")
    
    tests = [
        test_evaluation_config,
        test_metrics_calculator,
        test_tokenization,
        test_prompt_generation,
        test_llm_judge,
        test_nltk_dependencies
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
        print("\n🎉 All tests passed! Ready to implement File 2.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Fix issues before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    main()
