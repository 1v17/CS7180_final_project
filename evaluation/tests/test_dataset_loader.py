"""
Test script for locomo_dataset_loader.py

This script tests the LOCOMO dataset loading functionality
to ensure it correctly parses and standardizes the dataset format.
"""

import sys
import os

# Add the project root to the path so we can import evaluation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.locomo_dataset_loader import LOCOMODatasetLoader

def test_dataset_loading():
    """Test basic dataset loading functionality."""
    print("🧪 Testing LOCOMO Dataset Loading...")
    
    loader = LOCOMODatasetLoader()
    
    try:
        conversations = loader.load_conversations("locomo10_sample.json")
        print(f"✅ Successfully loaded {len(conversations)} conversations")
        
        # Test first conversation
        if conversations:
            first_conv = conversations[0]
            print(f"First conversation ID: {first_conv.conversation_id}")
            print(f"Speakers: {first_conv.speaker_a} and {first_conv.speaker_b}")
            print(f"Messages: {len(first_conv.messages)}")
            print(f"Questions: {len(first_conv.questions)}")
            
            # Show first message
            if first_conv.messages:
                first_msg = first_conv.messages[0]
                print(f"First message: {first_msg.speaker}: {first_msg.text[:50]}...")
            
            # Show first question
            if first_conv.questions:
                first_q = first_conv.questions[0]
                print(f"First question: {first_q.question[:50]}...")
                print(f"Answer: {first_q.answer[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_dataset_statistics():
    """Test dataset statistics generation."""
    print("\n🧪 Testing Dataset Statistics...")
    
    loader = LOCOMODatasetLoader()
    
    try:
        conversations = loader.load_conversations("locomo10_sample.json")
        stats = loader.get_dataset_statistics(conversations)
        
        print("📊 Dataset Statistics:")
        print(f"  Total conversations: {stats['total_conversations']}")
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Avg messages per conversation: {stats['average_messages_per_conversation']:.1f}")
        print(f"  Avg questions per conversation: {stats['average_questions_per_conversation']:.1f}")
        print(f"  Unique speakers: {stats['unique_speakers']}")
        
        if stats['question_categories']:
            print(f"  Question categories: {stats['question_categories']}")
        
        print("✅ Statistics generation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Statistics generation failed: {e}")
        return False

def test_conversation_validation():
    """Test conversation validation functionality."""
    print("\n🧪 Testing Conversation Validation...")
    
    loader = LOCOMODatasetLoader()
    
    try:
        conversations = loader.load_conversations("locomo10_sample.json")
        
        valid_count = 0
        invalid_count = 0
        
        for conv in conversations:
            is_valid, issues = loader.validate_conversation(conv)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                print(f"  Invalid conversation {conv.conversation_id}: {issues}")
        
        print(f"✅ Validation results: {valid_count} valid, {invalid_count} invalid")
        return True
        
    except Exception as e:
        print(f"❌ Validation testing failed: {e}")
        return False

def test_conversation_filtering():
    """Test conversation filtering functionality."""
    print("\n🧪 Testing Conversation Filtering...")
    
    loader = LOCOMODatasetLoader()
    
    try:
        conversations = loader.load_conversations("locomo10_sample.json")
        print(f"Original conversations: {len(conversations)}")
        
        # Filter with minimum requirements
        filtered = loader.filter_conversations(
            conversations, 
            min_messages=5, 
            min_questions=3
        )
        print(f"Filtered conversations (min 5 msgs, 3 qs): {len(filtered)}")
        
        # More strict filtering
        strict_filtered = loader.filter_conversations(
            conversations,
            min_messages=10,
            min_questions=5
        )
        print(f"Strict filtered conversations (min 10 msgs, 5 qs): {len(strict_filtered)}")
        
        print("✅ Filtering test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Filtering test failed: {e}")
        return False

def test_detailed_conversation_structure():
    """Test detailed conversation structure analysis."""
    print("\n🧪 Testing Detailed Conversation Structure...")
    
    loader = LOCOMODatasetLoader()
    
    try:
        conversations = loader.load_conversations("locomo10_sample.json")
        
        if conversations:
            conv = conversations[0]
            print(f"\n📋 Analyzing conversation: {conv.conversation_id}")
            
            # Analyze messages
            print(f"\n💬 Messages ({len(conv.messages)}):")
            for i, msg in enumerate(conv.messages[:3]):  # Show first 3
                print(f"  {i+1}. {msg.speaker} ({msg.dia_id}): {msg.text[:60]}...")
            
            # Analyze questions
            print(f"\n❓ Questions ({len(conv.questions)}):")
            for i, q in enumerate(conv.questions[:3]):  # Show first 3
                print(f"  {i+1}. Q: {q.question[:60]}...")
                print(f"      A: {q.answer[:60]}...")
                print(f"      Evidence: {q.evidence}")
                print(f"      Category: {q.category}")
            
            # Analyze metadata
            print(f"\n📄 Metadata keys: {list(conv.metadata.keys())}")
        
        print("✅ Detailed structure analysis successful!")
        return True
        
    except Exception as e:
        print(f"❌ Detailed structure analysis failed: {e}")
        return False

def main():
    """Run all dataset loader tests."""
    print("🚀 Testing locomo_dataset_loader.py components...\n")
    
    tests = [
        test_dataset_loading,
        test_dataset_statistics,
        test_conversation_validation,
        test_conversation_filtering,
        test_detailed_conversation_structure
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
        print("\n🎉 All dataset loader tests passed! Ready to implement File 3.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Fix issues before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    main()
