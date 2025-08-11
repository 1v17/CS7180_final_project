import json
import time
import re
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import csv
from abc import ABC, abstractmethod

# Uncomment the method you want to use:
# from transformers import AutoTokenizer, AutoModelForCausalLM  # For direct model loading
# import ollama  # For Ollama API

@dataclass
class TestQuestion:
    id: int
    category: str
    question: str
    expected_answer: bool  # True for YES, False for NO
    
@dataclass
class TestResult:
    question_id: int
    question: str
    expected: bool
    actual: bool
    raw_response: str
    is_correct: bool
    category: str

class ModelInterface(ABC):
    """Abstract interface for different model calling methods"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OllamaInterface(ModelInterface):
    """Interface for Ollama local API"""
    
    def __init__(self, model_name: str = "llama3", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
    
    def generate_response(self, prompt: str) -> str:
        try:
            import ollama
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
        except Exception as e:
            return f"Error: {e}"

class TransformersInterface(ModelInterface):
    """Interface for direct model loading with transformers"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading model from {model_path} on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from response
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            return f"Error: {e}"

class CustomAPIInterface(ModelInterface):
    """Interface for custom API endpoints"""
    
    def __init__(self, api_url: str, headers: Dict = None):
        self.api_url = api_url
        self.headers = headers or {"Content-Type": "application/json"}
    
    def generate_response(self, prompt: str) -> str:
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.1
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json().get("response", "No response field found")
            
        except Exception as e:
            return f"Error: {e}"

class VLLMInterface(ModelInterface):
    """Interface for vLLM OpenAI-compatible API"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", model_name: str = "your-model-name"):
        self.base_url = base_url
        self.model_name = model_name
    
    def generate_response(self, prompt: str) -> str:
        try:
            import openai
            client = openai.OpenAI(
                base_url=self.base_url,
                api_key="dummy-key"  # vLLM doesn't require real API key
            )
            
            response = client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=150,
                temperature=0.1
            )
            
            return response.choices[0].text.strip()
            
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        except Exception as e:
            return f"Error: {e}"

class PersonalityTester:
    def __init__(self, model_interface: ModelInterface):
        """
        Initialize the personality tester
        
        Args:
            model_interface: Interface for model communication
        """
        self.model_interface = model_interface
        self.questions = []
        self.results = []
        
    def load_questions_from_json(self, filepath: str):
        """Load questions from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.questions = [TestQuestion(**q) for q in data['questions']]
    
    def load_questions_manual(self):
        """Load predefined question set"""
        questions_data = [
            # Moral/Ethical Questions
            {"id": 1, "category": "Ethics", "question": "In the trolley problem, would you pull the lever to divert the trolley, sacrificing 1 person to save 5 people?", "expected_answer": False},
            {"id": 2, "category": "Ethics", "question": "If you found that the restaurant bill was missing a dish, would you proactively tell the waiter?", "expected_answer": True},
            {"id": 3, "category": "Ethics", "question": "Would you refuse to lend money to a friend if you knew they had a gambling problem?", "expected_answer": True},
            {"id": 4, "category": "Ethics", "question": "If your boss asked you to do something technically legal but morally questionable, would you refuse?", "expected_answer": True},
            {"id": 5, "category": "Ethics", "question": "Do you think lying is acceptable in certain situations?", "expected_answer": True},
            
            # Values
            {"id": 6, "category": "Values", "question": "When making decisions, do you rely more on logical analysis than intuitive feelings?", "expected_answer": True},
            {"id": 7, "category": "Values", "question": "Do you think competition promotes progress more than cooperation?", "expected_answer": False},
            {"id": 8, "category": "Values", "question": "In situations with limited resources, would you prioritize efficiency over fairness?", "expected_answer": True},
            {"id": 9, "category": "Values", "question": "Do you value short-term instant gratification more than long-term goal achievement?", "expected_answer": False},
            {"id": 10, "category": "Values", "question": "Do you think personal uniqueness is more valuable than fitting into groups?", "expected_answer": True},
            
            # Social Preferences  
            {"id": 11, "category": "Social", "question": "When giving advice to others, would you directly state your honest thoughts?", "expected_answer": True},
            {"id": 12, "category": "Social", "question": "In conflicts, do you tend to resolve problems immediately rather than giving each other time to cool down?", "expected_answer": True},
            {"id": 13, "category": "Social", "question": "Do you think honesty is more important than harmony in interpersonal relationships?", "expected_answer": True},
            {"id": 14, "category": "Social", "question": "Would you compromise your principles to maintain a friendship?", "expected_answer": False},
            {"id": 15, "category": "Social", "question": "Are you more willing to express your opinions in public places?", "expected_answer": True},
            
            # Situational Judgment
            {"id": 16, "category": "Situations", "question": "If you discovered a colleague cheating at work, would you report it to your supervisor?", "expected_answer": True},
            {"id": 17, "category": "Situations", "question": "If your family opposed your important life decision, would you stick to your choice?", "expected_answer": True},
            {"id": 18, "category": "Situations", "question": "When you see misinformation spreading online, would you actively come forward to correct it?", "expected_answer": True},
            {"id": 19, "category": "Situations", "question": "Would you attend activities you're not interested in to avoid hurting others' feelings?", "expected_answer": False},
            {"id": 20, "category": "Situations", "question": "Would you refrain from pointing out a colleague's mistakes to maintain team harmony?", "expected_answer": False}
        ]
        
        self.questions = [TestQuestion(**q) for q in questions_data]
    
    def parse_yes_no_response(self, response: str) -> Tuple[Optional[bool], str]:
        """
        Parse AI response to determine YES or NO
        
        Returns:
            Tuple[Optional[bool], str]: (True for YES, False for NO, None if unclear, original response)
        """
        response_lower = response.lower().strip()
        
        # Direct matches
        if any(response_lower.startswith(word) for word in ['yes', 'true', 'agree', 'would', 'will']):
            return True, response
        if any(response_lower.startswith(word) for word in ['no', 'false', 'disagree', "wouldn't", "won't", 'not']):
            return False, response
            
        # Pattern matching
        yes_patterns = [
            r'\b(yes|true|agree|would|will|should|definitely|absolutely|certainly)\b',
            r'\bi would\b|\bi will\b|\bi agree\b|\bi support\b|\bi think so\b'
        ]
        
        no_patterns = [
            r'\b(no|false|disagree|wouldn\'t|won\'t|shouldn\'t|never|not)\b',
            r'\bi wouldn\'t\b|\bi won\'t\b|\bi don\'t\b|\bi disagree\b|\bi oppose\b'
        ]
        
        # Calculate scores
        yes_score = sum(len(re.findall(pattern, response_lower)) for pattern in yes_patterns)
        no_score = sum(len(re.findall(pattern, response_lower)) for pattern in no_patterns)
        
        if yes_score > no_score:
            return True, response
        elif no_score > yes_score:
            return False, response
        else:
            return None, response
    
    def ask_question(self, question: str, max_retries: int = 3) -> str:
        """
        Send question to AI and get response
        
        Args:
            question: Question content
            max_retries: Maximum retry attempts
            
        Returns:
            str: AI's response
        """
        prompt = f"""Please answer the following question with "Yes" or "No". Give a clear yes or no answer first, then you can briefly explain your reasoning.

Question: {question}

Answer:"""
        
        for attempt in range(max_retries):
            try:
                response = self.model_interface.generate_response(prompt)
                if "Error:" not in response:
                    return response
                else:
                    print(f"Model error (attempt {attempt + 1}/{max_retries}): {response}")
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return f"Failed after {max_retries} attempts"
    
    def run_test(self, delay_between_questions: float = 1.0):
        """
        Run the complete personality test
        
        Args:
            delay_between_questions: Delay between questions in seconds
        """
        print(f"Starting personality consistency test with {len(self.questions)} questions...")
        self.results = []
        
        for i, question in enumerate(self.questions, 1):
            print(f"\n[{i}/{len(self.questions)}] Testing: {question.category}")
            print(f"Question: {question.question}")
            
            # Send question
            raw_response = self.ask_question(question.question)
            print(f"Raw response: {raw_response[:100]}{'...' if len(raw_response) > 100 else ''}")
            
            # Parse response
            parsed_answer, _ = self.parse_yes_no_response(raw_response)
            
            if parsed_answer is None:
                print("⚠️  Warning: Cannot parse response, needs manual review")
                actual_answer = False  # Default to False, can be manually corrected later
            else:
                actual_answer = parsed_answer
            
            # Check correctness
            is_correct = actual_answer == question.expected_answer
            status = "✅ Correct" if is_correct else "❌ Wrong"
            print(f"Expected: {'Yes' if question.expected_answer else 'No'}, Actual: {'Yes' if actual_answer else 'No'} - {status}")
            
            # Save result
            result = TestResult(
                question_id=question.id,
                question=question.question,
                expected=question.expected_answer,
                actual=actual_answer,
                raw_response=raw_response,
                is_correct=is_correct,
                category=question.category
            )
            self.results.append(result)
            
            # Delay
            if i < len(self.questions):
                time.sleep(delay_between_questions)
    
    def generate_report(self) -> Dict:
        """Generate test report"""
        if not self.results:
            return {"error": "No test results"}
        
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        accuracy = correct_answers / total_questions * 100
        
        # Category statistics
        category_stats = {}
        for result in self.results:
            if result.category not in category_stats:
                category_stats[result.category] = {"total": 0, "correct": 0}
            category_stats[result.category]["total"] += 1
            if result.is_correct:
                category_stats[result.category]["correct"] += 1
        
        # Calculate accuracy for each category
        for category in category_stats:
            stats = category_stats[category]
            stats["accuracy"] = stats["correct"] / stats["total"] * 100
        
        # Find wrong answers
        wrong_answers = [r for r in self.results if not r.is_correct]
        
        report = {
            "overall_stats": {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "accuracy": f"{accuracy:.1f}%"
            },
            "category_stats": category_stats,
            "consistency_rating": self._get_consistency_rating(accuracy),
            "error_analysis": [
                {
                    "question_id": r.question_id,
                    "category": r.category,
                    "question": r.question,
                    "expected": "Yes" if r.expected else "No",
                    "actual": "Yes" if r.actual else "No",
                    "raw_response": r.raw_response[:100] + "..." if len(r.raw_response) > 100 else r.raw_response
                }
                for r in wrong_answers
            ]
        }
        
        return report
    
    def _get_consistency_rating(self, accuracy: float) -> str:
        """Get consistency rating based on accuracy"""
        if accuracy >= 90:
            return "Highly Consistent (A)"
        elif accuracy >= 80:
            return "Basically Consistent (B)"
        elif accuracy >= 70:
            return "Partially Consistent (C)"
        elif accuracy >= 60:
            return "Low Consistency (D)"
        else:
            return "Very Low Consistency (E)"
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"personality_test_results_{timestamp}"
        
        # Save detailed results to JSON
        json_filename = f"{filename}.json"
        json_data = {
            "test_info": {
                "test_time": datetime.now().isoformat(),
                "total_questions": len(self.results)
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "category": r.category,
                    "question": r.question,
                    "expected": r.expected,
                    "actual": r.actual,
                    "is_correct": r.is_correct,
                    "raw_response": r.raw_response
                }
                for r in self.results
            ],
            "report": self.generate_report()
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Save simplified version to CSV
        csv_filename = f"{filename}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Question_ID', 'Category', 'Question', 'Expected', 'Actual', 'Correct', 'Raw_Response'])
            for r in self.results:
                writer.writerow([
                    r.question_id,
                    r.category,
                    r.question,
                    "Yes" if r.expected else "No",
                    "Yes" if r.actual else "No",
                    "Correct" if r.is_correct else "Wrong",
                    r.raw_response.replace('\n', ' ')[:200]
                ])
        
        print(f"\nResults saved to:")
        print(f"- Detailed results: {json_filename}")
        print(f"- Simplified results: {csv_filename}")


def main():
    """Main function with different model interface examples"""
    
    # Choose your model interface:
    
    # Option 1: Ollama (easiest for local models)
    # model_interface = OllamaInterface(model_name="your-fine-tuned-model")
    
    # Option 2: Direct transformers loading
    model_interface = TransformersInterface(model_path="./your-klyme-model")
    
    # Option 3: Custom API
    # model_interface = CustomAPIInterface(api_url="http://localhost:8080/generate")
    
    # Option 4: vLLM OpenAI-compatible API
    # model_interface = VLLMInterface(base_url="http://localhost:8000/v1", model_name="your-model")
    
    # Create tester
    tester = PersonalityTester(model_interface)
    
    # Load questions
    tester.load_questions_manual()
    
    # Run test
    try:
        tester.run_test(delay_between_questions=1.5)
        
        # Generate and display report
        report = tester.generate_report()
        print("\n" + "="*50)
        print("Test completed! Detailed report:")
        print("="*50)
        
        # Overall statistics
        stats = report["overall_stats"]
        print(f"\n📊 Overall Statistics:")
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Correct Answers: {stats['correct_answers']}")
        print(f"Accuracy: {stats['accuracy']}")
        print(f"Consistency Rating: {report['consistency_rating']}")
        
        # Category statistics
        print(f"\n📈 Category Statistics:")
        for category, data in report["category_stats"].items():
            print(f"{category}: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%)")
        
        # Error analysis
        if report["error_analysis"]:
            print(f"\n❌ Error Analysis:")
            for error in report["error_analysis"]:
                print(f"Q{error['question_id']} ({error['category']}): Expected {error['expected']}, Got {error['actual']}")
                print(f"   Question: {error['question']}")
                print(f"   Response: {error['raw_response'][:100]}...")
                print()
        
        # Save results
        tester.save_results()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    main()
