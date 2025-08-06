"""
Evaluation Configuration for LOCOMO Dataset Evaluation

This module provides configuration settings and utility classes for evaluating
the Ebbinghaus memory implementation against standard memory using the LOCOMO dataset.
"""

from dataclasses import dataclass
from typing import List, Dict
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize
import logging

JUDGE_MODEL = "gpt-4o-mini"  # Default model for LLM judge

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


@dataclass
class EvaluationConfig:
    """Configuration settings for the evaluation system."""
    
    # Model settings - matches your existing ChatBot setup
    local_model_path: str = "./models/Llama-3.1-8B-Instruct"
    use_existing_chatbot: bool = True
    temperature: float = 0.0
    max_conversations: int = 20  # There are in total 20 conversations in the LOCOMO dataset
    max_new_tokens: int = 100
    
    # Dataset settings
    dataset_path: str = "./resources/dataset/locomo10_sample.json"
    output_dir: str = "./evaluation/evaluation_output"
    
    # Evaluation settings
    judge_max_tokens: int = 10
    answer_max_tokens: int = 100
    use_llm_judge: bool = True
    use_traditional_metrics: bool = True
    
    # Memory mode settings
    memory_modes: List[str] = None
    config_modes: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.memory_modes is None:
            self.memory_modes = ["standard", "ebbinghaus"]
        
        if self.config_modes is None:
            self.config_modes = {
                "standard": "testing",
                "ebbinghaus": "testing"
            }


class MetricsCalculator:
    """Utility class for calculating evaluation metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.smoothing_function = SmoothingFunction().method1
        
    def calculate_f1_score(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate F1 score between predicted and ground truth answers.
        
        Args:
            predicted (str): Generated answer
            ground_truth (str): Ground truth answer
            
        Returns:
            float: F1 score between 0 and 1
        """
        try:
            # Tokenize and clean text
            predicted_tokens = self._tokenize_and_clean(predicted)
            ground_truth_tokens = self._tokenize_and_clean(ground_truth)
            
            if not predicted_tokens and not ground_truth_tokens:
                return 1.0  # Both empty
            elif not predicted_tokens or not ground_truth_tokens:
                return 0.0  # One empty, one not
            
            # Convert to sets for overlap calculation
            predicted_set = set(predicted_tokens)
            ground_truth_set = set(ground_truth_tokens)
            
            # Calculate precision, recall, and F1
            intersection = predicted_set.intersection(ground_truth_set)
            
            if len(predicted_set) == 0:
                precision = 0.0
            else:
                precision = len(intersection) / len(predicted_set)
                
            if len(ground_truth_set) == 0:
                recall = 0.0
            else:
                recall = len(intersection) / len(ground_truth_set)
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
            
        except Exception as e:
            logging.warning(f"Error calculating F1 score: {e}")
            return 0.0
    
    def calculate_bleu_1(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate BLEU-1 score between predicted and ground truth answers.
        
        Args:
            predicted (str): Generated answer
            ground_truth (str): Ground truth answer
            
        Returns:
            float: BLEU-1 score between 0 and 1
        """
        try:
            # Tokenize
            predicted_tokens = self._tokenize_and_clean(predicted)
            ground_truth_tokens = self._tokenize_and_clean(ground_truth)
            
            if not predicted_tokens:
                return 0.0
            
            # BLEU expects reference as list of tokens, hypothesis as list of tokens
            bleu_score = sentence_bleu(
                [ground_truth_tokens], 
                predicted_tokens,
                weights=(1.0, 0, 0, 0),  # BLEU-1 weights
                smoothing_function=self.smoothing_function
            )
            
            return bleu_score
            
        except Exception as e:
            logging.warning(f"Error calculating BLEU-1 score: {e}")
            return 0.0
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize and clean text for metric calculation using NLTK.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Cleaned tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase and normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip().lower())
        
        # Use NLTK tokenization (handles contractions and punctuation better)
        tokens = word_tokenize(text)
        
        # Filter out punctuation and keep only alphabetic tokens
        tokens = [token for token in tokens if token.isalnum()]
        
        return tokens


class LLMJudge:
    """LLM-as-a-judge implementation using OpenAI's GPT-4o-mini."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the judge with OpenAI API.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, will use OPENAI_API_KEY env var
        """
        import os
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = JUDGE_MODEL
    
    def judge_answer(self, question: str, predicted: str, ground_truth: str) -> float:
        """
        Evaluate a generated answer against ground truth using LLM-as-a-judge.
        
        Args:
            question (str): The original question
            predicted (str): Generated answer to evaluate
            ground_truth (str): Ground truth answer
            
        Returns:
            float: Score between 0 and 100
        """
        try:
            judge_prompt = self._create_judge_prompt(question, predicted, ground_truth)
            
            # Use OpenAI API to generate judgment
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond only with a numeric score between 1 and 100."},
                    {"role": "user", "content": judge_prompt}
                ],
                max_tokens=10,
                temperature=0.0
            )
            
            # Extract numeric score from response
            response_text = response.choices[0].message.content.strip()
            score = self._extract_score(response_text)
            return score
            
        except Exception as e:
            logging.warning(f"Error in LLM judge evaluation: {e}")
            return 0.0
    
    def _create_judge_prompt(self, question: str, predicted: str, ground_truth: str) -> str:
        """Create the judging prompt."""
        prompt = f"""Your task is to evaluate an answer to a question by comparing it with the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {predicted}

Please evaluate the generated answer on a scale of 1-100 based on:
1. Factual accuracy compared to the ground truth
2. Completeness of the answer
3. Relevance to the question

Consider the following:
- If the generated answer contains the same key information as the ground truth, it should score highly
- Minor differences in wording or format should not significantly impact the score
- If the answer is completely wrong or irrelevant, it should score very low
- If the answer is partially correct, give partial credit

Respond with only a number between 1 and 100."""
        
        return prompt
    
    def _extract_score(self, response: str) -> float:
        """
        Extract numeric score from LLM response.
        
        Args:
            response (str): LLM response containing score
            
        Returns:
            float: Extracted score, 0.0 if extraction fails
        """
        if not response:
            return 0.0
        
        # Look for numbers in the response
        numbers = re.findall(r'\b(\d{1,3})\b', response.strip())
        
        if numbers:
            try:
                score = float(numbers[0])
                # Ensure score is in valid range
                return max(0.0, min(100.0, score))
            except ValueError:
                pass
        
        # Fallback: look for percentage patterns
        percentage_match = re.search(r'(\d{1,3})%', response)
        if percentage_match:
            try:
                score = float(percentage_match.group(1))
                return max(0.0, min(100.0, score))
            except ValueError:
                pass
        
        logging.warning(f"Could not extract score from response: {response}")
        return 0.0


def create_answer_generation_prompt(memory_context: str, question: str) -> str:
    """
    Create prompt for answer generation using memory context.
    
    Args:
        memory_context (str): Retrieved memory context
        question (str): Question to answer
        
    Returns:
        str: Formatted prompt for answer generation
    """
    prompt = f"""Based on the following conversation memories, please answer the question.

Memories:
{memory_context}

Question: {question}

Instructions:
- Provide a concise, factual answer based only on the information in the memories
- If the memories don't contain enough information to answer the question, say "Information not available"
- Focus on the most relevant details
- Keep the answer brief and to the point

Answer:"""
    
    return prompt


def judge_answer_with_chatbot(api_key: str, question: str, predicted: str, ground_truth: str) -> float:
    """
    Standalone function to judge an answer using OpenAI's GPT-4o-mini.
    
    Args:
        api_key (str): OpenAI API key
        question (str): Original question
        predicted (str): Generated answer
        ground_truth (str): Ground truth answer
        
    Returns:
        float: Score between 0 and 100
    """
    judge = LLMJudge(api_key)
    return judge.judge_answer(question, predicted, ground_truth)
