"""
LOCOMO Dataset Loader

This module loads and standardizes the LOCOMO dataset format for evaluation.
It handles various JSON structures and normalizes conversations, messages, and questions.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class ConversationMessage:
    """Standardized conversation message format."""
    speaker: str
    text: str
    dia_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class EvaluationQuestion:
    """Standardized evaluation question format."""
    question: str
    answer: str
    evidence: List[str]
    category: Optional[int] = None


@dataclass
class StandardizedConversation:
    """Standardized conversation format for evaluation."""
    conversation_id: str
    speaker_a: str
    speaker_b: str
    messages: List[ConversationMessage]
    questions: List[EvaluationQuestion]
    metadata: Dict[str, Any]


class LOCOMODatasetLoader:
    """Loads and standardizes LOCOMO dataset format."""
    
    def __init__(self, dataset_dir: str = "./resources/dataset/"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_dir (str): Directory containing LOCOMO JSON files
        """
        self.dataset_dir = dataset_dir
        self.logger = logging.getLogger(__name__)
        
    def load_conversations(self, filename: str = "locomo10_sample.json") -> List[StandardizedConversation]:
        """
        Load conversations from LOCOMO dataset file.
        
        Args:
            filename (str): Name of the JSON file to load
            
        Returns:
            List[StandardizedConversation]: List of standardized conversation objects
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If the JSON structure is invalid
        """
        file_path = os.path.join(self.dataset_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            self.logger.info(f"Loaded {len(raw_data)} conversations from {filename}")
            
            standardized_conversations = []
            for i, conversation_data in enumerate(raw_data):
                try:
                    standardized = self._standardize_conversation(conversation_data, i)
                    standardized_conversations.append(standardized)
                except Exception as e:
                    self.logger.warning(f"Failed to standardize conversation {i}: {e}")
                    continue
            
            self.logger.info(f"Successfully standardized {len(standardized_conversations)} conversations")
            return standardized_conversations
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {filename}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading dataset from {filename}: {e}")
    
    def _standardize_conversation(self, conversation_data: Dict[str, Any], conv_index: int) -> StandardizedConversation:
        """
        Standardize a single conversation from LOCOMO format.
        
        Args:
            conversation_data (Dict): Raw conversation data from LOCOMO
            conv_index (int): Index of the conversation for ID generation
            
        Returns:
            StandardizedConversation: Standardized conversation object
            
        Raises:
            ValueError: If required fields are missing
        """
        try:
            # Extract basic conversation info
            conversation_info = conversation_data.get('conversation', {})
            speaker_a = conversation_info.get('speaker_a', 'Speaker_A')
            speaker_b = conversation_info.get('speaker_b', 'Speaker_B')
            
            # Generate conversation ID
            conversation_id = f"locomo_conv_{conv_index:03d}"
            
            # Extract messages from all sessions
            messages = self._extract_messages(conversation_data)
            
            # Extract questions
            questions = self._extract_questions(conversation_data)
            
            # Extract metadata
            metadata = self._extract_metadata(conversation_data)
            
            return StandardizedConversation(
                conversation_id=conversation_id,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                messages=messages,
                questions=questions,
                metadata=metadata
            )
            
        except Exception as e:
            raise ValueError(f"Failed to standardize conversation {conv_index}: {e}")
    
    def _extract_messages(self, conversation_data: Dict[str, Any]) -> List[ConversationMessage]:
        """
        Extract and standardize messages from conversation data.
        
        Args:
            conversation_data (Dict): Raw conversation data
            
        Returns:
            List[ConversationMessage]: List of standardized messages
        """
        messages = []
        conversation_info = conversation_data.get('conversation', {})
        
        # Extract messages from all sessions
        session_keys = [key for key in conversation_info.keys() if key.startswith('session_') and not key.endswith('_date_time')]
        
        for session_key in sorted(session_keys):
            session_messages = conversation_info.get(session_key, [])
            session_datetime = conversation_info.get(f"{session_key}_date_time", "")
            
            for msg_data in session_messages:
                if isinstance(msg_data, dict):
                    message = ConversationMessage(
                        speaker=msg_data.get('speaker', 'Unknown'),
                        text=msg_data.get('text', ''),
                        dia_id=msg_data.get('dia_id'),
                        session_id=session_key,
                        timestamp=session_datetime
                    )
                    messages.append(message)
        
        # If no session-based messages, try direct message extraction
        if not messages and 'messages' in conversation_data:
            for msg_data in conversation_data['messages']:
                if isinstance(msg_data, dict):
                    message = ConversationMessage(
                        speaker=msg_data.get('speaker', 'Unknown'),
                        text=msg_data.get('text', msg_data.get('content', '')),
                        dia_id=msg_data.get('dia_id', msg_data.get('id')),
                        session_id=msg_data.get('session_id'),
                        timestamp=msg_data.get('timestamp')
                    )
                    messages.append(message)
        
        return messages
    
    def _extract_questions(self, conversation_data: Dict[str, Any]) -> List[EvaluationQuestion]:
        """
        Extract and standardize questions from conversation data.
        
        Args:
            conversation_data (Dict): Raw conversation data
            
        Returns:
            List[EvaluationQuestion]: List of standardized questions
        """
        questions = []
        qa_data = conversation_data.get('qa', [])
        
        # Parse LOCOMO qa structure - handles adversarial_answer format
        for q_data in qa_data:
            if isinstance(q_data, dict):
                # Skip category-only entries
                if 'category' in q_data and len(q_data) == 1:
                    continue
                
                # Handle adversarial_answer entries (create synthetic questions)
                if 'adversarial_answer' in q_data:
                    question = EvaluationQuestion(
                        question=f"Question for answer: {q_data['adversarial_answer'][:50]}...",  # Synthetic question
                        answer=q_data['adversarial_answer'],
                        evidence=[],
                        category=q_data.get('category')
                    )
                    questions.append(question)
                # Handle standard question-answer pairs
                elif 'question' in q_data or 'answer' in q_data:
                    question = EvaluationQuestion(
                        question=q_data.get('question', ''),
                        answer=q_data.get('answer', ''),
                        evidence=q_data.get('evidence', []),
                        category=q_data.get('category')
                    )
                    # Only add if both question and answer are present
                    if question.question.strip() and str(question.answer).strip():
                        questions.append(question)
        
        # Alternative structure: direct questions list
        if not questions and 'questions' in conversation_data:
            for q_data in conversation_data['questions']:
                if isinstance(q_data, dict):
                    question = EvaluationQuestion(
                        question=q_data.get('question', q_data.get('text', '')),
                        answer=q_data.get('answer', q_data.get('ground_truth', '')),
                        evidence=q_data.get('evidence', q_data.get('sources', [])),
                        category=q_data.get('category', q_data.get('type'))
                    )
                    # Only add if both question and answer are present
                    question_text = str(question.question).strip() if question.question is not None else ""
                    answer_text = str(question.answer).strip() if question.answer is not None else ""
                    if question_text and answer_text:
                        questions.append(question)
        
        return questions
    
    def _extract_metadata(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from conversation data.
        
        Args:
            conversation_data (Dict): Raw conversation data
            
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        metadata = {}
        
        # Extract event summaries
        if 'event_summary' in conversation_data:
            metadata['event_summary'] = conversation_data['event_summary']
        
        # Extract observations
        if 'observation' in conversation_data:
            metadata['observation'] = conversation_data['observation']
        
        # Extract any other top-level metadata
        exclude_keys = {'conversation', 'qa', 'questions', 'messages', 'event_summary', 'observation'}
        for key, value in conversation_data.items():
            if key not in exclude_keys:
                metadata[key] = value
        
        return metadata
    
    def get_dataset_statistics(self, conversations: List[StandardizedConversation]) -> Dict[str, Any]:
        """
        Generate statistics about the loaded dataset.
        
        Args:
            conversations (List[StandardizedConversation]): List of conversations
            
        Returns:
            Dict[str, Any]: Dataset statistics
        """
        if not conversations:
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "total_questions": 0,
                "average_messages_per_conversation": 0,
                "average_questions_per_conversation": 0,
                "speakers": []
            }
        
        total_messages = sum(len(conv.messages) for conv in conversations)
        total_questions = sum(len(conv.questions) for conv in conversations)
        
        # Collect unique speakers
        speakers = set()
        for conv in conversations:
            speakers.add(conv.speaker_a)
            speakers.add(conv.speaker_b)
            for message in conv.messages:
                speakers.add(message.speaker)
        
        # Question categories
        categories = {}
        for conv in conversations:
            for question in conv.questions:
                if question.category is not None:
                    categories[question.category] = categories.get(question.category, 0) + 1
        
        return {
            "total_conversations": len(conversations),
            "total_messages": total_messages,
            "total_questions": total_questions,
            "average_messages_per_conversation": total_messages / len(conversations),
            "average_questions_per_conversation": total_questions / len(conversations),
            "unique_speakers": sorted(list(speakers)),
            "question_categories": categories,
            "conversation_ids": [conv.conversation_id for conv in conversations]
        }
    
    def validate_conversation(self, conversation: StandardizedConversation) -> Tuple[bool, List[str]]:
        """
        Validate a standardized conversation.
        
        Args:
            conversation (StandardizedConversation): Conversation to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if not conversation.conversation_id:
            issues.append("Missing conversation_id")
        
        if not conversation.speaker_a:
            issues.append("Missing speaker_a")
            
        if not conversation.speaker_b:
            issues.append("Missing speaker_b")
        
        # Check messages
        if not conversation.messages:
            issues.append("No messages found")
        else:
            for i, message in enumerate(conversation.messages):
                if not message.speaker:
                    issues.append(f"Message {i}: Missing speaker")
                if not message.text.strip():
                    issues.append(f"Message {i}: Empty text")
        
        # Check questions
        if not conversation.questions:
            issues.append("No questions found")
        else:
            for i, question in enumerate(conversation.questions):
                # Convert to string and then check if empty after stripping
                question_text = str(question.question).strip() if question.question is not None else ""
                answer_text = str(question.answer).strip() if question.answer is not None else ""
                
                if not question_text:
                    issues.append(f"Question {i}: Empty question text")
                if not answer_text:
                    issues.append(f"Question {i}: Empty answer text")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def filter_conversations(self, conversations: List[StandardizedConversation], 
                           min_messages: int = 1, 
                           min_questions: int = 1) -> List[StandardizedConversation]:
        """
        Filter conversations based on criteria.
        
        Args:
            conversations (List[StandardizedConversation]): Conversations to filter
            min_messages (int): Minimum number of messages required
            min_questions (int): Minimum number of questions required
            
        Returns:
            List[StandardizedConversation]: Filtered conversations
        """
        filtered = []
        
        for conv in conversations:
            if len(conv.messages) >= min_messages and len(conv.questions) >= min_questions:
                is_valid, issues = self.validate_conversation(conv)
                if is_valid:
                    filtered.append(conv)
                else:
                    self.logger.warning(f"Skipping invalid conversation {conv.conversation_id}: {issues}")
        
        self.logger.info(f"Filtered {len(conversations)} -> {len(filtered)} conversations")
        return filtered
