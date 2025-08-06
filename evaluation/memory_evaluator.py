"""
Memory Evaluator for LOCOMO Dataset

This module contains the core evaluation logic that compares memory systems 
using the existing ChatBot class. It evaluates both standard and Ebbinghaus 
memory modes against the LOCOMO dataset.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os

# Import local modules
from evaluation.evaluation_config import EvaluationConfig, MetricsCalculator, LLMJudge, create_answer_generation_prompt
from evaluation.locomo_dataset_loader import LOCOMODatasetLoader, StandardizedConversation

# Import the existing ChatBot - will be imported dynamically to avoid circular imports


@dataclass
class EvaluationResult:
    """Single evaluation result for a question."""
    conversation_id: str
    question_id: int
    question: str
    ground_truth: str
    generated_answer: str
    
    # Traditional metrics
    f1_score: float
    bleu_1_score: float
    
    # LLM judge score
    llm_judge_score: float
    
    # Performance metrics
    generation_time: float
    memory_search_time: float
    
    # Memory mode
    memory_mode: str
    
    # Additional metadata
    metadata: Dict[str, Any]


@dataclass
class ConversationEvaluationSummary:
    """Summary of evaluation results for a single conversation."""
    conversation_id: str
    memory_mode: str
    total_questions: int
    successful_evaluations: int
    failed_evaluations: int
    
    # Aggregated metrics
    avg_f1_score: float
    avg_bleu_1_score: float
    avg_llm_judge_score: float
    avg_generation_time: float
    avg_memory_search_time: float
    
    # Individual results
    results: List[EvaluationResult]


class MemoryEvaluator:
    """Core evaluation logic comparing memory systems using ChatBot."""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the memory evaluator.
        
        Args:
            config (EvaluationConfig): Evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.dataset_loader = LOCOMODatasetLoader()
        self.metrics_calculator = MetricsCalculator()
        
        # ChatBot instances will be initialized when needed
        self.chatbots = {}
        self.judge_chatbot = None
        self.llm_judge = None
        
        # Results storage
        self.conversation_summaries = []
        
    def initialize_chatbots(self):
        """Create ChatBot instances for both standard and Ebbinghaus modes."""
        try:
            # Import ChatBot here to avoid circular imports
            import sys
            import os
            import time
            import shutil
            
            # Add project root to path if not already there
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            from chatbot import ChatBot
            
            self.logger.info("Initializing ChatBot instances...")
            
            # Clean up any existing Qdrant database to ensure fresh start
            qdrant_path = "/tmp/qdrant"
            if os.path.exists(qdrant_path):
                self.logger.info(f"Cleaning up existing Qdrant database at {qdrant_path}")
                try:
                    shutil.rmtree(qdrant_path, ignore_errors=True)
                    time.sleep(1.0)  # Wait for cleanup
                except Exception as e:
                    self.logger.warning(f"Could not clean up existing Qdrant database: {e}")
            
            # Initialize ChatBots for each memory mode sequentially
            for memory_mode in self.config.memory_modes:
                config_mode = self.config.config_modes.get(memory_mode, "testing")
                
                self.logger.info(f"Creating ChatBot for {memory_mode} mode...")
                
                try:
                    chatbot = ChatBot(
                        model_path=self.config.local_model_path,
                        memory_mode=memory_mode,
                        config_mode=config_mode
                    )
                    self.chatbots[memory_mode] = chatbot
                    self.logger.info(f"[INIT] ChatBot for {memory_mode} mode initialized")
                    
                    # Small delay and cleanup before next instance to avoid conflicts
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize {memory_mode} ChatBot: {e}")
                    # If we can't create this ChatBot, continue with others
                    continue
            
            if not self.chatbots:
                raise Exception("No ChatBot instances could be initialized")
            
            # Initialize LLM judge (using OpenAI GPT-4o-mini)
            self.logger.info("Creating LLM judge...")
            import os
            from dotenv import load_dotenv
            
            load_dotenv()  # Load environment variables from .env file
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.warning("OPENAI_API_KEY not found. LLM judge will not be available.")
                self.llm_judge = None
            else:
                self.llm_judge = LLMJudge(api_key)
                self.logger.info("[INIT] LLM judge initialized with GPT-4o-mini")
            
            self.logger.info(f"Initialized {len(self.chatbots)} ChatBot instances successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatBots: {e}")
            raise
    
    def populate_chatbot_memory(self, chatbot, conversation: StandardizedConversation) -> Dict[str, Any]:
        """
        Populate memory with actual LOCOMO conversation data.
        
        Args:
            chatbot: ChatBot instance to populate
            conversation: Standardized conversation data
            
        Returns:
            Dict[str, Any]: Population statistics
        """
        start_time = time.time()
        user_id = f"locomo_{conversation.conversation_id}"
        
        try:
            # Create message pairs from LOCOMO conversation
            messages = []
            for i in range(0, len(conversation.messages) - 1, 2):
                if i + 1 < len(conversation.messages):
                    msg1 = conversation.messages[i]
                    msg2 = conversation.messages[i + 1]
                    
                    messages.extend([
                        {"role": "user", "content": msg1.text},
                        {"role": "assistant", "content": msg2.text}
                    ])
            
            # Add directly to memory (bypass chat simulation)
            messages_added = 0
            if messages:
                # Include timestamp metadata when adding to memory
                # Collect all session timestamps from the conversation
                session_timestamps = {}
                for key, value in conversation.metadata.items():
                    if key.endswith("_date_time") and key.startswith("session_"):
                        session_timestamps[key] = value
                
                # Get the earliest/first session timestamp dynamically
                earliest_session_date = ""
                if session_timestamps:
                    # Sort by session number to get the first one
                    sorted_sessions = sorted(session_timestamps.keys(), 
                                           key=lambda x: int(x.split('_')[1]))
                    earliest_session_date = session_timestamps[sorted_sessions[0]]
                
                metadata = {
                    "session_timestamps": session_timestamps,
                    "conversation_date": earliest_session_date,
                    "original_conversation_id": conversation.conversation_id
                }
                
                result = chatbot.memory.add(messages, user_id=user_id, metadata=metadata)
                messages_added = len(messages)
            
            population_time = time.time() - start_time
            
            return {
                "messages_processed": len(conversation.messages),
                "messages_added": messages_added,
                "population_time": population_time,
                "user_id": user_id
            }
            
        except Exception as e:
            self.logger.error(f"Error populating memory for {conversation.conversation_id}: {e}")
            return {
                "messages_processed": 0,
                "messages_added": 0,
                "population_time": time.time() - start_time,
                "user_id": user_id,
                "error": str(e)
            }
    
    def evaluate_question_with_chatbot(self, chatbot, conversation: StandardizedConversation, 
                                     question_data, user_id: str, memory_mode: str) -> Optional[EvaluationResult]:
        """
        Evaluate a single question using ChatBot.
        
        Args:
            chatbot: ChatBot instance to use
            conversation: Conversation context
            question_data: Question data from dataset
            user_id: User ID for memory retrieval
            memory_mode: Memory mode being evaluated
            
        Returns:
            Optional[EvaluationResult]: Evaluation result or None if failed
        """
        try:
            question = question_data.question
            ground_truth = question_data.answer
            
            # Measure memory search time
            search_start = time.time()
            memories = chatbot.get_memories(question, user_id=user_id, limit=5)
            search_time = time.time() - search_start
            
            # Create memory context for answer generation
            if memories:
                memory_context = "\n".join([mem.get("memory", str(mem)) for mem in memories])
            else:
                memory_context = "No relevant memories found."
            
            # Generate answer using the configured prompt
            generation_start = time.time()
            answer_prompt = create_answer_generation_prompt(memory_context, question)
            generated_answer = chatbot.chat(
                answer_prompt,
                user_id=f"{user_id}_eval",  # Separate eval user to avoid contaminating memory
                max_new_tokens=self.config.answer_max_tokens
            )
            generation_time = time.time() - generation_start
            
            # Calculate traditional metrics
            f1_score = self.metrics_calculator.calculate_f1_score(generated_answer, ground_truth)
            bleu_1_score = self.metrics_calculator.calculate_bleu_1(generated_answer, ground_truth)
            
            # Calculate LLM judge score if enabled
            llm_judge_score = 0.0
            if self.config.use_llm_judge and self.llm_judge:
                try:
                    llm_judge_score = self.llm_judge.judge_answer(question, generated_answer, ground_truth)
                except Exception as e:
                    self.logger.warning(f"LLM judge evaluation failed: {e}")
                    llm_judge_score = 0.0
            
            return EvaluationResult(
                conversation_id=conversation.conversation_id,
                question_id=getattr(question_data, 'id', 0),
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                f1_score=f1_score,
                bleu_1_score=bleu_1_score,
                llm_judge_score=llm_judge_score,
                generation_time=generation_time,
                memory_search_time=search_time,
                memory_mode=memory_mode,
                metadata={
                    "evidence": getattr(question_data, 'evidence', []),
                    "category": getattr(question_data, 'category', None),
                    "memory_count": len(memories),
                    "memory_context_length": len(memory_context)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate question: {e}")
            return None
    
    def evaluate_conversation(self, conversation: StandardizedConversation, 
                            memory_mode: str) -> ConversationEvaluationSummary:
        """
        Evaluate a single conversation with the specified memory mode.
        
        Args:
            conversation: Conversation to evaluate
            memory_mode: Memory mode to use ("standard" or "ebbinghaus")
            
        Returns:
            ConversationEvaluationSummary: Evaluation summary
        """
        self.logger.info(f"[EVAL] Evaluating conversation {conversation.conversation_id} with {memory_mode} memory...")
        
        chatbot = self.chatbots[memory_mode]
        
        # Populate memory with conversation
        population_stats = self.populate_chatbot_memory(chatbot, conversation)
        self.logger.info(f"Populated memory: {population_stats['messages_added']} messages in {population_stats['population_time']:.2f}s")
        
        # Evaluate each question
        results = []
        successful = 0
        failed = 0
        
        for i, question_data in enumerate(conversation.questions):
            result = self.evaluate_question_with_chatbot(
                chatbot, conversation, question_data, 
                population_stats['user_id'], memory_mode
            )
            
            if result:
                results.append(result)
                successful += 1
                self.logger.debug(f"[SUCCESS] Question {i+1}: F1={result.f1_score:.3f}, BLEU={result.bleu_1_score:.3f}, Judge={result.llm_judge_score:.1f}")
            else:
                failed += 1
                self.logger.warning(f"[FAILED] Question {i+1}: Evaluation failed")
        
        # Calculate averages
        if results:
            avg_f1 = sum(r.f1_score for r in results) / len(results)
            avg_bleu = sum(r.bleu_1_score for r in results) / len(results)
            avg_judge = sum(r.llm_judge_score for r in results) / len(results)
            avg_gen_time = sum(r.generation_time for r in results) / len(results)
            avg_search_time = sum(r.memory_search_time for r in results) / len(results)
        else:
            avg_f1 = avg_bleu = avg_judge = avg_gen_time = avg_search_time = 0.0
        
        summary = ConversationEvaluationSummary(
            conversation_id=conversation.conversation_id,
            memory_mode=memory_mode,
            total_questions=len(conversation.questions),
            successful_evaluations=successful,
            failed_evaluations=failed,
            avg_f1_score=avg_f1,
            avg_bleu_1_score=avg_bleu,
            avg_llm_judge_score=avg_judge,
            avg_generation_time=avg_gen_time,
            avg_memory_search_time=avg_search_time,
            results=results
        )
        
        self.logger.info(f"[SUMMARY] Conversation summary: {successful}/{len(conversation.questions)} questions, "
                        f"F1={avg_f1:.3f}, BLEU={avg_bleu:.3f}, Judge={avg_judge:.1f}")
        
        return summary
    
    def run_evaluation(self) -> Dict[str, List[ConversationEvaluationSummary]]:
        """
        Run complete evaluation on the dataset.
            
        Returns:
            Dict[str, List[ConversationEvaluationSummary]]: Results by memory mode
        """
        self.logger.info(f"[START] Starting LOCOMO evaluation with dataset: {self.config.dataset_path}")
        
        # Initialize ChatBots
        self.initialize_chatbots()
        
        # Load dataset - extract filename from full path
        import os
        dataset_filename = os.path.basename(self.config.dataset_path)
        self.logger.info("[LOAD] Loading dataset...")
        conversations = self.dataset_loader.load_conversations(dataset_filename)
        
        # Filter conversations based on config (-1 means all conversations)
        if self.config.max_conversations > 0:
            conversations = conversations[:self.config.max_conversations]
            self.logger.info(f"🔢 Limited to {len(conversations)} conversations for testing")
        else:
            self.logger.info(f"🔢 Using all {len(conversations)} conversations from dataset")
        
        # Filter for quality
        conversations = self.dataset_loader.filter_conversations(
            conversations, min_messages=3, min_questions=1
        )
        
        # Get dataset statistics
        stats = self.dataset_loader.get_dataset_statistics(conversations)
        self.logger.info(f"[STATS] Dataset stats: {stats['total_conversations']} conversations, "
                        f"{stats['total_questions']} questions, "
                        f"{stats['average_questions_per_conversation']:.1f} questions/conv")
        
        # Run evaluation for each memory mode
        results_by_mode = {}
        
        for memory_mode in self.config.memory_modes:
            self.logger.info(f"\n[MODE] Evaluating {memory_mode.upper()} memory mode...")
            mode_results = []
            
            for i, conversation in enumerate(conversations):
                self.logger.info(f"\n--- Conversation {i+1}/{len(conversations)} ---")
                summary = self.evaluate_conversation(conversation, memory_mode)
                mode_results.append(summary)
                
                # Brief progress update
                self.logger.info(f"Progress: {i+1}/{len(conversations)} conversations completed")
            
            results_by_mode[memory_mode] = mode_results
            self.logger.info(f"[COMPLETE] {memory_mode.upper()} mode evaluation completed!")
        
        # Store results
        self.conversation_summaries = results_by_mode
        
        self.logger.info("🎉 Evaluation completed successfully!")
        return results_by_mode
    
    def save_results(self, results: Dict[str, List[ConversationEvaluationSummary]], 
                    output_dir: str = None) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results by memory mode
            output_dir: Output directory (uses config default if None)
            
        Returns:
            str: Path to saved file
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for mode, summaries in results.items():
            serializable_results[mode] = []
            for summary in summaries:
                summary_dict = {
                    "conversation_id": summary.conversation_id,
                    "memory_mode": summary.memory_mode,
                    "total_questions": summary.total_questions,
                    "successful_evaluations": summary.successful_evaluations,
                    "failed_evaluations": summary.failed_evaluations,
                    "avg_f1_score": summary.avg_f1_score,
                    "avg_bleu_1_score": summary.avg_bleu_1_score,
                    "avg_llm_judge_score": summary.avg_llm_judge_score,
                    "avg_generation_time": summary.avg_generation_time,
                    "avg_memory_search_time": summary.avg_memory_search_time,
                    "results": []
                }
                
                # Add individual results
                for result in summary.results:
                    result_dict = {
                        "conversation_id": result.conversation_id,
                        "question_id": result.question_id,
                        "question": result.question,
                        "ground_truth": result.ground_truth,
                        "generated_answer": result.generated_answer,
                        "f1_score": result.f1_score,
                        "bleu_1_score": result.bleu_1_score,
                        "llm_judge_score": result.llm_judge_score,
                        "generation_time": result.generation_time,
                        "memory_search_time": result.memory_search_time,
                        "memory_mode": result.memory_mode,
                        "metadata": result.metadata
                    }
                    summary_dict["results"].append(result_dict)
                
                serializable_results[mode].append(summary_dict)
        
        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"locomo_evaluation_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Results saved to: {filepath}")
        return filepath
    
    def cleanup_chatbots(self):
        """Clean up ChatBot instances."""
        self.logger.info("🧹 Cleaning up ChatBot instances...")
        
        for mode, chatbot in self.chatbots.items():
            try:
                if hasattr(chatbot, 'shutdown'):
                    chatbot.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down {mode} chatbot: {e}")
        
        if self.judge_chatbot and hasattr(self.judge_chatbot, 'shutdown'):
            try:
                self.judge_chatbot.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down judge chatbot: {e}")
        
        self.chatbots.clear()
        self.judge_chatbot = None
        self.llm_judge = None
        
        self.logger.info("[CLEANUP] Cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup_chatbots()
