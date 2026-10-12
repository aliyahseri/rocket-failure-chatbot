"""
Chatbot Controller module for Rocket Failure Chatbot
Wires all modules together into a single ChatBot class.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
import sys
import os
from pathlib import Path

# Add src directory to path for imports (works from any working directory)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import LOG_LEVEL, LOG_FORMAT
from data_loader import DataLoader
from preprocessor import TextPreprocessor
from feature_engineering import FeatureEngineer
from intent_classifier import IntentClassifier
from retrieval_engine import RetrievalEngine
from answer_extractor import AnswerExtractor

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class RocketFailureChatBot:
    """
    Main chatbot class that integrates all components.
    """
    
    def __init__(self, load_artifacts: bool = True):
        """
        Initialize the chatbot.
        
        Args:
            load_artifacts: Whether to load pre-built artifacts or build from scratch
        """
        logger.info("="*60)
        logger.info("INITIALIZING ROCKET FAILURE CHATBOT")
        logger.info("="*60)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor(
            use_stemming=True,
            use_lemmatization=False,
            remove_domain_stopwords=True
        )
        self.intent_classifier = IntentClassifier(preprocessor=self.preprocessor)
        self.answer_extractor = AnswerExtractor()
        
        # Load dataset
        logger.info("Loading dataset...")
        self.df = self.data_loader.load_and_validate()
        logger.info(f"Dataset loaded: {len(self.df)} records")
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(preprocessor=self.preprocessor)
        
        # Load or build features
        if load_artifacts:
            try:
                logger.info("Loading pre-built artifacts...")
                features = self.feature_engineer.load_artifacts()
                logger.info("Artifacts loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load artifacts: {e}")
                logger.info("Building features from scratch...")
                features = self.feature_engineer.build_all_features(self.df, save_artifacts=True)
        else:
            logger.info("Building features from scratch...")
            features = self.feature_engineer.build_all_features(self.df, save_artifacts=True)
        
        # Initialize retrieval engine
        self.retrieval_engine = RetrievalEngine(
            tfidf_vectorizer=features['tfidf_vectorizer'],
            tfidf_matrix=features['tfidf_matrix'],
            bm25_index=features['bm25_index'],
            inverted_index=features['inverted_index'],
            metadata=features['metadata'],
            preprocessed_data=features['preprocessed_data'],
            preprocessor=self.preprocessor
        )
        
        logger.info("Chatbot initialized successfully")
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query.
        
        Args:
            query: User query string
            
        Returns:
            Preprocessed query string
        """
        result = self.preprocessor.preprocess(query)
        return result['processed_text']
    
    def classify_intent(self, query: str) -> Dict:
        """
        Classify intent of user query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with intent and slots
        """
        return self.intent_classifier.classify(query)
    
    def retrieve(self, query: str, intent_data: Dict) -> List[Dict]:
        """
        Retrieve relevant records for the query.
        
        Args:
            query: User query string
            intent_data: Intent classification result
            
        Returns:
            List of retrieved records
        """
        return self.retrieval_engine.retrieve(query, intent_data, self.df)
    
    def extract_answer(self, results: List[Dict], intent_data: Dict) -> str:
        """
        Extract answer from retrieved results.
        
        Args:
            results: Retrieved records
            intent_data: Intent classification result
            
        Returns:
            Formatted answer string
        """
        return self.answer_extractor.extract_answer(results, intent_data, self.df)
    
    def chat(self, user_input: str) -> Dict:
        """
        Full pipeline: preprocess → intent → retrieve → answer.
        
        Args:
            user_input: User query string
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {user_input}")
        
        # Step 1: Classify intent
        intent_data = self.classify_intent(user_input)
        logger.info(f"Intent: {intent_data['intent']}")
        
        # Step 2: Retrieve relevant records
        results = self.retrieve(user_input, intent_data)
        logger.info(f"Retrieved {len(results)} results")
        
        # Step 3: Extract answer
        answer = self.extract_answer(results, intent_data)
        
        return {
            'answer': answer,
            'intent': intent_data['intent'],
            'slots': intent_data['slots'],
            'num_results': len(results),
            'results': results
        }
    
    def run_cli(self):
        """
        Run interactive terminal loop with history.
        """
        print("\n" + "="*60)
        print("ROCKET FAILURE QA CHATBOT")
        print("="*60)
        print("Ask questions about rocket failure incidents (1957-2023)")
        print("Type 'quit' or 'exit' to end the session")
        print("="*60 + "\n")
        
        history = []
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Rocket Failure QA Chatbot!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process query
                response = self.chat(user_input)
                
                # Display answer
                print(f"\nBot [{response['intent']}]:")
                print(response['answer'])
                
                # Display metadata if results found
                if response['num_results'] > 0:
                    print(f"\n[Retrieved {response['num_results']} record(s)]")
                
                print("\n" + "-"*60 + "\n")
                
                # Add to history
                history.append({
                    'query': user_input,
                    'response': response
                })
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nError: {e}\nPlease try again.\n")
        
        return history


if __name__ == "__main__":
    # Create and run chatbot
    chatbot = RocketFailureChatBot(load_artifacts=False)
    
    # Run CLI interface
    chatbot.run_cli()
