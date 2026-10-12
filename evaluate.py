"""
Evaluation script for Rocket Failure Chatbot
Computes Precision@k, Recall@k, MRR, intent accuracy, and response time.
"""

import time
import pandas as pd
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from src.chatbot import RocketFailureChatBot
from src.intent_classifier import IntentClassifier


class Evaluator:
    """
    Evaluator for the chatbot system.
    """
    
    def __init__(self, chatbot: RocketFailureChatBot):
        """
        Initialize evaluator.
        
        Args:
            chatbot: Initialized chatbot instance
        """
        self.chatbot = chatbot
        self.intent_classifier = IntentClassifier()
    
    def compute_precision_at_k(
        self,
        retrieved: List[Dict],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Compute Precision@k.
        
        Args:
            retrieved: List of retrieved records
            relevant_ids: List of relevant incident IDs
            k: Cutoff rank
            
        Returns:
            Precision@k score
        """
        if not retrieved or not relevant_ids:
            return 0.0
        
        retrieved_ids = [r['incident_id'] for r in retrieved[:k]]
        relevant_retrieved = sum(1 for rid in retrieved_ids if rid in relevant_ids)
        
        return relevant_retrieved / min(k, len(retrieved))
    
    def compute_recall_at_k(
        self,
        retrieved: List[Dict],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Compute Recall@k.
        
        Args:
            retrieved: List of retrieved records
            relevant_ids: List of relevant incident IDs
            k: Cutoff rank
            
        Returns:
            Recall@k score
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_ids = [r['incident_id'] for r in retrieved[:k]]
        relevant_retrieved = sum(1 for rid in retrieved_ids if rid in relevant_ids)
        
        return relevant_retrieved / len(relevant_ids)
    
    def compute_mrr(
        self,
        retrieved: List[Dict],
        relevant_ids: List[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank.
        
        Args:
            retrieved: List of retrieved records
            relevant_ids: List of relevant incident IDs
            
        Returns:
            MRR score
        """
        if not retrieved or not relevant_ids:
            return 0.0
        
        for i, record in enumerate(retrieved):
            if record['incident_id'] in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate_retrieval(
        self,
        test_queries: List[Dict],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict:
        """
        Evaluate retrieval performance on test queries.
        
        Args:
            test_queries: List of test query dicts with 'query' and 'relevant_ids'
            k_values: List of k values for Precision@k and Recall@k
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'precision_at_k': {},
            'recall_at_k': {},
            'mrr': [],
            'response_times': []
        }
        
        for k in k_values:
            results['precision_at_k'][k] = []
            results['recall_at_k'][k] = []
        
        for test_case in test_queries:
            query = test_case['query']
            relevant_ids = test_case['relevant_ids']
            
            # Measure response time
            start_time = time.time()
            
            # Retrieve
            intent_data = self.intent_classifier.classify(query)
            retrieved = self.chatbot.retrieve(query, intent_data)
            
            end_time = time.time()
            response_time = end_time - start_time
            results['response_times'].append(response_time)
            
            # Compute metrics
            for k in k_values:
                p_at_k = self.compute_precision_at_k(retrieved, relevant_ids, k)
                r_at_k = self.compute_recall_at_k(retrieved, relevant_ids, k)
                results['precision_at_k'][k].append(p_at_k)
                results['recall_at_k'][k].append(r_at_k)
            
            # Compute MRR
            mrr = self.compute_mrr(retrieved, relevant_ids)
            results['mrr'].append(mrr)
        
        # Compute averages
        summary = {
            'num_queries': len(test_queries),
            'avg_response_time': sum(results['response_times']) / len(results['response_times']),
            'precision_at_k': {},
            'recall_at_k': {},
            'mrr': sum(results['mrr']) / len(results['mrr'])
        }
        
        for k in k_values:
            summary['precision_at_k'][k] = sum(results['precision_at_k'][k]) / len(results['precision_at_k'][k])
            summary['recall_at_k'][k] = sum(results['recall_at_k'][k]) / len(results['recall_at_k'][k])
        
        return summary
    
    def evaluate_intent_classification(
        self,
        test_queries: List[Dict]
    ) -> Dict:
        """
        Evaluate intent classification accuracy.
        
        Args:
            test_queries: List of test query dicts with 'query' and 'expected_intent'
            
        Returns:
            Dictionary with intent classification metrics
        """
        correct = 0
        total = len(test_queries)
        
        intent_predictions = []
        
        for test_case in test_queries:
            query = test_case['query']
            expected_intent = test_case['expected_intent']
            
            predicted_intent = self.intent_classifier.classify_intent(query)
            intent_predictions.append({
                'query': query,
                'expected': expected_intent,
                'predicted': predicted_intent,
                'correct': predicted_intent == expected_intent
            })
            
            if predicted_intent == expected_intent:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': intent_predictions
        }
    
    def print_evaluation_table(self, retrieval_results: Dict, intent_results: Dict):
        """
        Print evaluation results in a formatted table.
        
        Args:
            retrieval_results: Retrieval evaluation results
            intent_results: Intent classification results
        """
        print("\n" + "="*80)
        print("ROCKET FAILURE CHATBOT - EVALUATION RESULTS")
        print("="*80)
        
        print("\n" + "-"*80)
        print("RETRIEVAL METRICS")
        print("-"*80)
        print(f"Number of test queries: {retrieval_results['num_queries']}")
        print(f"Average response time: {retrieval_results['avg_response_time']:.4f} seconds")
        print(f"\nPrecision@k:")
        for k, p_at_k in retrieval_results['precision_at_k'].items():
            print(f"  P@{k}: {p_at_k:.4f}")
        
        print(f"\nRecall@k:")
        for k, r_at_k in retrieval_results['recall_at_k'].items():
            print(f"  R@{k}: {r_at_k:.4f}")
        
        print(f"\nMean Reciprocal Rank (MRR): {retrieval_results['mrr']:.4f}")
        
        print("\n" + "-"*80)
        print("INTENT CLASSIFICATION METRICS")
        print("-"*80)
        print(f"Accuracy: {intent_results['accuracy']:.4f} ({intent_results['correct']}/{intent_results['total']})")
        
        print("\n" + "-"*80)
        print("INTENT PREDICTION DETAILS")
        print("-"*80)
        for pred in intent_results['predictions']:
            status = "✓" if pred['correct'] else "✗"
            print(f"{status} Query: {pred['query'][:50]}...")
            print(f"  Expected: {pred['expected']}, Predicted: {pred['predicted']}")
        
        print("\n" + "="*80)


def get_test_queries() -> Tuple[List[Dict], List[Dict]]:
    """
    Get test queries for evaluation.
    
    Returns:
        Tuple of (retrieval_test_queries, intent_test_queries)
    """
    # Retrieval test queries with relevant incident IDs
    retrieval_test_queries = [
        {
            'query': 'What caused the N1 rocket failure?',
            'relevant_ids': ['RFI-0001', 'RFI-0037', 'RFI-0090', 'RFI-0095', 'RFI-0098', 'RFI-0158']
        },
        {
            'query': 'Show me failures in the USSR',
            'relevant_ids': ['RFI-0001', 'RFI-0037', 'RFI-0090', 'RFI-0095', 'RFI-0158']
        },
        {
            'query': 'Fuel system leak incidents',
            'relevant_ids': ['RFI-0002', 'RFI-0010', 'RFI-0013', 'RFI-0018', 'RFI-0118']
        },
        {
            'query': 'Failures during ascent phase',
            'relevant_ids': ['RFI-0001', 'RFI-0018', 'RFI-0029', 'RFI-0030', 'RFI-0031']
        },
        {
            'query': 'Catastrophic failures',
            'relevant_ids': ['RFI-0005', 'RFI-0006', 'RFI-0007', 'RFI-0011', 'RFI-0015']
        },
        {
            'query': 'Atlas V rocket failures',
            'relevant_ids': ['RFI-0005', 'RFI-0014', 'RFI-0019', 'RFI-0091', 'RFI-0147']
        },
        {
            'query': 'Failures in the 1970s',
            'relevant_ids': ['RFI-0001', 'RFI-0004', 'RFI-0019', 'RFI-0023', 'RFI-0034']
        },
        {
            'query': 'Nozzle erosion incidents',
            'relevant_ids': ['RFI-0001', 'RFI-0008', 'RFI-0037', 'RFI-0080', 'RFI-0084']
        },
        {
            'query': 'Electrical short circuit failures',
            'relevant_ids': ['RFI-0009', 'RFI-0020', 'RFI-0116', 'RFI-0145', 'RFI-0122']
        },
        {
            'query': 'Failures with LOX/LH2 propellant',
            'relevant_ids': ['RFI-0001', 'RFI-0003', 'RFI-0004', 'RFI-0013', 'RFI-0071']
        }
    ]
    
    # Intent classification test queries
    intent_test_queries = [
        {'query': 'Show me all failures in the USSR', 'expected_intent': 'FILTER_QUERY'},
        {'query': 'What caused the N1 rocket failure?', 'expected_intent': 'FACTUAL_LOOKUP'},
        {'query': 'How many failures occurred during ascent?', 'expected_intent': 'COUNT_QUERY'},
        {'query': 'Which country had more failures, USA or USSR?', 'expected_intent': 'COMPARISON_QUERY'},
        {'query': 'What failure type was most common in the 1970s?', 'expected_intent': 'TREND_QUERY'},
        {'query': 'What is the average altitude of failures?', 'expected_intent': 'STATISTICAL_QUERY'},
        {'query': 'What is nozzle erosion?', 'expected_intent': 'DEFINITION_QUERY'},
        {'query': 'List propulsion failures in the 1980s', 'expected_intent': 'FILTER_QUERY'},
        {'query': 'Tell me about incident RFI-0001', 'expected_intent': 'FACTUAL_LOOKUP'},
        {'query': 'Count the number of catastrophic failures', 'expected_intent': 'COUNT_QUERY'},
        {'query': 'Compare USA and Russia failure rates', 'expected_intent': 'COMPARISON_QUERY'},
        {'query': 'What is the most common severity level?', 'expected_intent': 'STATISTICAL_QUERY'},
        {'query': 'Define fuel system leak', 'expected_intent': 'DEFINITION_QUERY'},
        {'query': 'Show failures with solid propellant', 'expected_intent': 'FILTER_QUERY'},
        {'query': 'What happened during the Space Shuttle failure?', 'expected_intent': 'FACTUAL_LOOKUP'},
        {'query': 'How many incidents involved LOX/Kerosene?', 'expected_intent': 'COUNT_QUERY'},
        {'query': 'What trend do we see in failure types over decades?', 'expected_intent': 'TREND_QUERY'},
        {'query': 'What is the median time to failure?', 'expected_intent': 'STATISTICAL_QUERY'},
        {'query': 'Explain what turbopump failure means', 'expected_intent': 'DEFINITION_QUERY'},
        {'query': 'Find all major severity failures', 'expected_intent': 'FILTER_QUERY'}
    ]
    
    return retrieval_test_queries, intent_test_queries


def main():
    """Main evaluation function."""
    print("Initializing chatbot...")
    chatbot = RocketFailureChatBot(load_artifacts=True)
    
    evaluator = Evaluator(chatbot)
    
    # Get test queries
    retrieval_test_queries, intent_test_queries = get_test_queries()
    
    # Evaluate retrieval
    print("\nEvaluating retrieval performance...")
    retrieval_results = evaluator.evaluate_retrieval(retrieval_test_queries, k_values=[1, 3, 5])
    
    # Evaluate intent classification
    print("\nEvaluating intent classification...")
    intent_results = evaluator.evaluate_intent_classification(intent_test_queries)
    
    # Print results
    evaluator.print_evaluation_table(retrieval_results, intent_results)


if __name__ == "__main__":
    main()
