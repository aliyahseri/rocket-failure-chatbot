"""
Retrieval Engine module for Rocket Failure Chatbot
Hybrid retrieval system combining structured filtering, TF-IDF, and BM25.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from difflib import get_close_matches
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import RETRIEVAL_PARAMS, LOG_LEVEL, LOG_FORMAT
from preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Hybrid retrieval engine combining structured filtering, TF-IDF, and BM25.
    """
    
    def __init__(
        self,
        tfidf_vectorizer,
        tfidf_matrix,
        bm25_index,
        inverted_index,
        metadata: pd.DataFrame,
        preprocessed_data: Dict,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        """
        Initialize RetrievalEngine.
        
        Args:
            tfidf_vectorizer: Fitted TfidfVectorizer
            tfidf_matrix: TF-IDF matrix for documents
            bm25_index: BM25 index
            inverted_index: Inverted index (term -> incident_ids)
            metadata: DataFrame with structured metadata
            preprocessed_data: Dictionary with preprocessed documents and IDs
            preprocessor: TextPreprocessor instance
        """
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.bm25_index = bm25_index
        self.inverted_index = inverted_index
        self.metadata = metadata
        self.preprocessed_data = preprocessed_data
        self.preprocessor = preprocessor or TextPreprocessor(
            use_stemming=True,
            use_lemmatization=False,
            remove_domain_stopwords=True
        )
        
        # Build incident_id to index mapping
        self.incident_id_to_idx = {
            incident_id: idx 
            for idx, incident_id in enumerate(preprocessed_data['incident_ids'])
        }
        
        # Build index to incident_id mapping
        self.idx_to_incident_id = {
            idx: incident_id 
            for incident_id, idx in self.incident_id_to_idx.items()
        }
        
        # Get retrieval parameters
        self.top_k = RETRIEVAL_PARAMS['top_k']
        self.tfidf_weight = RETRIEVAL_PARAMS['tfidf_weight']
        self.bm25_weight = RETRIEVAL_PARAMS['bm25_weight']
        self.min_score = RETRIEVAL_PARAMS['min_score']
        
        logger.info("RetrievalEngine initialized")
    
    def apply_structured_filter(
        self,
        df: pd.DataFrame,
        slots: Dict
    ) -> pd.DataFrame:
        """
        Apply structured filtering based on extracted slots.
        
        Args:
            df: DataFrame to filter
            slots: Extracted slot values
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Filter by country
        if 'country' in slots:
            country = slots['country']
            # Try exact match first
            mask = filtered_df['country'].str.lower() == country.lower()
            if not mask.any():
                # Try fuzzy match
                countries = filtered_df['country'].unique().tolist()
                matches = get_close_matches(country, countries, n=1, cutoff=0.6)
                if matches:
                    mask = filtered_df['country'].str.lower() == matches[0].lower()
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by country: {country}, {len(filtered_df)} records remaining")
        
        # Filter by vehicle
        if 'vehicle' in slots:
            vehicle = slots['vehicle']
            mask = filtered_df['launch_vehicle'].str.lower() == vehicle.lower()
            if not mask.any():
                # Try fuzzy match
                vehicles = filtered_df['launch_vehicle'].unique().tolist()
                matches = get_close_matches(vehicle, vehicles, n=1, cutoff=0.6)
                if matches:
                    mask = filtered_df['launch_vehicle'].str.lower() == matches[0].lower()
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by vehicle: {vehicle}, {len(filtered_df)} records remaining")
        
        # Filter by decade
        if 'decade' in slots:
            decade = slots['decade']
            mask = filtered_df['decade'].str.lower() == decade.lower()
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by decade: {decade}, {len(filtered_df)} records remaining")
        
        # Filter by year
        if 'year' in slots:
            year = int(slots['year'])
            mask = filtered_df['year'] == year
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by year: {year}, {len(filtered_df)} records remaining")
        
        # Filter by failure_type
        if 'failure_type' in slots:
            failure_type = slots['failure_type']
            mask = filtered_df['failure_type'].str.lower() == failure_type.lower()
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by failure_type: {failure_type}, {len(filtered_df)} records remaining")
        
        # Filter by severity
        if 'severity' in slots:
            severity = slots['severity']
            mask = filtered_df['severity'].str.lower() == severity.lower()
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by severity: {severity}, {len(filtered_df)} records remaining")
        
        # Filter by failure_phase
        if 'failure_phase' in slots:
            failure_phase = slots['failure_phase']
            mask = filtered_df['failure_phase'].str.lower() == failure_phase.lower()
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by failure_phase: {failure_phase}, {len(filtered_df)} records remaining")
        
        # Filter by propellant_type
        if 'propellant_type' in slots:
            propellant_type = slots['propellant_type']
            mask = filtered_df['propellant_type'].str.lower() == propellant_type.lower()
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by propellant_type: {propellant_type}, {len(filtered_df)} records remaining")
        
        # Filter by altitude range
        if 'altitude_min' in slots and 'altitude_max' in slots:
            mask = (filtered_df['altitude_km'] >= slots['altitude_min']) & \
                   (filtered_df['altitude_km'] <= slots['altitude_max'])
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by altitude range: {slots['altitude_min']}-{slots['altitude_max']} km, {len(filtered_df)} records remaining")
        
        # Filter by incident_id
        if 'incident_id' in slots:
            incident_id = slots['incident_id'].upper()
            mask = filtered_df['incident_id'] == incident_id
            filtered_df = filtered_df[mask]
            logger.debug(f"Filtered by incident_id: {incident_id}, {len(filtered_df)} records remaining")
        
        return filtered_df
    
    def compute_tfidf_similarity(
        self,
        query: str,
        candidate_indices: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Compute TF-IDF cosine similarity between query and documents.
        
        Args:
            query: Preprocessed query string
            candidate_indices: List of document indices to consider (if None, use all)
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Get candidate matrix
        if candidate_indices:
            candidate_matrix = self.tfidf_matrix[candidate_indices]
        else:
            candidate_matrix = self.tfidf_matrix
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
        
        # Create (index, score) tuples
        if candidate_indices:
            results = [(candidate_indices[i], similarities[i]) for i in range(len(similarities))]
        else:
            results = [(i, similarities[i]) for i in range(len(similarities))]
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def compute_bm25_scores(
        self,
        query: str,
        candidate_indices: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Compute BM25 scores for query against documents.
        
        Args:
            query: Preprocessed query string
            candidate_indices: List of document indices to consider (if None, use all)
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        # Tokenize query
        query_tokens = query.split()
        
        # Get all documents
        all_docs = self.preprocessed_data['processed_documents']
        
        # Get candidate documents
        if candidate_indices:
            candidate_docs = [all_docs[i].split() for i in candidate_indices]
        else:
            candidate_docs = [doc.split() for doc in all_docs]
            candidate_indices = list(range(len(all_docs)))
        
        # Compute BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # If we have candidates, filter scores
        if candidate_indices:
            candidate_scores = [(candidate_indices[i], scores[i]) for i in range(len(candidate_indices))]
        else:
            candidate_scores = [(i, scores[i]) for i in range(len(scores))]
        
        # Sort by score descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        return candidate_scores
    
    def normalize_scores(self, scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: List of (index, score) tuples
            
        Returns:
            Normalized list of (index, score) tuples
        """
        if not scores:
            return []
        
        max_score = max(score for _, score in scores)
        if max_score == 0:
            return scores
        
        normalized = [(idx, score / max_score) for idx, score in scores]
        return normalized
    
    def fuse_scores(
        self,
        tfidf_scores: List[Tuple[int, float]],
        bm25_scores: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Fuse TF-IDF and BM25 scores using weighted combination.
        
        Args:
            tfidf_scores: List of (index, tfidf_score) tuples
            bm25_scores: List of (index, bm25_score) tuples
            
        Returns:
            List of (index, fused_score) tuples sorted by fused score
        """
        # Create score dictionaries
        tfidf_dict = {idx: score for idx, score in tfidf_scores}
        bm25_dict = {idx: score for idx, score in bm25_scores}
        
        # Get all unique indices
        all_indices = set(tfidf_dict.keys()) | set(bm25_dict.keys())
        
        # Fuse scores
        fused_scores = []
        for idx in all_indices:
            tfidf_score = tfidf_dict.get(idx, 0.0)
            bm25_score = bm25_dict.get(idx, 0.0)
            fused_score = (self.tfidf_weight * tfidf_score) + (self.bm25_weight * bm25_score)
            fused_scores.append((idx, fused_score))
        
        # Sort by fused score descending
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        
        return fused_scores
    
    def retrieve(
        self,
        query: str,
        intent_data: Dict,
        df: pd.DataFrame
    ) -> List[Dict]:
        """
        Full retrieval pipeline: filter → TF-IDF → BM25 → fuse → rank → threshold.
        
        Args:
            query: User query string
            intent_data: Intent classification result with slots
            df: Full dataframe
            
        Returns:
            List of retrieved records with scores
        """
        logger.info(f"Retrieving for query: {query}")
        
        # Step 1: Preprocess query
        preprocessed_query = self.preprocessor.preprocess(query)['processed_text']
        
        # Step 2: Apply structured filtering
        slots = intent_data.get('slots', {})
        filtered_df = self.apply_structured_filter(df, slots)
        
        if len(filtered_df) == 0:
            logger.warning("No records after structured filtering")
            return []
        
        # Get candidate indices
        candidate_ids = filtered_df['incident_id'].tolist()
        candidate_indices = [self.incident_id_to_idx[incident_id] for incident_id in candidate_ids]
        
        logger.info(f"Structured filtering: {len(df)} → {len(filtered_df)} records")
        
        # Step 3: Compute TF-IDF similarity
        tfidf_scores = self.compute_tfidf_similarity(preprocessed_query, candidate_indices)
        tfidf_scores = self.normalize_scores(tfidf_scores)
        
        # Step 4: Compute BM25 scores
        bm25_scores = self.compute_bm25_scores(preprocessed_query, candidate_indices)
        bm25_scores = self.normalize_scores(bm25_scores)
        
        # Step 5: Fuse scores
        fused_scores = self.fuse_scores(tfidf_scores, bm25_scores)
        
        # Step 6: Apply threshold
        filtered_scores = [(idx, score) for idx, score in fused_scores if score >= self.min_score]
        
        # Step 7: Get top-k results
        top_results = filtered_scores[:self.top_k]
        
        # Step 8: Build result dictionaries
        results = []
        for idx, score in top_results:
            incident_id = self.idx_to_incident_id[idx]
            record = filtered_df[filtered_df['incident_id'] == incident_id].iloc[0].to_dict()
            record['retrieval_score'] = score
            results.append(record)
        
        logger.info(f"Retrieved {len(results)} results")
        
        return results


if __name__ == "__main__":
    # Test the retrieval engine
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    # Load data
    loader = DataLoader()
    df = loader.load_and_validate()
    
    # Build features
    engineer = FeatureEngineer()
    features = engineer.build_all_features(df, save_artifacts=False)
    
    # Initialize retrieval engine
    retrieval_engine = RetrievalEngine(
        tfidf_vectorizer=features['tfidf_vectorizer'],
        tfidf_matrix=features['tfidf_matrix'],
        bm25_index=features['bm25_index'],
        inverted_index=features['inverted_index'],
        metadata=features['metadata'],
        preprocessed_data=features['preprocessed_data']
    )
    
    # Test retrieval
    from intent_classifier import IntentClassifier
    classifier = IntentClassifier()
    
    test_query = "What caused the N1 rocket failure?"
    intent_data = classifier.classify(test_query)
    
    results = retrieval_engine.retrieve(test_query, intent_data, df)
    
    print("\n" + "="*60)
    print("RETRIEVAL ENGINE TEST")
    print("="*60)
    print(f"Query: {test_query}")
    print(f"Intent: {intent_data['intent']}")
    print(f"Slots: {intent_data['slots']}")
    print(f"\nRetrieved {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['incident_id']} - Score: {result['retrieval_score']:.4f}")
        print(f"   Vehicle: {result['launch_vehicle']}")
        print(f"   Failure Type: {result['failure_type']}")
        print(f"   Cause: {result['cause_description'][:100]}...")
