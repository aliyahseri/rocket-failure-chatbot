"""
Feature Engineering module for Rocket Failure Chatbot
Handles TF-IDF, BM25, and inverted index construction.
"""

import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TFIDF_PARAMS, BM25_PARAMS, RETRIEVAL_PARAMS,
    TFIDF_MATRIX_PATH, BM25_INDEX_PATH, INVERTED_INDEX_PATH,
    VECTORIZER_PATH, PREPROCESSED_DATA_PATH, METADATA_PATH, LOG_LEVEL, LOG_FORMAT
)
from preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for retrieval-based QA system.
    Builds TF-IDF matrix, BM25 index, and inverted index.
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            preprocessor: TextPreprocessor instance. If None, creates default.
        """
        self.preprocessor = preprocessor or TextPreprocessor(
            use_stemming=True,
            use_lemmatization=False,
            remove_domain_stopwords=True
        )
        
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25_index = None
        self.inverted_index = None
        self.metadata = None
        self.documents = None
        
        logger.info("FeatureEngineer initialized")
    
    def combine_text_fields(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine cause_description + notes + failure_type + failure_category
        into a single searchable document per record.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.Series: Combined text documents
        """
        logger.info("Combining text fields into searchable documents")
        
        # Fill NaN with empty string
        df = df.copy()
        df['cause_description'] = df['cause_description'].fillna('')
        df['notes'] = df['notes'].fillna('')
        df['failure_type'] = df['failure_type'].fillna('')
        df['failure_category'] = df['failure_category'].fillna('')
        
        # Combine fields with weights
        documents = (
            df['cause_description'] + ' ' +
            df['notes'] + ' ' +
            df['failure_type'] + ' ' +
            df['failure_category']
        )
        
        logger.info(f"Created {len(documents)} combined documents")
        return documents
    
    def preprocess_documents(self, documents: pd.Series) -> List[str]:
        """
        Preprocess all documents using the preprocessor.
        
        Args:
            documents: Series of text documents
            
        Returns:
            List of preprocessed text strings
        """
        logger.info("Preprocessing documents...")
        
        processed_docs = []
        for doc in documents:
            result = self.preprocessor.preprocess(doc, extract_entities=False, extract_key_terms=False)
            processed_docs.append(result['processed_text'])
        
        logger.info(f"Preprocessed {len(processed_docs)} documents")
        return processed_docs
    
    def build_tfidf_matrix(self, documents: List[str]) -> None:
        """
        Build TF-IDF matrix using sklearn TfidfVectorizer.
        
        Args:
            documents: List of preprocessed documents
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        logger.info("Building TF-IDF matrix...")
        
        self.tfidf_vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def build_bm25_index(self, documents: List[str]) -> None:
        """
        Build BM25 index using rank_bm25 library.
        
        Args:
            documents: List of preprocessed documents (tokenized)
        """
        from rank_bm25 import BM25Okapi
        
        logger.info("Building BM25 index...")
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.split() for doc in documents]
        
        self.bm25_index = BM25Okapi(tokenized_docs, **BM25_PARAMS)
        
        logger.info(f"BM25 index built with {len(tokenized_docs)} documents")
    
    def build_inverted_index(self, documents: List[str], incident_ids: List[str]) -> None:
        """
        Build inverted index: term → list of incident_ids.
        
        Args:
            documents: List of preprocessed documents
            incident_ids: List of incident IDs corresponding to documents
        """
        logger.info("Building inverted index...")
        
        self.inverted_index = defaultdict(list)
        
        for doc, incident_id in zip(documents, incident_ids):
            tokens = set(doc.split())  # Use set to avoid duplicates per document
            for token in tokens:
                if incident_id not in self.inverted_index[token]:
                    self.inverted_index[token].append(incident_id)
        
        logger.info(f"Inverted index built with {len(self.inverted_index)} unique terms")
    
    def extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract structured filters as metadata per record.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with metadata columns
        """
        logger.info("Extracting metadata...")
        
        metadata_cols = [
            'incident_id', 'country', 'launch_vehicle', 'decade',
            'severity', 'failure_phase', 'failure_type', 'propellant_type'
        ]
        
        self.metadata = df[metadata_cols].copy()
        
        logger.info(f"Metadata extracted with {len(self.metadata)} records")
        return self.metadata
    
    def build_all_features(
        self,
        df: pd.DataFrame,
        save_artifacts: bool = True
    ) -> Dict:
        """
        Build all features (TF-IDF, BM25, inverted index, metadata).
        
        Args:
            df: Input dataframe
            save_artifacts: Whether to save artifacts to disk
            
        Returns:
            Dictionary containing all built features
        """
        logger.info("="*60)
        logger.info("BUILDING ALL FEATURES")
        logger.info("="*60)
        
        # Step 1: Combine text fields
        self.documents = self.combine_text_fields(df)
        
        # Step 2: Preprocess documents
        processed_docs = self.preprocess_documents(self.documents)
        
        # Step 3: Build TF-IDF matrix
        self.build_tfidf_matrix(processed_docs)
        
        # Step 4: Build BM25 index
        self.build_bm25_index(processed_docs)
        
        # Step 5: Build inverted index
        incident_ids = df['incident_id'].tolist()
        self.build_inverted_index(processed_docs, incident_ids)
        
        # Step 6: Extract metadata
        self.extract_metadata(df)
        
        # Store preprocessed data
        self.preprocessed_data = {
            'documents': self.documents.tolist(),
            'processed_documents': processed_docs,
            'incident_ids': incident_ids
        }
        
        # Step 7: Save artifacts if requested
        if save_artifacts:
            self.save_artifacts()
        
        logger.info("All features built successfully")
        
        return {
            'tfidf_matrix': self.tfidf_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'bm25_index': self.bm25_index,
            'inverted_index': self.inverted_index,
            'metadata': self.metadata,
            'preprocessed_data': self.preprocessed_data
        }
    
    def save_artifacts(self) -> None:
        """Save all artifacts to disk using pickle."""
        logger.info("Saving artifacts to disk...")
        
        # Save TF-IDF matrix
        with open(TFIDF_MATRIX_PATH, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        logger.info(f"Saved TF-IDF matrix to {TFIDF_MATRIX_PATH}")
        
        # Save BM25 index
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        logger.info(f"Saved BM25 index to {BM25_INDEX_PATH}")
        
        # Save inverted index
        with open(INVERTED_INDEX_PATH, 'wb') as f:
            pickle.dump(dict(self.inverted_index), f)
        logger.info(f"Saved inverted index to {INVERTED_INDEX_PATH}")
        
        # Save vectorizer
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        logger.info(f"Saved vectorizer to {VECTORIZER_PATH}")
        
        # Save preprocessed data
        with open(PREPROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(self.preprocessed_data, f)
        logger.info(f"Saved preprocessed data to {PREPROCESSED_DATA_PATH}")
        
        # Save metadata
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved metadata to {METADATA_PATH}")
        
        logger.info("All artifacts saved successfully")
    
    def load_artifacts(self) -> Dict:
        """
        Load all artifacts from disk.
        
        Returns:
            Dictionary containing loaded artifacts
        """
        logger.info("Loading artifacts from disk...")
        
        # Load TF-IDF matrix
        with open(TFIDF_MATRIX_PATH, 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        logger.info(f"Loaded TF-IDF matrix from {TFIDF_MATRIX_PATH}")
        
        # Load BM25 index
        with open(BM25_INDEX_PATH, 'rb') as f:
            self.bm25_index = pickle.load(f)
        logger.info(f"Loaded BM25 index from {BM25_INDEX_PATH}")
        
        # Load inverted index
        with open(INVERTED_INDEX_PATH, 'rb') as f:
            self.inverted_index = defaultdict(list, pickle.load(f))
        logger.info(f"Loaded inverted index from {INVERTED_INDEX_PATH}")
        
        # Load vectorizer
        with open(VECTORIZER_PATH, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        logger.info(f"Loaded vectorizer from {VECTORIZER_PATH}")
        
        # Load preprocessed data
        with open(PREPROCESSED_DATA_PATH, 'rb') as f:
            self.preprocessed_data = pickle.load(f)
        logger.info(f"Loaded preprocessed data from {PREPROCESSED_DATA_PATH}")
        
        # Load metadata
        with open(METADATA_PATH, 'rb') as f:
            self.metadata = pickle.load(f)
        logger.info(f"Loaded metadata from {METADATA_PATH}")
        
        logger.info("All artifacts loaded successfully")
        
        return {
            'tfidf_matrix': self.tfidf_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'bm25_index': self.bm25_index,
            'inverted_index': self.inverted_index,
            'preprocessed_data': self.preprocessed_data,
            'metadata': self.metadata
        }
    
    def get_document_vector(self, query: str) -> List[float]:
        """
        Transform a query using the fitted TF-IDF vectorizer.
        
        Args:
            query: Query string
            
        Returns:
            TF-IDF vector for the query
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not loaded. Call load_artifacts() or build_all_features() first.")
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess(query)['processed_text']
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        return query_vector.toarray()[0].tolist()


if __name__ == "__main__":
    # Test the feature engineer
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_and_validate()
    
    # Build features
    engineer = FeatureEngineer()
    features = engineer.build_all_features(df, save_artifacts=False)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING TEST")
    print("="*60)
    print(f"TF-IDF matrix shape: {features['tfidf_matrix'].shape}")
    print(f"Vocabulary size: {len(features['tfidf_vectorizer'].vocabulary_)}")
    print(f"BM25 index documents: {len(features['bm25_index'].doc_freqs)}")
    print(f"Inverted index terms: {len(features['inverted_index'])}")
    print(f"Metadata records: {len(features['metadata'])}")
    print(f"Preprocessed documents: {len(features['preprocessed_data']['processed_documents'])}")
