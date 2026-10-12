"""
Unit tests for RetrievalEngine module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.retrieval_engine import RetrievalEngine
from src.preprocessor import TextPreprocessor


class TestRetrievalEngine:
    """Test suite for RetrievalEngine class."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        data = {
            'incident_id': ['RFI-0001', 'RFI-0002', 'RFI-0003', 'RFI-0004', 'RFI-0005'],
            'country': ['USSR', 'Russia', 'USA', 'Russia', 'USA'],
            'launch_vehicle': ['N1', 'Rokot', 'New Shepard', 'Soyuz', 'Atlas V'],
            'decade': ['1970s', '1960s', '1980s', '1970s', '2000s'],
            'severity': ['Major', 'Partial Failure', 'Minor', 'Major', 'Catastrophic'],
            'failure_phase': ['Ascent', 'Launch', 'Upper Stage Burn', 'Stage Separation', 'Payload Deployment'],
            'failure_type': ['Nozzle Erosion', 'Fuel System Leak', 'Main Engine Cut-Off Anomaly', 'Attitude Control System Malfunction', 'Thermal Protection Failure'],
            'propellant_type': ['LOX/LH2', 'LOX/Kerosene', 'LOX/LH2', 'LOX/LH2', 'N2O4/UDMH']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_preprocessed_data(self):
        """Create sample preprocessed data for testing."""
        return {
            'documents': [
                'combustion gas erode nozzle material faster designed',
                'propellant leak feed line valve tank',
                'engine shut prematurely sensor malfunction',
                'thruster gyroscope failure prevent vehicle maintain orientation',
                'heat shield insufficient re-entry temperature'
            ],
            'processed_documents': [
                'combustion gas erode nozzle material faster designed',
                'propellant leak feed line valve tank',
                'engine shut prematurely sensor malfunction',
                'thruster gyroscope failure prevent vehicle maintain orientation',
                'heat shield insufficient re-entry temperature'
            ],
            'incident_ids': ['RFI-0001', 'RFI-0002', 'RFI-0003', 'RFI-0004', 'RFI-0005']
        }
    
    @pytest.fixture
    def mock_tfidf_vectorizer(self):
        """Create a mock TF-IDF vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        docs = [
            'combustion gas erode nozzle material faster designed',
            'propellant leak feed line valve tank',
            'engine shut prematurely sensor malfunction',
            'thruster gyroscope failure prevent vehicle maintain orientation',
            'heat shield insufficient re-entry temperature'
        ]
        vectorizer.fit(docs)
        return vectorizer
    
    @pytest.fixture
    def mock_tfidf_matrix(self, mock_tfidf_vectorizer):
        """Create a mock TF-IDF matrix."""
        docs = [
            'combustion gas erode nozzle material faster designed',
            'propellant leak feed line valve tank',
            'engine shut prematurely sensor malfunction',
            'thruster gyroscope failure prevent vehicle maintain orientation',
            'heat shield insufficient re-entry temperature'
        ]
        return mock_tfidf_vectorizer.transform(docs)
    
    @pytest.fixture
    def mock_bm25_index(self, sample_preprocessed_data):
        """Create a mock BM25 index."""
        from rank_bm25 import BM25Okapi
        tokenized_docs = [doc.split() for doc in sample_preprocessed_data['processed_documents']]
        return BM25Okapi(tokenized_docs)
    
    @pytest.fixture
    def mock_inverted_index(self, sample_preprocessed_data):
        """Create a mock inverted index."""
        inverted_index = defaultdict(list)
        for idx, doc in enumerate(sample_preprocessed_data['processed_documents']):
            tokens = set(doc.split())
            for token in tokens:
                if sample_preprocessed_data['incident_ids'][idx] not in inverted_index[token]:
                    inverted_index[token].append(sample_preprocessed_data['incident_ids'][idx])
        return dict(inverted_index)
    
    @pytest.fixture
    def retrieval_engine(
        self,
        mock_tfidf_vectorizer,
        mock_tfidf_matrix,
        mock_bm25_index,
        mock_inverted_index,
        sample_metadata,
        sample_preprocessed_data
    ):
        """Create a retrieval engine instance for testing."""
        preprocessor = TextPreprocessor(use_stemming=False, use_lemmatization=False)
        return RetrievalEngine(
            tfidf_vectorizer=mock_tfidf_vectorizer,
            tfidf_matrix=mock_tfidf_matrix,
            bm25_index=mock_bm25_index,
            inverted_index=mock_inverted_index,
            metadata=sample_metadata,
            preprocessed_data=sample_preprocessed_data,
            preprocessor=preprocessor
        )
    
    def test_initialization(
        self,
        retrieval_engine,
        mock_tfidf_vectorizer,
        mock_tfidf_matrix,
        mock_bm25_index,
        mock_inverted_index,
        sample_metadata,
        sample_preprocessed_data
    ):
        """Test retrieval engine initialization."""
        assert retrieval_engine.tfidf_vectorizer is not None
        assert retrieval_engine.tfidf_matrix is not None
        assert retrieval_engine.bm25_index is not None
        assert retrieval_engine.inverted_index is not None
        assert retrieval_engine.metadata is not None
        assert retrieval_engine.preprocessed_data is not None
        assert retrieval_engine.preprocessor is not None
        assert len(retrieval_engine.incident_id_to_idx) == 5
        assert len(retrieval_engine.idx_to_incident_id) == 5
    
    def test_apply_structured_filter_country(self, retrieval_engine, sample_metadata):
        """Test structured filtering by country."""
        slots = {'country': 'USA'}
        filtered = retrieval_engine.apply_structured_filter(sample_metadata, slots)
        assert len(filtered) == 2
        assert all(country == 'USA' for country in filtered['country'])
    
    def test_apply_structured_filter_vehicle(self, retrieval_engine, sample_metadata):
        """Test structured filtering by vehicle."""
        slots = {'vehicle': 'N1'}
        filtered = retrieval_engine.apply_structured_filter(sample_metadata, slots)
        assert len(filtered) == 1
        assert filtered.iloc[0]['launch_vehicle'] == 'N1'
    
    def test_apply_structured_filter_decade(self, retrieval_engine, sample_metadata):
        """Test structured filtering by decade."""
        slots = {'decade': '1970s'}
        filtered = retrieval_engine.apply_structured_filter(sample_metadata, slots)
        assert len(filtered) == 2
        assert all(decade == '1970s' for decade in filtered['decade'])
    
    def test_apply_structured_filter_multiple(self, retrieval_engine, sample_metadata):
        """Test structured filtering with multiple criteria."""
        slots = {'country': 'USA', 'decade': '1970s'}
        filtered = retrieval_engine.apply_structured_filter(sample_metadata, slots)
        assert len(filtered) == 0  # No USA incidents in 1970s in sample data
    
    def test_apply_structured_filter_no_match(self, retrieval_engine, sample_metadata):
        """Test structured filtering with no matches."""
        slots = {'country': 'China'}
        filtered = retrieval_engine.apply_structured_filter(sample_metadata, slots)
        assert len(filtered) == 0
    
    def test_compute_tfidf_similarity(self, retrieval_engine):
        """Test TF-IDF similarity computation."""
        query = "nozzle erosion combustion"
        scores = retrieval_engine.compute_tfidf_similarity(query)
        assert isinstance(scores, list)
        assert len(scores) == 5
        assert all(isinstance(score, tuple) and len(score) == 2 for score in scores)
        assert all(isinstance(idx, int) for idx, _ in scores)
        assert all(isinstance(score, (int, float, np.floating)) for _, score in scores)
    
    def test_compute_bm25_scores(self, retrieval_engine):
        """Test BM25 score computation."""
        query = "nozzle erosion combustion"
        scores = retrieval_engine.compute_bm25_scores(query)
        assert isinstance(scores, list)
        assert len(scores) == 5
        assert all(isinstance(score, tuple) and len(score) == 2 for score in scores)
    
    def test_normalize_scores(self, retrieval_engine):
        """Test score normalization."""
        scores = [(0, 0.5), (1, 0.8), (2, 0.3), (3, 0.9), (4, 0.2)]
        normalized = retrieval_engine.normalize_scores(scores)
        assert len(normalized) == len(scores)
        assert all(score <= 1.0 for _, score in normalized)
        assert max(score for _, score in normalized) == 1.0
    
    def test_normalize_scores_empty(self, retrieval_engine):
        """Test score normalization with empty list."""
        normalized = retrieval_engine.normalize_scores([])
        assert normalized == []
    
    def test_fuse_scores(self, retrieval_engine):
        """Test score fusion."""
        tfidf_scores = [(0, 0.8), (1, 0.6), (2, 0.4)]
        bm25_scores = [(0, 0.7), (1, 0.9), (2, 0.5)]
        fused = retrieval_engine.fuse_scores(tfidf_scores, bm25_scores)
        assert len(fused) == 3
        assert all(isinstance(score, tuple) and len(score) == 2 for score in fused)
        # Check that fused scores are weighted
        assert fused[0][1] == pytest.approx(0.4 * 0.8 + 0.6 * 0.7)
    
    def test_retrieve_full_pipeline(self, retrieval_engine, sample_metadata):
        """Test full retrieval pipeline."""
        query = "nozzle erosion"
        intent_data = {'intent': 'FACTUAL_LOOKUP', 'slots': {}}
        results = retrieval_engine.retrieve(query, intent_data, sample_metadata)
        
        assert isinstance(results, list)
        assert len(results) <= retrieval_engine.top_k
        assert all('retrieval_score' in result for record in results)
        assert all(isinstance(record['retrieval_score'], (int, float, np.floating)) for record in results)
    
    def test_retrieve_with_filter(self, retrieval_engine, sample_metadata):
        """Test retrieval with structured filtering."""
        query = "failure"
        intent_data = {'intent': 'FILTER_QUERY', 'slots': {'country': 'USA'}}
        results = retrieval_engine.retrieve(query, intent_data, sample_metadata)
        
        assert isinstance(results, list)
        assert all(record['country'] == 'USA' for record in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
