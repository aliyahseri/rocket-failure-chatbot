"""
Unit tests for TextPreprocessor module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing."""
        return TextPreprocessor(
            use_stemming=True,
            use_lemmatization=False,
            remove_domain_stopwords=True
        )
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.use_stemming is True
        assert preprocessor.use_lemmatization is False
        assert preprocessor.remove_domain_stopwords is True
        assert preprocessor.stemmer is not None
        assert preprocessor.lemmatizer is not None
        assert len(preprocessor.stop_words) > 0
        assert len(preprocessor.synonym_dict) > 0
    
    def test_lowercase(self, preprocessor):
        """Test lowercase transformation."""
        text = "HELLO World"
        result = preprocessor.lowercase(text)
        assert result == "hello world"
    
    def test_remove_punctuation(self, preprocessor):
        """Test punctuation removal."""
        text = "Hello, World! This is a test."
        result = preprocessor.remove_punctuation(text)
        assert "," not in result
        assert "!" not in result
        assert "." not in result
        assert " " in result
    
    def test_tokenize(self, preprocessor):
        """Test tokenization."""
        text = "hello world test"
        tokens = preprocessor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) == 3
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_remove_stopwords(self, preprocessor):
        """Test stopword removal."""
        tokens = ["the", "rocket", "failed", "during", "launch"]
        filtered = preprocessor.remove_stopwords(tokens)
        assert "the" not in filtered
        assert "during" not in filtered
        assert "rocket" in filtered
        assert "failed" in filtered
    
    def test_apply_synonyms(self, preprocessor):
        """Test synonym application."""
        tokens = ["exploded", "engine", "fuel"]
        result = preprocessor.apply_synonyms(tokens)
        assert "combustion" in result  # exploded -> combustion
        assert "propulsion" in result  # engine -> propulsion
        assert "propellant" in result  # fuel -> propellant
    
    def test_stem(self, preprocessor):
        """Test stemming."""
        tokens = ["running", "failed", "launching"]
        result = preprocessor.stem(tokens)
        assert "run" in result
        assert "fail" in result
        assert "launch" in result
    
    def test_lemmatize(self):
        """Test lemmatization."""
        preprocessor = TextPreprocessor(use_stemming=False, use_lemmatization=True)
        tokens = ["running", "failed", "launching"]
        result = preprocessor.lemmatize(tokens)
        assert "running" in result or "run" in result
    
    def test_extract_pos_tags(self, preprocessor):
        """Test POS tagging."""
        tokens = ["the", "rocket", "failed"]
        pos_tags = preprocessor.extract_pos_tags(tokens)
        assert isinstance(pos_tags, list)
        assert len(pos_tags) == 3
        assert all(isinstance(tag, tuple) and len(tag) == 2 for tag in pos_tags)
    
    def test_extract_key_terms(self, preprocessor):
        """Test key term extraction."""
        tokens = ["the", "rocket", "failed", "during", "launch"]
        pos_tags = preprocessor.extract_pos_tags(tokens)
        key_terms = preprocessor.extract_key_terms(tokens, pos_tags)
        assert isinstance(key_terms, list)
        assert len(key_terms) > 0
    
    def test_extract_named_entities(self, preprocessor):
        """Test named entity extraction."""
        text = "The N1 rocket failed in the USSR in 1971 at 184 km altitude"
        entities = preprocessor.extract_named_entities(text)
        assert isinstance(entities, dict)
        # Check for vehicle
        if 'vehicle' in entities:
            assert 'n1' in [e.lower() for e in entities['vehicle']]
        # Check for country
        if 'country' in entities:
            assert 'ussr' in [e.lower() for e in entities['country']]
        # Check for year
        if 'year' in entities:
            assert '1971' in entities['year']
    
    def test_preprocess_empty_text(self, preprocessor):
        """Test preprocessing of empty text."""
        result = preprocessor.preprocess("")
        assert result['processed_text'] == ''
        assert result['tokens'] == []
        assert result['entities'] == {}
        assert result['key_terms'] == []
    
    def test_preprocess_full_pipeline(self, preprocessor):
        """Test full preprocessing pipeline."""
        text = "The N1 rocket exploded during ascent due to nozzle erosion."
        result = preprocessor.preprocess(text)
        
        assert 'processed_text' in result
        assert 'tokens' in result
        assert 'entities' in result
        assert 'key_terms' in result
        assert 'pos_tags' in result
        
        assert isinstance(result['processed_text'], str)
        assert isinstance(result['tokens'], list)
        assert isinstance(result['entities'], dict)
        assert isinstance(result['key_terms'], list)
        assert isinstance(result['pos_tags'], list)
    
    def test_preprocess_batch(self, preprocessor):
        """Test batch preprocessing."""
        texts = [
            "The rocket failed during launch",
            "Nozzle erosion caused thrust loss",
            "Fuel system leak detected"
        ]
        results = preprocessor.preprocess_batch(texts)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all('processed_text' in r for r in results)
        assert all('tokens' in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
