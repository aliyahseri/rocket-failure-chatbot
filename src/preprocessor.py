"""
Text Preprocessor module for Rocket Failure Chatbot
Handles all text preprocessing using NLTK and regex.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import LOG_LEVEL, LOG_FORMAT, PREPROCESSING_PARAMS, SYNONYM_DICT

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """
    Text preprocessing pipeline for rocket failure incident data.
    Uses classical NLP techniques only (no neural models).
    """
    
    def __init__(
        self,
        use_stemming: bool = True,
        use_lemmatization: bool = False,
        remove_domain_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            use_stemming: Whether to apply Porter stemming
            use_lemmatization: Whether to apply WordNet lemmatization
            remove_domain_stopwords: Whether to remove domain-specific stopwords
            custom_stopwords: Additional stopwords to remove
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_domain_stopwords = remove_domain_stopwords
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Get standard stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Add domain-specific stopwords if enabled
        if remove_domain_stopwords:
            domain_stops = PREPROCESSING_PARAMS.get('domain_stopwords', [])
            self.stop_words.update(domain_stops)
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Domain-specific synonym dictionary
        self.synonym_dict = SYNONYM_DICT
        
        # Regex patterns for named entity extraction
        self.patterns = {
            'vehicle': r'\b(?:Atlas V|Falcon 9|Soyuz|Proton[-\s]?M?|Delta IV|Titan II(?: GLV)?|Ariane [45]|Space Shuttle|Saturn V|SLS|New Shepard|Starship|Electron|Long March [0-9]+[A-Z]*|H-IIA|PSLV|GSLV|Vega|Zenit[-\s]?3SL|Rokot|Antares|Pegasus XL|Vulcan Centaur|Dnepr|Kosmos|Energia|Thor[-\s]?Agena)\b',
            'country': r'\b(?:USA|USSR|Russia|China|Japan|India|France|ESA|Ukraine|New Zealand|Italy)\b',
            'year': r'\b(19[5-9]\d|20[0-2]\d)\b',
            'altitude': r'\b(\d{1,3}(?:\.\d+)?)\s*(?:km|kilometers?)\b',
            'incident_id': r'\bRFI-\d{4}\b'
        }
        
        logger.info(f"TextPreprocessor initialized: stemming={use_stemming}, lemmatization={use_lemmatization}")
    
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation and special characters using regex."""
        # Keep alphanumeric, spaces, and basic punctuation for sentence structure
        text = re.sub(r'[^\w\s\-\.,]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK word_tokenize."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [token for token in tokens if token not in self.stop_words]
    
    def apply_synonyms(self, tokens: List[str]) -> List[str]:
        """Apply domain-specific synonym replacement."""
        return [self.synonym_dict.get(token, token) for token in tokens]
    
    def stem(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Apply WordNet lemmatization to tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_pos_tags(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Extract POS tags using NLTK pos_tag.
        Returns list of (token, pos_tag) tuples.
        """
        return pos_tag(tokens)
    
    def extract_key_terms(self, tokens: List[str], pos_tags: List[Tuple[str, str]]) -> List[str]:
        """
        Extract key terms (nouns and verbs) from POS-tagged tokens.
        """
        key_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        key_terms = [token for token, pos in pos_tags if pos in key_pos_tags]
        return key_terms
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using regex patterns.
        Returns dictionary with entity types as keys and lists of matches as values.
        """
        entities = {}
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Normalize matches
                normalized = [m.lower() if isinstance(m, str) else str(m).lower() for m in matches]
                entities[entity_type] = list(set(normalized))  # Remove duplicates
        return entities
    
    def preprocess(
        self,
        text: str,
        extract_entities: bool = True,
        extract_key_terms: bool = True
    ) -> Dict:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
            extract_entities: Whether to extract named entities
            extract_key_terms: Whether to extract key terms (nouns/verbs)
            
        Returns:
            Dictionary containing:
                - 'processed_text': Preprocessed text string
                - 'tokens': List of preprocessed tokens
                - 'entities': Named entities (if extract_entities=True)
                - 'key_terms': Key terms (if extract_key_terms=True)
                - 'pos_tags': POS tags
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text input")
            return {
                'processed_text': '',
                'tokens': [],
                'entities': {},
                'key_terms': [],
                'pos_tags': []
            }
        
        # Step 1: Lowercase
        text = self.lowercase(text)
        
        # Step 2: Extract named entities before preprocessing (from original lowercase text)
        entities = {}
        if extract_entities:
            entities = self.extract_named_entities(text)
        
        # Step 3: Remove punctuation
        text = self.remove_punctuation(text)
        
        # Step 4: Tokenize
        tokens = self.tokenize(text)
        
        # Step 5: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 6: Apply synonyms
        tokens = self.apply_synonyms(tokens)
        
        # Step 7: Stemming or Lemmatization
        if self.use_stemming:
            tokens = self.stem(tokens)
        elif self.use_lemmatization:
            tokens = self.lemmatize(tokens)
        
        # Step 8: POS tagging
        pos_tags = self.extract_pos_tags(tokens)
        
        # Step 9: Extract key terms
        key_terms = []
        if extract_key_terms:
            key_terms = self.extract_key_terms(tokens, pos_tags)
        
        # Reconstruct processed text
        processed_text = ' '.join(tokens)
        
        return {
            'processed_text': processed_text,
            'tokens': tokens,
            'entities': entities,
            'key_terms': key_terms,
            'pos_tags': pos_tags
        }
    
    def preprocess_batch(
        self,
        texts: List[str],
        extract_entities: bool = True,
        extract_key_terms: bool = True
    ) -> List[Dict]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            extract_entities: Whether to extract named entities
            extract_key_terms: Whether to extract key terms
            
        Returns:
            List of preprocessing result dictionaries
        """
        results = []
        for text in texts:
            result = self.preprocess(text, extract_entities, extract_key_terms)
            results.append(result)
        return results


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor(
        use_stemming=True,
        use_lemmatization=False,
        remove_domain_stopwords=True
    )
    
    # Test with sample text
    sample_text = "The N1 rocket exploded during ascent due to nozzle erosion. The combustion gases caused thrust loss."
    
    print("="*60)
    print("TEXT PREPROCESSOR TEST")
    print("="*60)
    print(f"\nOriginal text: {sample_text}")
    
    result = preprocessor.preprocess(sample_text)
    
    print(f"\nProcessed text: {result['processed_text']}")
    print(f"\nTokens: {result['tokens']}")
    print(f"\nKey terms: {result['key_terms']}")
    print(f"\nNamed entities: {result['entities']}")
    print(f"\nPOS tags (first 10): {result['pos_tags'][:10]}")
