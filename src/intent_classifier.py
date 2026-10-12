"""
Intent Classifier module for Rocket Failure Chatbot
Rule-based intent classification using regex and keyword matching.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import INTENT_PATTERNS, LOG_LEVEL, LOG_FORMAT
from preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Rule-based intent classifier for user queries.
    Classifies queries into 7 intent types and extracts slot values.
    """
    
    # Intent types (ordered by priority - checked in this order)
    INTENTS = [
        'COUNT_QUERY',
        'STATISTICAL_QUERY',
        'FILTER_QUERY',
        'FACTUAL_LOOKUP',
        'COMPARISON_QUERY',
        'TREND_QUERY',
        'DEFINITION_QUERY'
    ]
    
    # Slot extraction patterns
    SLOT_PATTERNS = {
        'vehicle': r'\b(?:Atlas V|Falcon 9|Soyuz|Proton[-\s]?M?|Delta IV|Titan II(?: GLV)?|Ariane [45]|Space Shuttle|Saturn V|SLS|New Shepard|Starship|Electron|Long March [0-9]+[A-Z]*|H-IIA|PSLV|GSLV|Vega|Zenit[-\s]?3SL|Rokot|Antares|Pegasus XL|Vulcan Centaur|Dnepr|Kosmos|Energia|Thor[-\s]?Agena)\b',
        'country': r'\b(?:USA|USSR|Russia|China|Japan|India|France|ESA|Ukraine|New Zealand|Italy)\b',
        'year': r'\b(19[5-9]\d|20[0-2]\d)\b',
        'decade': r'\b(1950s|1960s|1970s|1980s|1990s|2000s|2010s|2020s)\b',
        'failure_type': r'\b(?:Nozzle Erosion|Fuel System Leak|Ignition Failure|Engine Combustion Instability|Turbopump Failure|Electrical Short Circuit|Guidance System Error|Attitude Control System Malfunction|Software Command Error|Structural Failure|Hydraulic System Failure|Thermal Protection Failure|Propellant Feed System Blockage|Oxidizer Tank Rupture|Oxidizer Valve Malfunction|Fuel Tank Pressurization Loss|Payload Fairing Separation Failure|Stage Separation Failure|Main Engine Cut-Off Anomaly|Roll Control Failure)\b',
        'severity': r'\b(?:Minor|Partial Failure|Major|Catastrophic)\b',
        'failure_phase': r'\b(?:Launch|Ascent|Stage Separation|Upper Stage Burn|Payload Deployment|Re-entry)\b',
        'propellant_type': r'\b(?:LOX/LH2|LOX/Kerosene|N2O4/UDMH|Solid|Hypergolic|LOX/Methane|Hybrid)\b',
        'altitude_range': r'\b(\d{1,3})\s*(?:to|-)\s*(\d{1,3})\s*(?:km|kilometers?)\b',
        'incident_id': r'\bRFI-\d{4}\b'
    }
    
    # Keyword lookup per intent
    INTENT_KEYWORDS = {
        'FILTER_QUERY': ['show', 'list', 'display', 'find', 'get', 'all', 'failures', 'incidents', 'in', 'from', 'by', 'during', 'at'],
        'FACTUAL_LOOKUP': ['what', 'how', 'why', 'when', 'where', 'caused', 'happened', 'occurred', 'tell', 'explain', 'describe', 'incident', 'failure'],
        'COUNT_QUERY': ['how many', 'count', 'number of', 'total', 'sum'],
        'COMPARISON_QUERY': ['which', 'what', 'country', 'vehicle', 'had', 'has', 'more', 'most', 'less', 'fewer', 'compare', 'comparison', 'vs', 'versus', 'difference', 'differ'],
        'TREND_QUERY': ['what', 'which', 'failure', 'type', 'category', 'was', 'is', 'most', 'least', 'common', 'frequent', 'trend', 'pattern', 'over', 'across', 'during', 'in'],
        'STATISTICAL_QUERY': ['average', 'mean', 'median', 'mode', 'std', 'standard deviation', 'what is the', 'maximum', 'minimum', 'highest', 'lowest', 'statistics', 'stats', 'distribution'],
        'DEFINITION_QUERY': ['what is', 'what does', 'define', 'definition', 'meaning', 'explain', 'describe', 'nozzle', 'erosion', 'leak', 'rupture', 'malfunction']
    }
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize IntentClassifier.
        
        Args:
            preprocessor: TextPreprocessor instance for entity extraction
        """
        self.preprocessor = preprocessor or TextPreprocessor(
            use_stemming=False,
            use_lemmatization=False,
            remove_domain_stopwords=False
        )
        
        # Compile regex patterns
        self.compiled_intent_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }
        
        self.compiled_slot_patterns = {
            slot: re.compile(pattern, re.IGNORECASE)
            for slot, pattern in self.SLOT_PATTERNS.items()
        }
        
        logger.info("IntentClassifier initialized")
    
    def classify_intent(self, query: str) -> str:
        """
        Classify the intent of a user query.
        
        Args:
            query: User query string
            
        Returns:
            Intent type string
        """
        query_lower = query.lower().strip()
        
        # Force early exits before any pattern loop
        if re.search(r'^explain\s+what\s+', query_lower):
            logger.debug("Early exit: 'explain what' pattern matched as DEFINITION_QUERY")
            return 'DEFINITION_QUERY'
        
        if re.search(r'^(define|what\s+is\s+a?n?\s+(?!the\s+(most|least|average|mean|median)))', query_lower):
            logger.debug("Early exit: definition pattern matched as DEFINITION_QUERY")
            return 'DEFINITION_QUERY'
        
        # Check each intent's patterns
        for intent in self.INTENTS:
            patterns = self.compiled_intent_patterns.get(intent, [])
            for pattern in patterns:
                if pattern.search(query):
                    logger.debug(f"Intent classified as {intent} using pattern match")
                    return intent
        
        # Fallback: use keyword matching
        intent_scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Intent classified as {best_intent} using keyword matching")
            return best_intent
        
        # Default to FACTUAL_LOOKUP if no match
        logger.debug("No intent pattern matched, defaulting to FACTUAL_LOOKUP")
        return 'FACTUAL_LOOKUP'
    
    def extract_slots(self, query: str) -> Dict[str, any]:
        """
        Extract slot values from the query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary of extracted slot values
        """
        slots = {}
        
        # Extract each slot type
        for slot_name, pattern in self.compiled_slot_patterns.items():
            matches = pattern.findall(query)
            if matches:
                if slot_name == 'altitude_range':
                    # Special handling for range
                    if len(matches[0]) == 2:
                        slots['altitude_min'] = float(matches[0][0])
                        slots['altitude_max'] = float(matches[0][1])
                else:
                    # Take first match
                    slots[slot_name] = matches[0] if isinstance(matches[0], str) else str(matches[0][0])
        
        # Also use preprocessor for entity extraction
        preprocessor_result = self.preprocessor.preprocess(query, extract_entities=True, extract_key_terms=False)
        entities = preprocessor_result['entities']
        
        # Merge entities with slots (prefer regex extraction)
        for entity_type, entity_values in entities.items():
            if entity_type not in slots and entity_values:
                slots[entity_type] = entity_values[0]  # Take first value
        
        logger.debug(f"Extracted slots: {slots}")
        return slots
    
    def classify(self, query: str) -> Dict[str, any]:
        """
        Full classification: intent + slots.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with 'intent' and 'slots' keys
        """
        intent = self.classify_intent(query)
        slots = self.extract_slots(query)
        
        return {
            'intent': intent,
            'slots': slots,
            'query': query
        }
    
    def get_intent_description(self, intent: str) -> str:
        """
        Get a human-readable description of an intent.
        
        Args:
            intent: Intent type string
            
        Returns:
            Description string
        """
        descriptions = {
            'FILTER_QUERY': 'Filter and list incidents based on criteria',
            'FACTUAL_LOOKUP': 'Look up specific information about an incident',
            'COUNT_QUERY': 'Count the number of incidents matching criteria',
            'COMPARISON_QUERY': 'Compare statistics between entities',
            'TREND_QUERY': 'Analyze trends over time',
            'STATISTICAL_QUERY': 'Calculate statistical measures',
            'DEFINITION_QUERY': 'Define or explain a term'
        }
        return descriptions.get(intent, 'Unknown intent')


if __name__ == "__main__":
    # Test the intent classifier
    classifier = IntentClassifier()
    
    # Test queries
    test_queries = [
        "Show me all failures in the USSR",
        "What caused the N1 rocket failure?",
        "How many failures occurred during ascent?",
        "Which country had more failures, USA or USSR?",
        "What failure type was most common in the 1970s?",
        "What is the average altitude of failures?",
        "What is nozzle erosion?",
        "List propulsion failures in the 1980s",
        "Tell me about incident RFI-0001"
    ]
    
    print("="*60)
    print("INTENT CLASSIFIER TEST")
    print("="*60)
    
    for query in test_queries:
        result = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']}")
        print(f"Description: {classifier.get_intent_description(result['intent'])}")
        print(f"Slots: {result['slots']}")
