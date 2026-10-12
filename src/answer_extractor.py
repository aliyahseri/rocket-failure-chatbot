"""
Answer Extractor module for Rocket Failure Chatbot
Template-based answer generation for different intent types.
"""

import pandas as pd
from typing import Dict, List, Optional
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class AnswerExtractor:
    """
    Template-based answer generator for different intent types.
    All answers are deterministic and cite source records.
    """
    
    def __init__(self):
        """Initialize AnswerExtractor."""
        logger.info("AnswerExtractor initialized")
    
    def format_incident_details(self, record: Dict) -> str:
        """
        Format incident details in readable format.
        
        Args:
            record: Dictionary containing incident data
            
        Returns:
            Formatted string with incident details
        """
        details = []
        details.append(f"Incident ID: {record.get('incident_id', 'N/A')}")
        details.append(f"Date: {record.get('date', 'N/A')}")
        details.append(f"Launch Vehicle: {record.get('launch_vehicle', 'N/A')}")
        details.append(f"Country: {record.get('country', 'N/A')}")
        details.append(f"Launch Site: {record.get('launch_site', 'N/A')}")
        details.append(f"Failure Type: {record.get('failure_type', 'N/A')}")
        details.append(f"Failure Category: {record.get('failure_category', 'N/A')}")
        details.append(f"Failure Phase: {record.get('failure_phase', 'N/A')}")
        details.append(f"Severity: {record.get('severity', 'N/A')}")
        details.append(f"Propellant Type: {record.get('propellant_type', 'N/A')}")
        details.append(f"Altitude: {record.get('altitude_km', 'N/A')} km")
        details.append(f"Time to Failure: {record.get('time_to_failure_sec', 'N/A')} sec")
        details.append(f"Mission Success: {'Yes' if record.get('mission_success') == 1 else 'No'}")
        details.append(f"\nCause Description: {record.get('cause_description', 'N/A')}")
        details.append(f"Notes: {record.get('notes', 'N/A')}")
        
        return "\n".join(details)
    
    def extract_factual_lookup(self, results: List[Dict], intent_data: Dict) -> str:
        """
        Extract answer for FACTUAL_LOOKUP intent.
        
        Args:
            results: Retrieved records
            intent_data: Intent classification result
            
        Returns:
            Formatted answer string
        """
        if not results:
            return "No matching incidents found for your query."
        
        # Return the top result with full details
        top_result = results[0]
        answer = self.format_incident_details(top_result)
        
        # Add source citation
        source = f"\n\nSource: incident_id {top_result['incident_id']}, {top_result['launch_vehicle']}, {top_result['date']}"
        answer += source
        
        return answer
    
    def extract_filter_query(self, results: List[Dict], intent_data: Dict) -> str:
        """
        Extract answer for FILTER_QUERY intent.
        
        Args:
            results: Retrieved records
            intent_data: Intent classification result
            
        Returns:
            Formatted answer string with summarized table
        """
        if not results:
            return "No matching incidents found for your query."
        
        # Create summary table
        answer = f"Found {len(results)} matching incidents:\n\n"
        answer += f"{'Incident ID':<12} {'Date':<12} {'Vehicle':<15} {'Failure Type':<30} {'Severity':<15}\n"
        answer += "-" * 84 + "\n"
        
        for record in results:
            answer += f"{record['incident_id']:<12} {record['date']:<12} {record['launch_vehicle']:<15} {record['failure_type'][:28]:<30} {record['severity']:<15}\n"
        
        # Add source citations
        sources = ", ".join([f"{r['incident_id']}" for r in results])
        answer += f"\n\nSources: {sources}"
        
        return answer
    
    def extract_count_query(self, results: List[Dict], intent_data: Dict, df: pd.DataFrame) -> str:
        """
        Extract answer for COUNT_QUERY intent.
        
        Args:
            results: Retrieved records
            intent_data: Intent classification result
            df: Full dataframe for aggregation
            
        Returns:
            Formatted answer string with count
        """
        slots = intent_data.get('slots', {})
        
        # Perform aggregation based on slots
        if not slots:
            count = len(df)
            answer = f"There are {count} total rocket failure incidents in the dataset (1957-2023)."
        else:
            # Filter based on slots
            filtered_df = df.copy()
            
            if 'country' in slots:
                filtered_df = filtered_df[filtered_df['country'].str.lower() == slots['country'].lower()]
            if 'vehicle' in slots:
                filtered_df = filtered_df[filtered_df['launch_vehicle'].str.lower() == slots['vehicle'].lower()]
            if 'decade' in slots:
                filtered_df = filtered_df[filtered_df['decade'].str.lower() == slots['decade'].lower()]
            if 'failure_type' in slots:
                filtered_df = filtered_df[filtered_df['failure_type'].str.lower() == slots['failure_type'].lower()]
            if 'severity' in slots:
                filtered_df = filtered_df[filtered_df['severity'].str.lower() == slots['severity'].lower()]
            if 'failure_phase' in slots:
                filtered_df = filtered_df[filtered_df['failure_phase'].str.lower() == slots['failure_phase'].lower()]
            if 'propellant_type' in slots:
                filtered_df = filtered_df[filtered_df['propellant_type'].str.lower() == slots['propellant_type'].lower()]
            
            count = len(filtered_df)
            
            # Build context string
            context_parts = []
            if 'country' in slots:
                context_parts.append(f"in {slots['country']}")
            if 'vehicle' in slots:
                context_parts.append(f"for {slots['vehicle']}")
            if 'decade' in slots:
                context_parts.append(f"in the {slots['decade']}")
            if 'failure_type' in slots:
                context_parts.append(f"with {slots['failure_type']}")
            if 'severity' in slots:
                context_parts.append(f"with {slots['severity']} severity")
            if 'failure_phase' in slots:
                context_parts.append(f"during {slots['failure_phase']}")
            if 'propellant_type' in slots:
                context_parts.append(f"using {slots['propellant_type']}")
            
            context = " ".join(context_parts) if context_parts else "matching your criteria"
            answer = f"There are {count} rocket failure incidents {context}."
        
        return answer
    
    def extract_comparison_query(self, results: List[Dict], intent_data: Dict, df: pd.DataFrame) -> str:
        """
        Extract answer for COMPARISON_QUERY intent.
        
        Args:
            results: Retrieved records
            intent_data: Intent classification result
            df: Full dataframe for aggregation
            
        Returns:
            Formatted answer string with comparison
        """
        slots = intent_data.get('slots', {})
        query_lower = intent_data.get('query', '').lower()
        
        # If no specific entities, do country-level aggregation
        if not slots.get('country') and not slots.get('vehicle'):
            counts = df.groupby('country').size().sort_values(ascending=False)
            top = counts.head(5)
            lines = [f"  {c}: {n} incidents" for c, n in top.items()]
            return "Top countries by rocket failure incidents:\n" + "\n".join(lines)
        
        # Country vs country comparison
        countries = []
        if 'country' in slots:
            countries.append(slots['country'])
        # Also check for multiple countries in query (e.g., "usa and uk")
        import re
        and_pattern = re.compile(r'\b([A-Za-z]+)\s+and\s+([A-Za-z]+)\b', re.IGNORECASE)
        and_match = and_pattern.search(query_lower)
        if and_match:
            countries.extend([and_match.group(1), and_match.group(2)])
        
        if len(countries) >= 2:
            results_text = []
            for c in countries:
                subset = df[df['country'].str.lower() == c.lower()]
                results_text.append(
                    f"  {c}: {len(subset)} incidents | "
                    f"Most common failure: {subset['failure_type'].mode()[0] if len(subset) > 0 else 'N/A'}"
                )
            return "Comparison:\n" + "\n".join(results_text)
        
        # For single country comparison
        if 'country' in slots:
            country = slots['country']
            country_counts = df['country'].value_counts()
            
            if country.lower() in [c.lower() for c in country_counts.index]:
                count = country_counts[country_counts.index.str.lower() == country.lower()].values[0]
                max_count = country_counts.max()
                max_country = country_counts.idxmax()
                
                if count == max_count:
                    answer = f"{country} has the most failures ({count} incidents), tied with other countries."
                else:
                    answer = f"{country} has {count} failure incidents. The country with the most failures is {max_country} with {max_count} incidents."
            else:
                answer = f"No data found for country: {country}"
            return answer
        
        # For vehicle comparison
        if 'vehicle' in slots:
            vehicle = slots['vehicle']
            vehicle_counts = df['launch_vehicle'].value_counts()
            
            if vehicle.lower() in [v.lower() for v in vehicle_counts.index]:
                count = vehicle_counts[vehicle_counts.index.str.lower() == vehicle.lower()].values[0]
                max_count = vehicle_counts.max()
                max_vehicle = vehicle_counts.idxmax()
                
                if count == max_count:
                    answer = f"{vehicle} has the most failures ({count} incidents), tied with other vehicles."
                else:
                    answer = f"{vehicle} has {count} failure incidents. The vehicle with the most failures is {max_vehicle} with {max_count} incidents."
            else:
                answer = f"No data found for vehicle: {vehicle}"
            return answer
        
        return "Please specify which countries or vehicles you want to compare."
    
    def extract_trend_query(self, results: List[Dict], intent_data: Dict, df: pd.DataFrame) -> str:
        """
        Extract answer for TREND_QUERY intent.
        
        Args:
            results: Retrieved records
            intent_data: Intent classification result
            df: Full dataframe for aggregation
            
        Returns:
            Formatted answer string with trend analysis
        """
        slots = intent_data.get('slots', {})
        
        # Analyze by decade
        if 'decade' in slots:
            decade = slots['decade']
            decade_df = df[df['decade'].str.lower() == decade.lower()]
            
            if len(decade_df) == 0:
                return f"No data found for decade: {decade}"
            
            # Find most common failure type in that decade
            failure_counts = decade_df['failure_type'].value_counts()
            top_failure = failure_counts.idxmax()
            top_count = failure_counts.max()
            
            answer = f"In the {decade}, the most common failure type was '{top_failure}' with {top_count} incidents ({top_count/len(decade_df)*100:.1f}% of all failures in that decade)."
        
        # Analyze overall trend
        else:
            failure_counts = df['failure_type'].value_counts()
            top_failure = failure_counts.idxmax()
            top_count = failure_counts.max()
            
            answer = f"The most common failure type overall is '{top_failure}' with {top_count} incidents ({top_count/len(df)*100:.1f}% of all failures)."
        
        return answer
    
    def extract_statistical_query(self, results: List[Dict], intent_data: Dict, df: pd.DataFrame) -> str:
        """
        Extract answer for STATISTICAL_QUERY intent.
        
        Args:
            results: Retrieved records
            intent_data: Intent classification result
            df: Full dataframe for aggregation
            
        Returns:
            Formatted answer string with statistics
        """
        slots = intent_data.get('slots', {})
        query_lower = intent_data.get('query', '').lower()
        
        # Average altitude
        if 'average' in query_lower and 'altitude' in query_lower:
            avg_altitude = df['altitude_km'].mean()
            answer = f"The average altitude at failure is {avg_altitude:.2f} km."
        
        # Most common severity
        elif 'most common' in query_lower and 'severity' in query_lower:
            severity_counts = df['severity'].value_counts()
            top_severity = severity_counts.idxmax()
            answer = f"The most common severity level is '{top_severity}' with {severity_counts[top_severity]} incidents."
        
        # Average time to failure
        elif 'average' in query_lower and 'time' in query_lower:
            avg_time = df['time_to_failure_sec'].mean()
            answer = f"The average time to failure is {avg_time:.2f} seconds."
        
        # General statistics
        else:
            answer = f"Dataset statistics:\n"
            answer += f"- Total incidents: {len(df)}\n"
            answer += f"- Average altitude: {df['altitude_km'].mean():.2f} km\n"
            answer += f"- Average time to failure: {df['time_to_failure_sec'].mean():.2f} sec\n"
            answer += f"- Most common severity: {df['severity'].mode()[0]}\n"
            answer += f"- Most common failure type: {df['failure_type'].mode()[0]}"
        
        return answer
    
    def extract_definition_query(self, results: List[Dict], intent_data: Dict) -> str:
        """
        Extract answer for DEFINITION_QUERY intent.
        
        Args:
            results: Retrieved records
            intent_data: Intent classification result
            
        Returns:
            Formatted answer string with definition
        """
        if not results:
            return "No matching information found for your query."
        
        slots = intent_data.get('slots', {})
        
        # If a specific term was mentioned, try to find it in the results
        if 'failure_type' in slots:
            failure_type = slots['failure_type']
            matching_results = [r for r in results if r['failure_type'].lower() == failure_type.lower()]
            if matching_results:
                top_result = matching_results[0]
                answer = f"Definition for {failure_type}:\n\n"
                answer += top_result.get('cause_description', 'No description available.')
                source = f"\n\nSource: incident_id {top_result['incident_id']}, {top_result['launch_vehicle']}, {top_result['date']}"
                answer += source
                return answer
        
        # Otherwise, return the top result's cause description
        top_result = results[0]
        answer = f"Based on the incident data:\n\n"
        answer += top_result.get('cause_description', 'No description available.')
        source = f"\n\nSource: incident_id {top_result['incident_id']}, {top_result['launch_vehicle']}, {top_result['date']}"
        answer += source
        
        return answer
    
    def extract_answer(
        self,
        results: List[Dict],
        intent_data: Dict,
        df: pd.DataFrame
    ) -> str:
        """
        Extract answer based on intent type and retrieved results.
        
        Args:
            results: Retrieved records from retrieval engine
            intent_data: Intent classification result
            df: Full dataframe for aggregation queries
            
        Returns:
            Formatted answer string
        """
        intent = intent_data.get('intent', 'FACTUAL_LOOKUP')
        
        logger.info(f"Extracting answer for intent: {intent}")
        
        # Route to appropriate extraction method
        if intent == 'FACTUAL_LOOKUP':
            return self.extract_factual_lookup(results, intent_data)
        elif intent == 'FILTER_QUERY':
            return self.extract_filter_query(results, intent_data)
        elif intent == 'COUNT_QUERY':
            return self.extract_count_query(results, intent_data, df)
        elif intent == 'COMPARISON_QUERY':
            return self.extract_comparison_query(results, intent_data, df)
        elif intent == 'TREND_QUERY':
            return self.extract_trend_query(results, intent_data, df)
        elif intent == 'STATISTICAL_QUERY':
            return self.extract_statistical_query(results, intent_data, df)
        elif intent == 'DEFINITION_QUERY':
            return self.extract_definition_query(results, intent_data)
        else:
            # Default to factual lookup
            return self.extract_factual_lookup(results, intent_data)


if __name__ == "__main__":
    # Test the answer extractor
    extractor = AnswerExtractor()
    
    # Create sample data
    sample_record = {
        'incident_id': 'RFI-0001',
        'date': '1971-01-24',
        'launch_vehicle': 'N1',
        'country': 'USSR',
        'launch_site': 'Baikonur Cosmodrome',
        'failure_type': 'Nozzle Erosion',
        'failure_category': 'Propulsion',
        'failure_phase': 'Ascent',
        'severity': 'Major',
        'propellant_type': 'LOX/LH2',
        'altitude_km': 184.1,
        'time_to_failure_sec': 324.8,
        'mission_success': 0,
        'cause_description': 'Combustion gases erode nozzle material faster than designed, leading to thrust loss and asymmetric exhaust causing loss of vehicle control.',
        'notes': 'Incident recorded at Baikonur Cosmodrome.'
    }
    
    print("="*60)
    print("ANSWER EXTRACTOR TEST")
    print("="*60)
    
    # Test factual lookup
    intent_data = {'intent': 'FACTUAL_LOOKUP', 'slots': {}, 'query': 'What caused the N1 rocket failure?'}
    answer = extractor.extract_factual_lookup([sample_record], intent_data)
    print(f"\nFACTUAL_LOOKUP:\n{answer}")
