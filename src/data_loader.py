"""
Data Loader module for Rocket Failure Chatbot
Handles loading and validation of the rocket failures dataset.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import CSV_PATH, LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and validates the rocket failures dataset."""
    
    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            csv_path: Path to the CSV file. If None, uses default from config.
        """
        self.csv_path = Path(csv_path or CSV_PATH)
        self.data = None
        logger.info(f"DataLoader initialized with path: {self.csv_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the rocket failures CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is empty or malformed
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
        
        logger.info(f"Loading data from {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.data)} records")
        
        return self.data
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate the loaded dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        expected_columns = [
            'incident_id', 'date', 'year', 'month', 'launch_vehicle', 'country',
            'launch_site', 'failure_type', 'failure_category', 'failure_phase',
            'severity', 'propellant_type', 'altitude_km', 'time_to_failure_sec',
            'mission_success', 'cause_description', 'notes', 'decade'
        ]
        
        # Check if all expected columns exist
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Check if dataframe is empty
        if len(df) == 0:
            return False, "DataFrame is empty"
        
        # Check for duplicate incident IDs
        duplicate_ids = df['incident_id'].duplicated().sum()
        if duplicate_ids > 0:
            logger.warning(f"Found {duplicate_ids} duplicate incident IDs")
        
        # Check for null values in critical columns
        critical_cols = ['incident_id', 'date', 'launch_vehicle', 'failure_type']
        null_counts = df[critical_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values in critical columns:\n{null_counts}")
        
        logger.info("Data validation passed")
        return True, "Data validation passed"
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        return info
    
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load and validate data in one step.
        
        Returns:
            pd.DataFrame: Validated dataset
        """
        df = self.load_data()
        is_valid, message = self.validate_data(df)
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")
        
        logger.info("Data loaded and validated successfully")
        return df


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.load_and_validate()
    info = loader.get_data_info(df)
    
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    print(f"Shape: {info['shape']}")
    print(f"Columns: {len(info['columns'])}")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")
    print("\nColumn Types:")
    for col, dtype in info['dtypes'].items():
        print(f"  {col}: {dtype}")
    print("\nNull Counts:")
    for col, count in info['null_counts'].items():
        if count > 0:
            print(f"  {col}: {count}")
