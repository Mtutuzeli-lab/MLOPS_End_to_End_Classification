"""
Data Ingestion Component for BigQuery
This module handles data ingestion from BigQuery for the MLOps pipeline
"""

import os
import sys
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging


@dataclass
class DataIngestionConfig:
    """
    Configuration class for Data Ingestion
    """
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    """
    Data Ingestion Component
    Handles loading data from BigQuery and splitting into train/test sets
    """
    
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initialize Data Ingestion Component
        
        Args:
            config (DataIngestionConfig): Configuration for data ingestion
        """
        self.config = config
        logging.info("Data Ingestion Component initialized")
    
    def load_data_from_bigquery(self) -> pd.DataFrame:
        """
        Load data from BigQuery
        
        Returns:
            pd.DataFrame: Loaded dataframe from BigQuery
        """
        try:
            logging.info("Starting data loading from BigQuery")
            
            # Import BigQuery loader
            from Data.bigquery_loader import load_cleaned_data
            
            # Load data from BigQuery
            df = load_cleaned_data()
            
            logging.info(f"Data loaded successfully from BigQuery. Shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logging.error("Error occurred while loading data from BigQuery")
            raise NetworkSecurityException(e, sys)
    
    def save_raw_data(self, df: pd.DataFrame) -> str:
        """
        Save raw data to artifacts folder
        
        Args:
            df (pd.DataFrame): Dataframe to save
            
        Returns:
            str: Path where raw data is saved
        """
        try:
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)
            
            logging.info(f"Raw data saved at: {self.config.raw_data_path}")
            
            return self.config.raw_data_path
            
        except Exception as e:
            logging.error("Error occurred while saving raw data")
            raise NetworkSecurityException(e, sys)
    
    def perform_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df (pd.DataFrame): Dataframe to split
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
        """
        try:
            logging.info("Starting train-test split")
            
            # Perform train-test split
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df['Churn'] if 'Churn' in df.columns else None
            )
            
            logging.info(f"Train set shape: {train_df.shape}")
            logging.info(f"Test set shape: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            logging.error("Error occurred during train-test split")
            raise NetworkSecurityException(e, sys)
    
    def save_train_test_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[str, str]:
        """
        Save train and test datasets
        
        Args:
            train_df (pd.DataFrame): Training dataframe
            test_df (pd.DataFrame): Test dataframe
            
        Returns:
            Tuple[str, str]: Paths where train and test data are saved
        """
        try:
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            
            # Save train data
            train_df.to_csv(self.config.train_data_path, index=False)
            logging.info(f"Train data saved at: {self.config.train_data_path}")
            
            # Save test data
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Test data saved at: {self.config.test_data_path}")
            
            return self.config.train_data_path, self.config.test_data_path
            
        except Exception as e:
            logging.error("Error occurred while saving train/test data")
            raise NetworkSecurityException(e, sys)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Perform basic validation on the loaded data
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            
        Returns:
            bool: True if validation passes
        """
        try:
            logging.info("Starting data validation")
            
            # Check if dataframe is empty
            if df.empty:
                raise ValueError("Dataframe is empty")
            
            # Check for target column
            if 'Churn' not in df.columns:
                raise ValueError("Target column 'Churn' not found in dataframe")
            
            # Log basic statistics
            logging.info(f"Number of rows: {len(df)}")
            logging.info(f"Number of columns: {len(df.columns)}")
            logging.info(f"Missing values:\n{df.isnull().sum()}")
            logging.info(f"Target distribution:\n{df['Churn'].value_counts()}")
            
            logging.info("Data validation completed successfully")
            
            return True
            
        except Exception as e:
            logging.error("Error occurred during data validation")
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Main method to initiate the data ingestion process
        
        Returns:
            Tuple[str, str]: Paths to train and test data
        """
        try:
            logging.info("=" * 50)
            logging.info("Data Ingestion Process Started")
            logging.info("=" * 50)
            
            # Step 1: Load data from BigQuery
            df = self.load_data_from_bigquery()
            
            # Step 2: Validate data
            self.validate_data(df)
            
            # Step 3: Save raw data
            self.save_raw_data(df)
            
            # Step 4: Perform train-test split
            train_df, test_df = self.perform_train_test_split(df)
            
            # Step 5: Save train and test data
            train_path, test_path = self.save_train_test_data(train_df, test_df)
            
            logging.info("=" * 50)
            logging.info("Data Ingestion Process Completed Successfully")
            logging.info("=" * 50)
            
            return train_path, test_path
            
        except Exception as e:
            logging.error("Error occurred in data ingestion process")
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    # Test the data ingestion component
    try:
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        
        # Run data ingestion
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        print(f"\nData Ingestion Completed!")
        print(f"Train Data Path: {train_path}")
        print(f"Test Data Path: {test_path}")
        
    except Exception as e:
        print(f"Error in data ingestion: {str(e)}")
        raise e
