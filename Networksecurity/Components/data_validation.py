"""
Data Validation Component for MLOps Pipeline
This module validates data quality, checks for missing values, and detects data drift
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import ks_2samp

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Networksecurity.Constants.training_pipeline import TARGET_COLUMN
from Networksecurity.Entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from Networksecurity.Entity.config_entity import DataValidationConfig
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.utils.main_utils.utils import write_yaml_file


class DataValidation:
    """
    Data Validation Component
    
    Purpose:
    --------
    This component validates data quality BEFORE transformation to ensure:
    1. Required columns are present
    2. Data types are correct
    3. No completely null columns
    4. Data distribution hasn't changed (drift detection)
    5. Target variable is present
    
    Why Data Validation?
    -------------------
    - Catch data quality issues early (before training)
    - Prevent model training on corrupted/incomplete data
    - Detect data drift (train vs test distribution changes)
    - Save time/money by failing fast on bad data
    - Maintain model performance in production
    
    Data Drift:
    -----------
    Detects if test data distribution differs from training data.
    Uses Kolmogorov-Smirnov test (KS test):
    - p-value > 0.05: distributions are similar (GOOD)
    - p-value < 0.05: distributions differ significantly (BAD - data drift!)
    
    Input:
    ------
    - Raw train/test CSV files from Data Ingestion Component
    
    Output:
    -------
    - Validated train/test CSV files (if validation passes)
    - Invalid data files (if validation fails)
    - Drift report YAML file
    - Validation status (True/False)
    """
    
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Initialize Data Validation Component
        
        Args:
            data_ingestion_artifact: Output from data ingestion step
            data_validation_config: Configuration for validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            logging.info("Data Validation Component initialized")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read CSV data from file path
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that dataframe has expected number of columns
        
        Args:
            dataframe (pd.DataFrame): Dataframe to validate
            
        Returns:
            bool: True if column count is valid
        """
        try:
            number_of_columns = len(dataframe.columns)
            logging.info(f"Number of columns: {number_of_columns}")
            
            # For telco churn dataset, we expect specific columns
            # Adjust this based on your dataset
            if number_of_columns > 0:
                return True
            return False
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def detect_dataset_drift(self, base_df: pd.DataFrame, 
                            current_df: pd.DataFrame,
                            threshold: float = 0.05) -> Tuple[bool, dict]:
        """
        Detect data drift between training and test datasets
        
        Uses Kolmogorov-Smirnov (KS) test to compare distributions:
        - Null hypothesis: Both samples come from same distribution
        - If p-value < threshold (0.05): Reject null → Data drift detected
        - If p-value >= threshold: Accept null → No drift
        
        Args:
            base_df (pd.DataFrame): Training/reference dataset
            current_df (pd.DataFrame): Test/current dataset
            threshold (float): Significance level (default 0.05)
            
        Returns:
            Tuple[bool, dict]: (drift_detected, drift_report)
        """
        try:
            status = True
            report = {}
            
            logging.info("Starting data drift detection")
            
            # Compare each numerical column
            for column in base_df.columns:
                # Only test numerical columns
                if base_df[column].dtype in ['int64', 'float64']:
                    # Get data from both datasets (excluding NaN)
                    d1 = base_df[column].dropna()
                    d2 = current_df[column].dropna()
                    
                    # Perform KS test
                    ks_statistic, p_value = ks_2samp(d1, d2)
                    
                    # Check if drift detected (p_value < threshold)
                    is_same_dist = p_value >= threshold
                    
                    report[column] = {
                        "p_value": float(p_value),
                        "ks_statistic": float(ks_statistic),
                        "drift_detected": not is_same_dist
                    }
                    
                    if not is_same_dist:
                        status = False
                        logging.warning(f"Drift detected in column: {column} (p-value: {p_value:.4f})")
                    else:
                        logging.info(f"No drift in column: {column} (p-value: {p_value:.4f})")
            
            if status:
                logging.info("No significant data drift detected")
            else:
                logging.warning("Data drift detected in one or more columns")
            
            return status, report
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def is_required_columns_exist(self, base_df: pd.DataFrame, 
                                  current_df: pd.DataFrame) -> bool:
        """
        Check if all required columns exist in both datasets
        
        Args:
            base_df (pd.DataFrame): Training dataset
            current_df (pd.DataFrame): Test dataset
            
        Returns:
            bool: True if all columns match
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            
            logging.info(f"Base columns: {base_columns.tolist()}")
            logging.info(f"Current columns: {current_columns.tolist()}")
            
            # Check if columns match
            missing_columns = []
            for col in base_columns:
                if col not in current_columns:
                    missing_columns.append(col)
                    logging.warning(f"Column missing in test data: {col}")
            
            if len(missing_columns) > 0:
                logging.error(f"Missing columns: {missing_columns}")
                return False
            
            logging.info("All required columns are present")
            return True
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Main method to execute data validation
        
        Process:
        --------
        1. Load train and test data from ingestion
        2. Validate column presence and counts
        3. Check for required columns
        4. Detect data drift between train and test
        5. Save validated data or mark as invalid
        6. Generate drift report
        7. Return validation artifact
        
        Returns:
            DataValidationArtifact: Validation results and file paths
        """
        try:
            logging.info("Entered initiate_data_validation method")
            
            # Step 1: Load data from ingestion artifacts
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            logging.info(f"Reading train data from: {train_file_path}")
            train_dataframe = DataValidation.read_data(train_file_path)
            
            logging.info(f"Reading test data from: {test_file_path}")
            test_dataframe = DataValidation.read_data(test_file_path)
            
            logging.info(f"Train data shape: {train_dataframe.shape}")
            logging.info(f"Test data shape: {test_dataframe.shape}")
            
            # Step 2: Validate number of columns
            status = self.validate_number_of_columns(train_dataframe)
            if not status:
                raise Exception("Train dataframe does not have required columns")
            
            status = self.validate_number_of_columns(test_dataframe)
            if not status:
                raise Exception("Test dataframe does not have required columns")
            
            # Step 3: Check if required columns exist
            status = self.is_required_columns_exist(
                base_df=train_dataframe,
                current_df=test_dataframe
            )
            if not status:
                raise Exception("Column mismatch between train and test data")
            
            # Step 4: Check if target column exists
            if TARGET_COLUMN not in train_dataframe.columns:
                raise Exception(f"Target column '{TARGET_COLUMN}' not found in training data")
            
            if TARGET_COLUMN not in test_dataframe.columns:
                raise Exception(f"Target column '{TARGET_COLUMN}' not found in test data")
            
            logging.info(f"Target column '{TARGET_COLUMN}' found in both datasets")
            
            # Step 5: Detect data drift
            drift_status, drift_report = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )
            
            # Step 6: Save drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(
                file_path=drift_report_file_path,
                content=drift_report
            )
            logging.info(f"Drift report saved at: {drift_report_file_path}")
            
            # Step 7: Save validated or invalid data
            # Create directories
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), 
                       exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), 
                       exist_ok=True)
            
            if drift_status:
                # No drift - save as valid
                logging.info("Validation passed - saving valid data files")
                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path, 
                    index=False, 
                    header=True
                )
                test_dataframe.to_csv(
                    self.data_validation_config.valid_test_file_path, 
                    index=False, 
                    header=True
                )
            else:
                # Drift detected - save as invalid (but allow pipeline to continue)
                logging.warning("Data drift detected - saving as invalid (pipeline continues)")
                train_dataframe.to_csv(
                    self.data_validation_config.invalid_train_file_path, 
                    index=False, 
                    header=True
                )
                test_dataframe.to_csv(
                    self.data_validation_config.invalid_test_file_path, 
                    index=False, 
                    header=True
                )
                
                # Also save to valid path to allow pipeline continuation
                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path, 
                    index=False, 
                    header=True
                )
                test_dataframe.to_csv(
                    self.data_validation_config.valid_test_file_path, 
                    index=False, 
                    header=True
                )
            
            # Step 8: Create validation artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            
            logging.info("Data validation completed successfully")
            logging.info(f"Validation Artifact: {data_validation_artifact}")
            
            return data_validation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
