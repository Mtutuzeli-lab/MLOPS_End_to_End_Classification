"""
Data Transformation Component for MLOps Pipeline
This module handles feature engineering, missing value imputation, and data preprocessing
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Networksecurity.Constants.training_pipeline import TARGET_COLUMN
from Networksecurity.Constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from Networksecurity.Entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from Networksecurity.Entity.config_entity import DataTransformationConfig
from Networksecurity.exception.exception import NetworkSecurityException 
from Networksecurity.logging.logger import logging
from Networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    """
    Data Transformation Component
    
    Purpose:
    --------
    This component transforms raw validated data into ML-ready format by:
    1. Handling missing values using KNN Imputation
    2. Separating features from target variable
    3. Encoding target labels (converting -1 to 0 for binary classification)
    4. Creating numpy arrays for efficient model training
    5. Saving preprocessor object for future predictions
    
    Why KNN Imputation?
    ------------------
    - Preserves data patterns better than mean/median imputation
    - Uses similar records (neighbors) to estimate missing values
    - Works well with mixed numerical/categorical data
    - Better for machine learning than dropping rows
    
    Input:
    ------
    - Validated train/test CSV files from Data Validation Component
    
    Output:
    -------
    - Transformed numpy arrays (train.npy, test.npy)
    - Preprocessor pickle file (for inference pipeline)
    """
    
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Initialize Data Transformation Component
        
        Args:
            data_validation_artifact: Output from data validation step
            data_transformation_config: Configuration for transformation
        """
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            logging.info("Data Transformation Component initialized")
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
        
    def get_data_transformer_object(self, dataframe: pd.DataFrame) -> Pipeline:
        """
        Create preprocessing pipeline with encoding and KNN Imputer
        
        Steps:
        ------
        1. Identify categorical and numerical columns
        2. OneHotEncode categorical features
        3. StandardScale numerical features
        4. Apply KNN Imputation on all features
        
        Returns:
            ColumnTransformer: Sklearn transformer with encoding and imputation
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        
        try:
            # Identify categorical and numerical columns
            categorical_features = dataframe.select_dtypes(include=['object']).columns.tolist()
            numerical_features = dataframe.select_dtypes(exclude=['object']).columns.tolist()
            
            # Remove target column if present
            if TARGET_COLUMN in categorical_features:
                categorical_features.remove(TARGET_COLUMN)
            if TARGET_COLUMN in numerical_features:
                numerical_features.remove(TARGET_COLUMN)
            
            logging.info(f"Categorical features: {categorical_features}")
            logging.info(f"Numerical features: {numerical_features}")
            
            # Create transformers for categorical and numerical features
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            # Combine transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features),
                    ('num', numerical_transformer, numerical_features)
                ],
                remainder='passthrough'
            )
            
            # Create final pipeline with preprocessor and imputer
            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('imputer', KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS))
            ])
            
            logging.info("Created preprocessing pipeline with encoding, scaling, and imputation")
            
            return final_pipeline
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Main method to execute data transformation
        
        Process:
        --------
        1. Load validated train and test data
        2. Separate features (X) from target (y)
        3. Convert target labels: -1 -> 0 (for binary classification)
        4. Fit KNN imputer on training data
        5. Transform both train and test data
        6. Combine transformed features with target
        7. Save numpy arrays and preprocessor object
        
        Returns:
            DataTransformationArtifact: Paths to transformed data and preprocessor
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        
        try:
            logging.info("Starting data transformation")
            
            # Step 1: Load validated data
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )
            
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            # Step 2: Separate features and target for training data
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            # Step 3: Encode target variable (convert -1 to 0 for binary classification)
            target_feature_train_df = target_feature_train_df.replace(-1, 0)
            
            logging.info(f"Training features shape: {input_feature_train_df.shape}")
            logging.info(f"Training target shape: {target_feature_train_df.shape}")

            # Step 4: Separate features and target for testing data
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)
            
            logging.info(f"Testing features shape: {input_feature_test_df.shape}")
            logging.info(f"Testing target shape: {target_feature_test_df.shape}")

            # Step 5: Get preprocessor pipeline (pass training dataframe to identify column types)
            preprocessor = self.get_data_transformer_object(input_feature_train_df)

            # Step 6: Fit preprocessor on training data
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            logging.info("Preprocessor fitted on training data")
            
            # Step 7: Transform train and test features
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            
            logging.info("Features transformed using fitted preprocessor")

            # Step 8: Combine transformed features with target variable
            # np.c_ concatenates arrays column-wise
            train_arr = np.c_[
                transformed_input_train_feature, 
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                transformed_input_test_feature, 
                np.array(target_feature_test_df)
            ]
            
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")

            # Step 9: Save numpy arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, 
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            
            # Step 10: Save preprocessor object for inference pipeline
            save_object(
                self.data_transformation_config.transformed_object_file_path, 
                preprocessor_object
            )
            
            # Also save to final_model directory for deployment
            save_object(
                "final_model/preprocessor.pkl", 
                preprocessor_object
            )
            
            logging.info("Preprocessor object saved successfully")

            # Step 11: Create and return artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            logging.info("Data transformation completed successfully")
            logging.info(f"Transformation Artifact: {data_transformation_artifact}")
            
            return data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
