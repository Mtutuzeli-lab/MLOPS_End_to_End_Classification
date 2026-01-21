"""
Configuration Entity Classes
Defines configuration dataclasses for each pipeline component
"""

import os
from dataclasses import dataclass
from datetime import datetime

# Generate timestamp for artifact versioning
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    """
    Configuration for the overall training pipeline
    """
    pipeline_name: str = "networksecurity"
    artifact_dir: str = os.path.join("artifacts", TIMESTAMP)
    timestamp: str = TIMESTAMP


@dataclass
class DataIngestionConfig:
    """
    Configuration for Data Ingestion Component
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_ingestion"
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, "feature_store", "raw_data.csv"
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, "ingested", "train.csv"
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, "ingested", "test.csv"
        )
        self.train_test_split_ratio: float = 0.2


@dataclass
class DataValidationConfig:
    """
    Configuration for Data Validation Component
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_validation"
        )
        self.valid_train_file_path: str = os.path.join(
            self.data_validation_dir, "validated", "train.csv"
        )
        self.valid_test_file_path: str = os.path.join(
            self.data_validation_dir, "validated", "test.csv"
        )
        self.invalid_train_file_path: str = os.path.join(
            self.data_validation_dir, "invalid", "train.csv"
        )
        self.invalid_test_file_path: str = os.path.join(
            self.data_validation_dir, "invalid", "test.csv"
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir, "drift_report", "report.yaml"
        )


@dataclass
class DataTransformationConfig:
    """
    Configuration for Data Transformation Component
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_transformation"
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, "transformed", "train.npy"
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, "transformed", "test.npy"
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir, "transformed_object", "preprocessor.pkl"
        )


@dataclass
class ModelTrainerConfig:
    """
    Configuration for Model Training Component
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "model_trainer"
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, "trained_model", "model.pkl"
        )
        self.expected_accuracy: float = 0.6
        self.model_config_file_path: str = "config/model.yaml"
