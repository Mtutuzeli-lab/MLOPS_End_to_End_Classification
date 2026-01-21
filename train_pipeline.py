"""
Training Pipeline - End-to-End ML Pipeline Execution
This script runs the complete MLOps pipeline: Ingestion → Validation → Transformation → Training
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from Networksecurity.Components.data_ingestion import DataIngestion, DataIngestionConfig
from Networksecurity.Components.data_validation import DataValidation
from Networksecurity.Components.data_transformation import DataTransformation
from Networksecurity.Components.model_trainer import ModelTrainer

from Networksecurity.Entity.config_entity import (
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging


def start_training_pipeline():
    """
    Execute the complete training pipeline
    
    Pipeline Steps:
    1. Data Ingestion - Load from BigQuery and split
    2. Data Validation - Check quality and drift
    3. Data Transformation - Handle missing values, create features
    4. Model Training - Train multiple models and select best
    """
    try:
        logging.info("=" * 80)
        logging.info("STARTING TRAINING PIPELINE")
        logging.info("=" * 80)
        
        # Initialize pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        
        # ==================== STEP 1: DATA INGESTION ====================
        logging.info("\n" + "=" * 80)
        logging.info("STEP 1: DATA INGESTION")
        logging.info("=" * 80)
        
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        logging.info(f"Data Ingestion completed")
        logging.info(f"   Train file: {data_ingestion_artifact.trained_file_path}")
        logging.info(f"   Test file: {data_ingestion_artifact.test_file_path}")
        
        # ==================== STEP 2: DATA VALIDATION ====================
        logging.info("\n" + "=" * 80)
        logging.info("STEP 2: DATA VALIDATION")
        logging.info("=" * 80)
        
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        
        logging.info(f"Data Validation completed")
        logging.info(f"   Validation Status: {data_validation_artifact.validation_status}")
        logging.info(f"   Valid train file: {data_validation_artifact.valid_train_file_path}")
        logging.info(f"   Drift report: {data_validation_artifact.drift_report_file_path}")
        
        # ==================== STEP 3: DATA TRANSFORMATION ====================
        logging.info("\n" + "=" * 80)
        logging.info("STEP 3: DATA TRANSFORMATION")
        logging.info("=" * 80)
        
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        logging.info(f"Data Transformation completed")
        logging.info(f"   Transformed train: {data_transformation_artifact.transformed_train_file_path}")
        logging.info(f"   Transformed test: {data_transformation_artifact.transformed_test_file_path}")
        logging.info(f"   Preprocessor: {data_transformation_artifact.transformed_object_file_path}")
        
        # ==================== STEP 4: MODEL TRAINING ====================
        logging.info("\n" + "=" * 80)
        logging.info("STEP 4: MODEL TRAINING")
        logging.info("=" * 80)
        
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        logging.info(f"Model Training completed")
        logging.info(f"   Model saved: {model_trainer_artifact.trained_model_file_path}")
        
        # ==================== FINAL RESULTS ====================
        logging.info("\n" + "=" * 80)
        logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        print("\nTraining Metrics:")
        print(f"   F1-Score:   {model_trainer_artifact.train_metric_artifact.f1_score:.4f}")
        print(f"   Precision:  {model_trainer_artifact.train_metric_artifact.precision_score:.4f}")
        print(f"   Recall:     {model_trainer_artifact.train_metric_artifact.recall_score:.4f}")
        print(f"   ROC-AUC:    {model_trainer_artifact.train_metric_artifact.roc_auc_score:.4f}")
        
        print("\nTesting Metrics:")
        print(f"   F1-Score:   {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")
        print(f"   Precision:  {model_trainer_artifact.test_metric_artifact.precision_score:.4f}")
        print(f"   Recall:     {model_trainer_artifact.test_metric_artifact.recall_score:.4f}")
        print(f"   ROC-AUC:    {model_trainer_artifact.test_metric_artifact.roc_auc_score:.4f}")
        
        print("\nArtifacts Saved:")
        print(f"   Model: {model_trainer_artifact.trained_model_file_path}")
        print(f"   Preprocessor: {data_transformation_artifact.transformed_object_file_path}")
        print(f"   Drift Report: {data_validation_artifact.drift_report_file_path}")
        
        print("\nAll artifacts saved successfully!")
        print("=" * 80)
        
        return model_trainer_artifact
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("STARTING ML TRAINING PIPELINE")
        print("=" * 80)
        print("\nThis will execute:")
        print("  1. Data Ingestion from BigQuery")
        print("  2. Data Validation (drift detection)")
        print("  3. Data Transformation (KNN imputation)")
        print("  4. Model Training (5 models with SMOTE)")
        print("\n" + "=" * 80 + "\n")
        
        model_artifact = start_training_pipeline()
        
        print("\nTraining pipeline completed successfully!")
        print("Check the 'artifacts' folder for all generated files.")
        print("Check the 'logs' folder for detailed execution logs.")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        print("Check logs folder for detailed error information.")
        sys.exit(1)
