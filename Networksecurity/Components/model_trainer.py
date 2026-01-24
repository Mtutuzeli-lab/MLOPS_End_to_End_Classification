"""
Model Training Component for MLOps Pipeline
This module handles model training, evaluation, and model selection
"""

import sys
import os
import numpy as np
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Networksecurity.Entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from Networksecurity.Entity.config_entity import ModelTrainerConfig
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.utils.main_utils.utils import (
    load_numpy_array_data,
    save_object
)


class ModelTrainer:
    """
    Model Training Component
    
    Trains multiple ML models and selects the best one
    """
    
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("Model Trainer Component initialized")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        try:
            train_report = {}
            test_report = {}
            
            logging.info(f"Evaluating {len(models)} models")
            
            for model_name, model in models.items():
                logging.info(f"\n{'='*50}")
                logging.info(f"Training: {model_name}")
                logging.info(f"{'='*50}")
                
                # Start MLflow run for this model
                with mlflow.start_run(run_name=model_name, nested=True):
                    # Log model parameters
                    mlflow.log_params(model.get_params())
                    
                    # Train model
                    model.fit(X_train, y_train)
                    logging.info(f"{model_name} training completed")
                    
                    # Predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    # Calculate metrics for training data
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
                    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
                    train_roc_auc = roc_auc_score(y_train, y_train_pred)
                    
                    # Calculate metrics for test data
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
                    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                    test_roc_auc = roc_auc_score(y_test, y_test_pred)
                    
                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        'train_accuracy': train_accuracy,
                        'train_f1_score': train_f1,
                        'train_precision': train_precision,
                        'train_recall': train_recall,
                        'train_roc_auc': train_roc_auc,
                        'test_accuracy': test_accuracy,
                        'test_f1_score': test_f1,
                        'test_precision': test_precision,
                        'test_recall': test_recall,
                        'test_roc_auc': test_roc_auc
                    })
                
                # Store results
                train_report[model_name] = {
                    'accuracy': train_accuracy,
                    'f1_score': train_f1,
                    'precision': train_precision,
                    'recall': train_recall,
                    'roc_auc_score': train_roc_auc
                }
                
                test_report[model_name] = {
                    'accuracy': test_accuracy,
                    'f1_score': test_f1,
                    'precision': test_precision,
                    'recall': test_recall,
                    'roc_auc_score': test_roc_auc
                }
                
                # Log results
                logging.info(f"\n{model_name} - Training Metrics:")
                logging.info(f"  Accuracy:  {train_accuracy:.4f}")
                logging.info(f"  F1-Score:  {train_f1:.4f}")
                logging.info(f"  Precision: {train_precision:.4f}")
                logging.info(f"  Recall:    {train_recall:.4f}")
                logging.info(f"  ROC-AUC:   {train_roc_auc:.4f}")
                
                logging.info(f"\n{model_name} - Testing Metrics:")
                logging.info(f"  Accuracy:  {test_accuracy:.4f}")
                logging.info(f"  F1-Score:  {test_f1:.4f}")
                logging.info(f"  Precision: {test_precision:.4f}")
                logging.info(f"  Recall:    {test_recall:.4f}")
                logging.info(f"  ROC-AUC:   {test_roc_auc:.4f}")
                
                # Check for overfitting
                overfitting = train_accuracy - test_accuracy
                if overfitting > 0.1:
                    logging.warning(f"⚠️  Potential overfitting detected! Train-Test gap: {overfitting:.4f}")
            
            return train_report, test_report
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def get_best_model(self, train_report, test_report, models):
        try:
            best_model_name = None
            best_model_score = 0
            best_model = None
            
            logging.info("\n" + "="*50)
            logging.info("MODEL COMPARISON - Test F1-Scores:")
            logging.info("="*50)
            
            for model_name in test_report.keys():
                f1 = test_report[model_name]['f1_score']
                accuracy = test_report[model_name]['accuracy']
                
                logging.info(f"{model_name:25} - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
                
                if f1 > best_model_score:
                    best_model_score = f1
                    best_model_name = model_name
                    best_model = models[model_name]
            
            logging.info("="*50)
            logging.info(f"BEST MODEL: {best_model_name}")
            logging.info(f"   F1-Score: {best_model_score:.4f}")
            logging.info(f"   Accuracy: {test_report[best_model_name]['accuracy']:.4f}")
            logging.info("="*50)
            
            return best_model_name, best_model, best_model_score
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_model_trainer(self):
        try:
            logging.info("Entered initiate_model_trainer method")
            
            # Set up MLflow
            mlflow.set_experiment("Telco_Churn_Model_Training")
            
            with mlflow.start_run(run_name="Pipeline_Run"):
                # Step 1: Load transformed data
                logging.info("Loading transformed train and test arrays")
                train_arr = load_numpy_array_data(
                    self.data_transformation_artifact.transformed_train_file_path
                )
                test_arr = load_numpy_array_data(
                    self.data_transformation_artifact.transformed_test_file_path
                )
                
                logging.info(f"Train array shape: {train_arr.shape}")
                logging.info(f"Test array shape: {test_arr.shape}")
                
                # Step 2: Split features and target
                X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
                X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
                
                logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
                
                # Log dataset information to MLflow
                mlflow.log_params({
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'n_features': X_train.shape[1]
                })
            
                # Step 3: Check for class imbalance and apply SMOTE if needed
                unique, counts = np.unique(y_train, return_counts=True)
                class_distribution = dict(zip(unique, counts))
                logging.info(f"Original class distribution: {class_distribution}")
                
                # Calculate imbalance ratio
                minority_class_count = min(counts)
                majority_class_count = max(counts)
                imbalance_ratio = minority_class_count / majority_class_count
                
                logging.info(f"Imbalance ratio: {imbalance_ratio:.4f}")
                mlflow.log_param('imbalance_ratio', imbalance_ratio)
                
                # Apply SMOTE if data is imbalanced (ratio < 0.5)
                if imbalance_ratio < 0.5:
                    logging.info("Imbalanced dataset detected! Applying SMOTE...")
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                    
                    unique_balanced, counts_balanced = np.unique(y_train_balanced, return_counts=True)
                    balanced_distribution = dict(zip(unique_balanced, counts_balanced))
                    logging.info(f"Balanced class distribution: {balanced_distribution}")
                    logging.info(f"SMOTE applied: {X_train.shape[0]} -> {X_train_balanced.shape[0]} samples")
                    
                    mlflow.log_param('smote_applied', True)
                    mlflow.log_param('train_samples_after_smote', X_train_balanced.shape[0])
                    
                    # Use balanced data for training
                    X_train = X_train_balanced
                    y_train = y_train_balanced
                else:
                    logging.info("Dataset is reasonably balanced. Skipping SMOTE.")
                    mlflow.log_param('smote_applied', False)
            
                # Step 4: Initialize models with tuned hyperparameters
                models = {
                    "Logistic Regression": LogisticRegression(
                        C=0.1, 
                        max_iter=100, 
                        penalty='l1', 
                        solver='saga',
                        random_state=42
                    ),
                    "Decision Tree": DecisionTreeClassifier(
                        criterion='entropy',
                        max_depth=10,
                        min_samples_leaf=8,
                        min_samples_split=20,
                        random_state=42
                    ),
                    "Random Forest": RandomForestClassifier(
                        max_depth=None,
                        max_features='log2',
                        min_samples_split=2,
                        n_estimators=100,
                        random_state=42
                    ),
                    "Gradient Boosting": GradientBoostingClassifier(
                        criterion='squared_error',
                        loss='log_loss',
                        max_depth=8,
                        min_samples_split=2,
                        n_estimators=100,
                        random_state=42
                    ),
                    "AdaBoost": AdaBoostClassifier(
                        learning_rate=1,
                        n_estimators=100,
                        random_state=42
                    )
                }
            
                # Step 5: Train and evaluate all models
                train_report, test_report = self.evaluate_model(
                    X_train, y_train, X_test, y_test, models
                )
                
                # Step 6: Select best model
                best_model_name, best_model, best_model_score = self.get_best_model(
                    train_report, test_report, models
                )
                
                # Log best model info to parent run
                mlflow.log_param('best_model', best_model_name)
                mlflow.log_metric('best_model_f1_score', best_model_score)
            
                # Step 7: Check if best model meets minimum threshold
                if best_model_score < self.model_trainer_config.expected_accuracy:
                    logging.error(f"No model met the expected accuracy threshold")
                    logging.error(f"   Best score: {best_model_score:.4f}")
                    logging.error(f"   Required:   {self.model_trainer_config.expected_accuracy:.4f}")
                    raise Exception(
                        f"No model achieved expected accuracy of "
                        f"{self.model_trainer_config.expected_accuracy}. "
                        f"Best was {best_model_score:.4f}"
                    )
                
                logging.info(f"Best model meets required threshold!")
                
                # Step 8: Save best model
                logging.info(f"Saving best model: {best_model_name}")
                
                # Save to artifacts directory (versioned)
                save_object(
                    self.model_trainer_config.trained_model_file_path,
                    best_model
                )
                
                # Also save to final_model directory (for deployment)
                save_object("final_model/model.pkl", best_model)
                
                logging.info(f"Model saved successfully")
            
                # Step 9: Create metric artifacts
                train_metric_artifact = ClassificationMetricArtifact(
                    f1_score=train_report[best_model_name]['f1_score'],
                    precision_score=train_report[best_model_name]['precision'],
                    recall_score=train_report[best_model_name]['recall'],
                    roc_auc_score=train_report[best_model_name]['roc_auc_score']
                )
                
                test_metric_artifact = ClassificationMetricArtifact(
                    f1_score=test_report[best_model_name]['f1_score'],
                    precision_score=test_report[best_model_name]['precision'],
                    recall_score=test_report[best_model_name]['recall'],
                    roc_auc_score=test_report[best_model_name]['roc_auc_score']
                )
                
                # Step 10: Create model trainer artifact
                model_trainer_artifact = ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    train_metric_artifact=train_metric_artifact,
                    test_metric_artifact=test_metric_artifact
                )
                
                logging.info("Model training completed successfully")
                logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
                
                return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
