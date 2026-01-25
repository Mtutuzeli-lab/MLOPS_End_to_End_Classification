"""
Batch Prediction Script for Telco Customer Churn
Loads customer data and makes batch predictions using deployed model
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from google.cloud import bigquery

# Configuration
PROJECT_ID = "mlops-churn-prediction-484819"
REGION = "us-central1"
ENDPOINT_ID = "telco-churn-endpoint"  # Get from deployment
DATASET_ID = "telco_churn_dataset"
TABLE_ID = "customer_churn_cleaned"  # Use cleaned table loaded by etl_to_bigquery.py
OUTPUT_TABLE_ID = "churn_predictions"
USE_LOCAL_MODEL = True

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_credentials():
    """Setup GCP credentials"""
    credentials_path = "config/service-account-key.json"
    if not os.path.exists(credentials_path):
        logger.error(f"Credentials not found at {credentials_path}")
        sys.exit(1)
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    logger.info("GCP credentials configured")


def load_batch_data(limit: int = None) -> pd.DataFrame:
    """
    Load customer data from BigQuery for batch prediction
    
    Args:
        limit: Maximum number of records to load (None = all)
        
    Returns:
        DataFrame with customer features
    """
    logger.info("=" * 80)
    logger.info("STEP 1: LOADING BATCH DATA FROM BIGQUERY")
    logger.info("=" * 80)
    
    try:
        client = bigquery.Client(project=PROJECT_ID)
        
        # Query to load customer data
        query = f"""
        SELECT 
            gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService,
            OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies,
            Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE Churn IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Executing query...")
        df = client.query(query).to_dataframe()
        
        logger.info(f"Loaded {len(df)} records from BigQuery")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess batch data using the trained preprocessor
    
    Args:
        df: Raw customer data
        
    Returns:
        Preprocessed data ready for prediction
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: PREPROCESSING BATCH DATA")
    logger.info("=" * 80)
    
    try:
        # Load the preprocessor that was trained during pipeline
        from Networksecurity.utils.main_utils.utils import load_object
        
        preprocessor_path = "final_model/preprocessor.pkl"
        
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessor not found at {preprocessor_path}")
            logger.info("Run train_pipeline.py first to create preprocessor")
            sys.exit(1)
        
        preprocessor = load_object(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        
        # Apply preprocessing
        df_processed = preprocessor.transform(df)
        
        logger.info(f"Preprocessing completed")
        logger.info(f"Processed data shape: {df_processed.shape}")
        
        return df_processed
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise


def make_batch_predictions(X: pd.DataFrame, batch_size: int = 100) -> np.ndarray:
    """
    Make batch predictions using deployed model endpoint
    
    Args:
        X: Preprocessed features
        batch_size: Batch size for predictions
        
    Returns:
        Array of predictions (0/1 for churn likelihood)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: MAKING BATCH PREDICTIONS")
    logger.info("=" * 80)
    
    try:
        if USE_LOCAL_MODEL:
            from Networksecurity.utils.main_utils.utils import load_object
            model_path = "final_model/model.pkl"
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                sys.exit(1)
            model = load_object(model_path)

            logger.info("Using local model for predictions")
            # If model supports predict_proba, use probability of class 1
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                # Handle binary classifiers returning shape (n_samples, 2)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    return probs[:, 1]
                # Some models may return (n_samples,) probabilities
                return probs
            else:
                preds = model.predict(X)
                # Convert to float probabilities if possible
                return preds.astype(float)
        from google.cloud import aiplatform
        
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        logger.info(f"Getting endpoint: {ENDPOINT_ID}")
        
        # Get the deployed endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f"display_name={ENDPOINT_ID}"
        )
        
        if not endpoints:
            logger.error(f"Endpoint '{ENDPOINT_ID}' not found")
            logger.info("Deploy model first using: python deploy_to_vertex_ai.py")
            sys.exit(1)
        
        endpoint = endpoints[0]
        logger.info(f"Endpoint found: {endpoint.resource_name}")
        
        # Make predictions in batches
        all_predictions = []
        total_records = len(X)

        for i in range(0, total_records, batch_size):
            batch_end = min(i + batch_size, total_records)
            batch = X[i:batch_end].tolist()

            logger.info(f"Predicting batch {i//batch_size + 1} ({i}-{batch_end}/{total_records})")

            # Call endpoint
            predictions = endpoint.predict(instances=batch)
            all_predictions.extend(predictions.predictions)

        logger.info(f"Predictions completed for {total_records} records")

        return np.array(all_predictions)
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise


def save_predictions(df_original: pd.DataFrame, predictions: np.ndarray) -> None:
    """
    Save predictions back to BigQuery and CSV
    
    Args:
        df_original: Original customer data
        predictions: Model predictions
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: SAVING PREDICTIONS")
    logger.info("=" * 80)
    
    try:
        # Create results dataframe
        df_results = df_original.copy()
        df_results['churn_probability'] = predictions
        df_results['churn_prediction'] = (predictions > 0.5).astype(int)
        df_results['prediction_timestamp'] = datetime.now().isoformat()
        
        # Save to CSV
        csv_path = f"batch_predictions/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        Path("batch_predictions").mkdir(exist_ok=True)
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Predictions saved to CSV: {csv_path}")
        
        # Save to BigQuery
        logger.info("Saving predictions to BigQuery...")
        client = bigquery.Client(project=PROJECT_ID)
        
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{OUTPUT_TABLE_ID}"
        
        job = client.load_table_from_dataframe(
            df_results,
            table_id,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        )
        job.result()
        logger.info(f"Predictions saved to BigQuery table: {table_id}")
        
        # Print summary
        churn_count = (df_results['churn_prediction'] == 1).sum()
        churn_rate = churn_count / len(df_results) * 100
        
        logger.info("\n" + "=" * 80)
        logger.info("BATCH PREDICTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total records processed: {len(df_results)}")
        logger.info(f"Predicted churn count: {churn_count}")
        logger.info(f"Predicted churn rate: {churn_rate:.2f}%")
        logger.info(f"CSV output: {csv_path}")
        logger.info(f"BigQuery table: {table_id}")
        logger.info("=" * 80 + "\n")

        # Model monitoring: generate Evidently report
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
            Path("monitoring_reports").mkdir(exist_ok=True)
            report = Report(metrics=[ClassificationPreset()])
            # Use original features and predictions
            report.run(
                reference_data=df_results,
                current_data=df_results,
                column_mapping={
                    "target": "churn_prediction",
                    "prediction": "churn_probability"
                }
            )
            report_path = f"monitoring_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(report_path)
            logger.info(f"Evidently monitoring report saved: {report_path}")
        except Exception as ev_e:
            logger.info(f"Evidently monitoring skipped or failed: {ev_e}")

        # Optional: log metrics/artifacts to MLflow (e.g., DagsHub) if configured
        try:
            import mlflow
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            token = os.environ.get("MLFLOW_TRACKING_TOKEN")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_experiment("batch_predictions")
            with mlflow.start_run(run_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_metric("records", int(len(df_results)))
                mlflow.log_metric("churn_count", int(churn_count))
                mlflow.log_metric("churn_rate", float(churn_rate))
                mlflow.log_artifact(csv_path)
        except Exception as ml_e:
            logger.info(f"MLflow logging skipped or failed: {ml_e}")
    
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise


def main():
    """Main batch prediction pipeline"""
    try:
        setup_credentials()
        
        # Load data
        df = load_batch_data(limit=1000)  # Load 1000 for testing, remove limit for production
        
        # Preprocess
        X_processed = preprocess_batch_data(df)
        
        # Make predictions
        predictions = make_batch_predictions(X_processed, batch_size=100)
        
        # Save results
        save_predictions(df, predictions)
        
        logger.info("\n" + "=" * 80)
        logger.info("BATCH PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80 + "\n")
    
    except Exception as e:
        logger.error(f"\nBatch prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
