"""
Local Batch Prediction (Testing)
Makes batch predictions using local model without needing deployed endpoint
Useful for testing before deployment
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(csv_path: str = "artifacts/test.csv") -> pd.DataFrame:
    """
    Load test data from local CSV
    
    Args:
        csv_path: Path to test CSV file
        
    Returns:
        DataFrame with customer features
    """
    logger.info("=" * 80)
    logger.info("STEP 1: LOADING TEST DATA")
    logger.info("=" * 80)
    
    try:
        if not os.path.exists(csv_path):
            logger.error(f"Test data not found at {csv_path}")
            logger.info("Run train_pipeline.py first to generate test data")
            sys.exit(1)
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} test records from {csv_path}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def make_local_predictions(model, X: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Make batch predictions using local trained model
    
    Args:
        model: Trained sklearn model
        X: Feature data
        preprocessor: Fitted preprocessor/transformer
        
    Returns:
        Array of predictions
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: MAKING LOCAL BATCH PREDICTIONS")
    logger.info("=" * 80)
    
    try:
        # Separate features and target
        if 'Churn' in X.columns:
            X = X.drop('Churn', axis=1)
        
        # Transform features
        logger.info(f"Transforming {len(X)} records...")
        X_transformed = preprocessor.transform(X)
        
        logger.info(f"Transformed shape: {X_transformed.shape}")
        
        # Make predictions
        logger.info("Making predictions with trained model...")
        predictions = model.predict_proba(X_transformed)[:, 1]  # Probability of churn
        
        logger.info(f"Predictions completed")
        logger.info(f"Min probability: {predictions.min():.4f}")
        logger.info(f"Max probability: {predictions.max():.4f}")
        logger.info(f"Mean probability: {predictions.mean():.4f}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise


def save_local_predictions(df_original: pd.DataFrame, predictions: np.ndarray, 
                          model_name: str = "local") -> str:
    """
    Save predictions to local CSV file
    
    Args:
        df_original: Original data
        predictions: Model predictions
        model_name: Name for output file
        
    Returns:
        Path to saved predictions file
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: SAVING BATCH PREDICTIONS")
    logger.info("=" * 80)
    
    try:
        # Create results
        df_results = df_original.copy()
        df_results['churn_probability'] = predictions
        df_results['churn_prediction'] = (predictions > 0.5).astype(int)
        df_results['prediction_timestamp'] = datetime.now().isoformat()
        
        # Save to CSV
        output_dir = Path("batch_predictions")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = output_dir / f"predictions_{model_name}_{timestamp}.csv"
        
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Predictions saved to: {csv_path}")
        
        # Summary statistics
        churn_count = (df_results['churn_prediction'] == 1).sum()
        churn_rate = churn_count / len(df_results) * 100
        
        logger.info("\n" + "=" * 80)
        logger.info("BATCH PREDICTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total records: {len(df_results)}")
        logger.info(f"Predicted churn: {churn_count}")
        logger.info(f"Predicted retention: {len(df_results) - churn_count}")
        logger.info(f"Churn rate: {churn_rate:.2f}%")
        logger.info(f"Output file: {csv_path}")
        logger.info("=" * 80 + "\n")
        
        # Print sample predictions
        logger.info("Sample predictions (first 10 records):")
        logger.info(df_results[['churn_probability', 'churn_prediction']].head(10).to_string())
        
        return str(csv_path)
    
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise


def main():
    """Main local batch prediction pipeline"""
    try:
        from Networksecurity.utils.main_utils.utils import load_object
        
        # Load trained model and preprocessor
        logger.info("Loading trained model and preprocessor...")
        
        model_path = "final_model/model.pkl"
        preprocessor_path = "final_model/preprocessor.pkl"
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            logger.info("Run train_pipeline.py first to train the model")
            sys.exit(1)
        
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessor not found at {preprocessor_path}")
            logger.info("Run train_pipeline.py first to create preprocessor")
            sys.exit(1)
        
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        
        logger.info(f"Model loaded: {type(model).__name__}")
        logger.info(f"Preprocessor loaded: {type(preprocessor).__name__}")
        
        # Load data
        df = load_test_data()
        
        # Make predictions
        predictions = make_local_predictions(model, df, preprocessor)
        
        # Save results
        save_local_predictions(df, predictions, model_name="local_test")
        
        logger.info("\n" + "=" * 80)
        logger.info("LOCAL BATCH PREDICTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80 + "\n")
        logger.info("Next steps:")
        logger.info("1. Review predictions in batch_predictions/ folder")
        logger.info("2. Deploy model: python deploy_to_vertex_ai.py")
        logger.info("3. Run cloud batch prediction: python batch_prediction.py")
        logger.info("=" * 80 + "\n")
    
    except Exception as e:
        logger.error(f"\nLocal batch prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
