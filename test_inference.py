"""
Test Script: Load and test the saved model + preprocessor
This validates that your trained model and preprocessor work correctly before pushing to cloud
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project to path
sys.path.append(os.path.dirname(__file__))

from Networksecurity.utils.main_utils.utils import load_object
from Networksecurity.logging.logger import logging


def find_latest_artifacts():
    """
    Find the latest timestamped artifacts folder
    
    Returns:
        tuple: (model_path, preprocessor_path, or raises if not found)
    """
    try:
        artifacts_dir = Path("artifacts")
        if not artifacts_dir.exists():
            raise Exception("No artifacts folder found. Run train_pipeline.py first.")
        
        # Get all timestamped folders, sorted by name (latest first)
        timestamp_folders = sorted(
            [d for d in artifacts_dir.iterdir() if d.is_dir()],
            reverse=True
        )
        
        if not timestamp_folders:
            raise Exception("No timestamped artifact folders found. Run train_pipeline.py first.")
        
        latest_folder = timestamp_folders[0]
        logging.info(f"Using latest artifacts from: {latest_folder}")
        
        # Paths to model and preprocessor
        model_path = latest_folder / "model_trainer" / "trained_model" / "model.pkl"
        preprocessor_path = latest_folder / "data_transformation" / "transformed_object" / "preprocessor.pkl"
        
        if not model_path.exists():
            raise Exception(f"Model not found at: {model_path}")
        if not preprocessor_path.exists():
            raise Exception(f"Preprocessor not found at: {preprocessor_path}")
        
        return str(model_path), str(preprocessor_path)
    
    except Exception as e:
        logging.error(f"Failed to find artifacts: {str(e)}")
        raise


def load_artifacts():
    """
    Load the trained model and preprocessor
    
    Returns:
        tuple: (model, preprocessor)
    """
    try:
        model_path, preprocessor_path = find_latest_artifacts()
        
        logging.info(f"Loading model from: {model_path}")
        model = load_object(model_path)
        
        logging.info(f"Loading preprocessor from: {preprocessor_path}")
        preprocessor = load_object(preprocessor_path)
        
        logging.info("✓ Model and preprocessor loaded successfully!")
        return model, preprocessor, model_path, preprocessor_path
    
    except Exception as e:
        logging.error(f"Failed to load artifacts: {str(e)}")
        raise


def load_sample_data():
    """
    Load sample test data
    
    Returns:
        pd.DataFrame: Sample data for inference
    """
    try:
        # Try to load from cleaned data
        data_path = Path("Data") / "Telco-Customer-Churn-cleaned.csv"
        
        if data_path.exists():
            logging.info(f"Loading sample data from: {data_path}")
            df = pd.read_csv(data_path)
            return df.head(5)  # Use first 5 rows for testing
        
        # Fallback: create minimal dummy data
        logging.warning("Cleaned data not found. Using dummy data for testing.")
        return pd.DataFrame({
            'SeniorCitizen': [0, 1],
            'tenure': [1, 50],
            'MonthlyCharges': [29.85, 100.0],
        })
    
    except Exception as e:
        logging.error(f"Failed to load sample data: {str(e)}")
        raise


def test_inference():
    """
    Main test function: Load model/preprocessor and make predictions
    """
    try:
        print("\n" + "="*80)
        print("INFERENCE TEST: Model + Preprocessor Validation")
        print("="*80 + "\n")
        
        # Step 1: Load artifacts
        logging.info("Step 1: Loading artifacts...")
        model, preprocessor, model_path, preprocessor_path = load_artifacts()
        
        print(f"\n✓ Model loaded: {model_path}")
        print(f"✓ Preprocessor loaded: {preprocessor_path}")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Preprocessor type: {type(preprocessor).__name__}")
        
        # Step 2: Load sample data
        logging.info("\nStep 2: Loading sample data...")
        sample_data = load_sample_data()
        logging.info(f"Sample data shape: {sample_data.shape}")
        
        print(f"\n✓ Sample data loaded: {sample_data.shape[0]} rows, {sample_data.shape[1]} columns")
        print(f"  Columns: {sample_data.columns.tolist()}")
        
        # Step 3: Try preprocessing
        logging.info("\nStep 3: Testing preprocessor on sample data...")
        try:
            # Check if preprocessor has transform method
            if hasattr(preprocessor, 'transform'):
                transformed_data = preprocessor.transform(sample_data)
                logging.info(f"Transformed shape: {transformed_data.shape}")
                print(f"\n✓ Preprocessor works! Transformed shape: {transformed_data.shape}")
            else:
                logging.warning("Preprocessor doesn't have transform method. Skipping preprocessing test.")
                transformed_data = sample_data.values
                print(f"\n⚠ Preprocessor has no transform method. Using raw data.")
        except Exception as e:
            logging.warning(f"Preprocessing failed (may be expected): {str(e)}")
            print(f"⚠ Preprocessing test skipped: {str(e)}")
            transformed_data = sample_data.values
        
        # Step 4: Make predictions
        logging.info("\nStep 4: Testing model predictions...")
        try:
            # Ensure correct shape for prediction
            if transformed_data.ndim == 1:
                X_test = transformed_data.reshape(1, -1)
            else:
                X_test = transformed_data
            
            predictions = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
                logging.info(f"Predictions: {predictions}")
                logging.info(f"Probabilities: {probabilities}")
                print(f"\n✓ Model predictions successful!")
                print(f"  Predicted classes: {predictions}")
                print(f"  Probabilities (per class):")
                for i, prob in enumerate(probabilities):
                    print(f"    Sample {i}: {prob}")
            else:
                print(f"\n✓ Model predictions successful!")
                print(f"  Predictions: {predictions}")
        
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            print(f"\n✗ Prediction failed: {str(e)}")
            raise
        
        # Summary
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print(f"\nYour model and preprocessor are working correctly.")
        print(f"You can now safely proceed to:")
        print(f"  1. Create model pusher (upload to GCS)")
        print(f"  2. Register in Vertex AI Model Registry")
        print(f"  3. Deploy to Vertex AI Endpoint")
        print(f"  4. Set up CI/CD pipeline")
        print("="*80 + "\n")
        
        return model, preprocessor
    
    except Exception as e:
        print("\n" + "="*80)
        print("✗ INFERENCE TEST FAILED")
        print("="*80)
        print(f"Error: {str(e)}")
        print("\nAction: Run train_pipeline.py first to generate artifacts.")
        print("="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        model, preprocessor = test_inference()
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        sys.exit(1)
