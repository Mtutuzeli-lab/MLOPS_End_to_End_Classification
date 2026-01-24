"""
Quick Test: Model Pusher Only
Tests the pusher without running full training
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from Networksecurity.Components.model_pusher import ModelPusher, ModelPusherConfig
from Networksecurity.Entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact
from Networksecurity.logging.logger import logging


def test_model_pusher():
    """
    Test model pusher with the latest artifacts
    """
    try:
        print("\n" + "="*80)
        print("MODEL PUSHER TEST")
        print("="*80 + "\n")
        
        # Find latest artifacts
        artifacts_dir = Path("artifacts")
        timestamp_folders = sorted(
            [d for d in artifacts_dir.iterdir() if d.is_dir()],
            reverse=True
        )
        
        if not timestamp_folders:
            print("✗ No artifacts found. Run train_pipeline.py first.")
            return
        
        latest = timestamp_folders[0]
        model_path = latest / "model_trainer" / "trained_model" / "model.pkl"
        
        if not model_path.exists():
            print(f"✗ Model not found at {model_path}")
            return
        
        print(f"✓ Using model from: {latest.name}")
        
        # Create artifact object
        dummy_artifact = ModelTrainerArtifact(
            trained_model_file_path=str(model_path),
            train_metric_artifact=ClassificationMetricArtifact(
                f1_score=0.85, precision_score=0.84, recall_score=0.86, roc_auc_score=0.92
            ),
            test_metric_artifact=ClassificationMetricArtifact(
                f1_score=0.82, precision_score=0.81, recall_score=0.83, roc_auc_score=0.90
            )
        )
        
        # Initialize pusher
        print("\nInitializing model pusher...")
        config = ModelPusherConfig()
        print(f"  GCS Bucket: {config.gcs_bucket_name}")
        print(f"  Vertex Project: {config.vertex_ai_project}")
        print(f"  Vertex Region: {config.vertex_ai_region}")
        
        pusher = ModelPusher(dummy_artifact, config)
        
        # Run pusher
        print("\nPushing model to GCS + Vertex AI...")
        artifact = pusher.initiate_model_push()
        
        # Results
        print("\n" + "="*80)
        print("PUSHER RESULTS")
        print("="*80)
        print(f"✓ GCS URI: {artifact.gcs_model_uri}")
        print(f"✓ Vertex Model ID: {artifact.vertex_model_id}")
        print(f"✓ Success: {artifact.pushed_successfully}")
        print("="*80 + "\n")
        
        if artifact.gcs_model_uri != "PUSH_FAILED":
            print("✓ Model successfully pushed to GCS!")
            print(f"  You can download it: {artifact.gcs_model_uri}")
        
        if artifact.vertex_model_id != "REGISTRATION_FAILED":
            print("✓ Model registered in Vertex AI!")
            print(f"  Model ID: {artifact.vertex_model_id}")
    
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_pusher()
