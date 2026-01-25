"""
Model Pusher Component for MLOps Pipeline
Uploads trained model and metadata to GCS + Vertex AI Model Registry
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.utils.main_utils.utils import load_object

# Import GCP packages at module level
try:
    from google.cloud import storage
    from google.cloud import aiplatform
    GCP_PACKAGES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"GCP packages import failed: {e}")
    # Try to install if missing
    try:
        import subprocess
        logging.info("Attempting to install google-cloud-storage and google-cloud-aiplatform...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", 
                       "google-cloud-storage", "google-cloud-aiplatform"], 
                      check=True, capture_output=True)
        logging.info("Installation successful, attempting import again...")
        from google.cloud import storage
        from google.cloud import aiplatform
        GCP_PACKAGES_AVAILABLE = True
        logging.info("GCP packages imported successfully after installation")
    except Exception as install_error:
        GCP_PACKAGES_AVAILABLE = False
        logging.warning(f"Could not install GCP packages: {install_error}")


@dataclass
class ModelPusherConfig:
    """Configuration for Model Pusher"""
    
    gcs_bucket_name: str = "mlops-churn-models"  # Change to your bucket
    gcs_model_path: str = "telco-churn-model"
    vertex_ai_project: str = "mlops-churn-prediction-484819"  # Change to your project
    vertex_ai_region: str = "us-central1"
    
    def __post_init__(self):
        """Validate GCP credentials"""
        self.credentials_path = os.path.join(
            os.path.dirname(__file__), 
            "../..", 
            "config", 
            "service-account-key.json"
        )
        
        if not os.path.exists(self.credentials_path):
            logging.warning(
                f"GCP credentials not found at {self.credentials_path}. "
                f"Model push will fail. Set up GCP service account first."
            )


@dataclass
class ModelPusherArtifact:
    """Output from model pusher"""
    gcs_model_uri: str
    vertex_model_id: str
    pushed_successfully: bool


class ModelPusher:
    """
    Model Pusher Component
    
    Responsibilities:
    1. Load trained model from artifacts
    2. Create model directory with metadata (version, timestamp, metrics)
    3. Upload model + metadata to GCS
    4. Register model in Vertex AI Model Registry
    5. Return artifact with GCS URI and model ID
    
    Prerequisites:
    - GCP project configured
    - Service account JSON in config/service-account-key.json
    - GCS bucket created
    - Vertex AI API enabled
    """
    
    def __init__(self, model_trainer_artifact, model_pusher_config: ModelPusherConfig = None):
        """
        Initialize Model Pusher
        
        Args:
            model_trainer_artifact: Output from ModelTrainer with model path
            model_pusher_config: Configuration for pusher
        """
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_pusher_config = model_pusher_config or ModelPusherConfig()
            
            logging.info("Model Pusher Component initialized")
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def _load_model(self) -> tuple:
        """
        Load the trained model from artifact
        
        Returns:
            tuple: (model, model_path)
        """
        try:
            model_path = self.model_trainer_artifact.trained_model_file_path
            
            logging.info(f"Loading model from: {model_path}")
            model = load_object(model_path)
            
            logging.info(f"Model loaded successfully. Type: {type(model).__name__}")
            return model, model_path
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def _create_model_metadata(self, model, model_path: str) -> dict:
        """
        Create metadata for the model
        
        Args:
            model: Trained model object
            model_path: Path to model file
            
        Returns:
            dict: Model metadata
        """
        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "model_path": model_path,
                "source_artifact": str(self.model_trainer_artifact),
            }
            
            # Add metrics if available
            if hasattr(self.model_trainer_artifact, 'test_metric_artifact'):
                metrics = self.model_trainer_artifact.test_metric_artifact
                metadata["metrics"] = {
                    "f1_score": metrics.f1_score,
                    "precision": metrics.precision_score,
                    "recall": metrics.recall_score,
                    "roc_auc": metrics.roc_auc_score,
                }
            
            logging.info(f"Model metadata created: {metadata}")
            return metadata
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def _push_to_gcs(self, model_path: str, metadata: dict) -> str:
        """
        Upload model to GCS bucket
        
        Args:
            model_path: Local path to model file
            metadata: Model metadata dict
            
        Returns:
            str: GCS URI of uploaded model
        """
        try:
            if not GCP_PACKAGES_AVAILABLE:
                raise ImportError("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            
            logging.info("Initializing GCS client...")
            
            # Set credentials
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.model_pusher_config.credentials_path
            
            client = storage.Client(
                project=self.model_pusher_config.vertex_ai_project
            )
            bucket = client.bucket(self.model_pusher_config.gcs_bucket_name)
            
            # Create version directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_version = f"v{timestamp}"
            
            gcs_blob_path = f"{self.model_pusher_config.gcs_model_path}/{model_version}/model.pkl"
            metadata_blob_path = f"{self.model_pusher_config.gcs_model_path}/{model_version}/metadata.json"
            
            # Upload model
            logging.info(f"Uploading model to gs://{self.model_pusher_config.gcs_bucket_name}/{gcs_blob_path}")
            model_blob = bucket.blob(gcs_blob_path)
            model_blob.upload_from_filename(model_path)
            logging.info("✓ Model uploaded successfully")
            
            # Upload metadata
            logging.info(f"Uploading metadata to gs://{self.model_pusher_config.gcs_bucket_name}/{metadata_blob_path}")
            metadata_blob = bucket.blob(metadata_blob_path)
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type="application/json"
            )
            logging.info("✓ Metadata uploaded successfully")
            
            gcs_uri = f"gs://{self.model_pusher_config.gcs_bucket_name}/{gcs_blob_path}"
            logging.info(f"Model GCS URI: {gcs_uri}")
            
            return gcs_uri, model_version
        
        except ImportError:
            logging.error("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            raise
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def _register_in_vertex_ai(self, gcs_uri: str, model_version: str, metadata: dict) -> str:
        """
        Register model in Vertex AI Model Registry
        
        Args:
            gcs_uri: GCS URI of the model
            model_version: Version identifier
            metadata: Model metadata
            
        Returns:
            str: Vertex AI model ID
        """
        try:
            if not GCP_PACKAGES_AVAILABLE:
                raise ImportError("google-cloud-aiplatform not installed. Install with: pip install google-cloud-aiplatform")
            
            logging.info("Initializing Vertex AI client...")
            
            # Set credentials and project
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.model_pusher_config.credentials_path
            
            aiplatform.init(
                project=self.model_pusher_config.vertex_ai_project,
                location=self.model_pusher_config.vertex_ai_region,
            )
            
            # Upload model to Vertex AI
            logging.info(f"Registering model in Vertex AI Model Registry...")
            
            model = aiplatform.Model.upload(
                display_name=f"telco-churn-model-{model_version}",
                artifact_uri=gcs_uri,
                artifact_labels={
                    "version": model_version,
                    "timestamp": metadata["timestamp"],
                    "model_type": metadata["model_type"],
                }
            )
            
            model_id = model.resource_name.split('/')[-1]
            logging.info(f"✓ Model registered in Vertex AI. ID: {model_id}")
            
            return model_id
        
        except ImportError:
            logging.error("google-cloud-aiplatform not installed. Install with: pip install google-cloud-aiplatform")
            raise
        except Exception as e:
            logging.warning(f"Vertex AI registration failed (may need credentials): {str(e)}")
            return "REGISTRATION_PENDING"
    
    def initiate_model_push(self) -> ModelPusherArtifact:
        """
        Main method to push model to GCS and register in Vertex AI
        
        Returns:
            ModelPusherArtifact: Result of model push
        """
        try:
            logging.info("Entered initiate_model_push method")
            
            # Step 1: Load model
            logging.info("Step 1: Loading trained model...")
            model, model_path = self._load_model()
            
            # Step 2: Create metadata
            logging.info("Step 2: Creating model metadata...")
            metadata = self._create_model_metadata(model, model_path)
            
            # Step 3: Push to GCS
            logging.info("Step 3: Pushing to GCS...")
            try:
                gcs_uri, model_version = self._push_to_gcs(model_path, metadata)
            except Exception as e:
                logging.warning(f"GCS push failed: {str(e)}. Continuing without GCS upload.")
                gcs_uri = "PUSH_FAILED"
                model_version = "unknown"
            
            # Step 4: Register in Vertex AI
            logging.info("Step 4: Registering in Vertex AI...")
            try:
                vertex_model_id = self._register_in_vertex_ai(gcs_uri, model_version, metadata)
            except Exception as e:
                logging.warning(f"Vertex AI registration failed: {str(e)}. You may need to set up credentials.")
                vertex_model_id = "REGISTRATION_FAILED"
            
            # Step 5: Create artifact
            model_pusher_artifact = ModelPusherArtifact(
                gcs_model_uri=gcs_uri,
                vertex_model_id=vertex_model_id,
                pushed_successfully=(gcs_uri != "PUSH_FAILED" and vertex_model_id != "REGISTRATION_FAILED")
            )
            
            logging.info(f"Model Push completed")
            logging.info(f"   GCS URI: {model_pusher_artifact.gcs_model_uri}")
            logging.info(f"   Vertex Model ID: {model_pusher_artifact.vertex_model_id}")
            
            return model_pusher_artifact
        
        except Exception as e:
            logging.error(f"Model push failed: {str(e)}")
            raise NetworkSecurityException(e, sys)


# Example usage
if __name__ == "__main__":
    from Networksecurity.Entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact
    
    try:
        # For testing, create a dummy artifact
        dummy_artifact = ModelTrainerArtifact(
            trained_model_file_path="final_model/model.pkl",
            train_metric_artifact=ClassificationMetricArtifact(
                f1_score=0.85, precision_score=0.84, recall_score=0.86, roc_auc_score=0.92
            ),
            test_metric_artifact=ClassificationMetricArtifact(
                f1_score=0.82, precision_score=0.81, recall_score=0.83, roc_auc_score=0.90
            )
        )
        
        pusher = ModelPusher(dummy_artifact)
        artifact = pusher.initiate_model_push()
        
        print("\n" + "="*80)
        print("MODEL PUSH RESULT")
        print("="*80)
        print(f"GCS URI: {artifact.gcs_model_uri}")
        print(f"Vertex Model ID: {artifact.vertex_model_id}")
        print(f"Success: {artifact.pushed_successfully}")
        print("="*80)
    
    except Exception as e:
        print(f"Model push failed: {str(e)}")
