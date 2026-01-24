"""
Artifact Entity Classes
Defines dataclasses for output artifacts from each pipeline component
"""

from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Output artifact from Data Ingestion Component
    Contains paths to ingested train and test files
    """
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Output artifact from Data Validation Component
    Contains validation status and paths to validated files
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """
    Output artifact from Data Transformation Component
    Contains paths to transformed numpy arrays and preprocessor object
    """
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    """
    Classification metrics for model evaluation
    """
    f1_score: float
    precision_score: float
    recall_score: float
    roc_auc_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Output artifact from Model Training Component
    Contains path to trained model and performance metrics
    """
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    """
    Output artifact from Model Pusher Component
    Contains GCS URI and Vertex AI model registry ID
    """
    gcs_model_uri: str
    vertex_model_id: str
    pushed_successfully: bool
