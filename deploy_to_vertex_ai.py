"""
Vertex AI Model Deployment Script
Deploys the model to Vertex AI Endpoint for serving predictions
"""

import os
import sys
from google.cloud import aiplatform

# Configuration
PROJECT_ID = "mlops-churn-prediction-484819"
REGION = "us-central1"
ENDPOINT_DISPLAY_NAME = "telco-churn-endpoint"
MODEL_DISPLAY_NAME = "telco-churn-model"
BUCKET_URI = "gs://mlops-churn-models/telco-churn-model"

def get_latest_model():
    """Get the latest model from local directory"""
    local_model_path = "final_model/model.pkl"
    
    if not os.path.exists(local_model_path):
        raise Exception(f"Model not found at {local_model_path}")
    
    print(f"✓ Local model found: {local_model_path}")
    return local_model_path


def deploy_to_vertex_ai():
    """Deploy model to Vertex AI Endpoint"""
    
    try:
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        print("\n" + "="*80)
        print("DEPLOYING MODEL TO VERTEX AI")
        print("="*80 + "\n")
        
        # Get latest model URI
        model_uri = get_latest_model()
        
        # Step 1: Check if endpoint exists
        print("Step 1: Checking for existing endpoint...")
        existing_endpoints = aiplatform.Endpoint.list(
            filter=f"display_name={ENDPOINT_DISPLAY_NAME}"
        )
        
        if existing_endpoints:
            endpoint = existing_endpoints[0]
            print(f"✓ Using existing endpoint: {endpoint.display_name}")
        else:
            # Create new endpoint
            print("Creating new endpoint...")
            endpoint = aiplatform.Endpoint.create(
                display_name=ENDPOINT_DISPLAY_NAME,
                project=PROJECT_ID,
                location=REGION
            )
            print(f"✓ Endpoint created: {endpoint.display_name}")
        
        # Step 2: Upload model to Vertex AI Model Registry
        print("\nStep 2: Uploading model to Vertex AI...")
        model = aiplatform.Model.upload(
            display_name=MODEL_DISPLAY_NAME,
            artifact_uri=model_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
            # For sklearn models, use this container image
        )
        print(f"✓ Model uploaded: {model.display_name}")
        print(f"  Model ID: {model.resource_name}")
        
        # Step 3: Deploy model to endpoint
        print("\nStep 3: Deploying model to endpoint...")
        endpoint.deploy(
            model=model,
            display_name="telco-churn-predictor",
            machine_type="n1-standard-4",  # 4 CPU cores
            accelerator_count=0,  # No GPU needed for inference
            traffic_split={"0": 100},  # Route 100% traffic to this model
            min_replica_count=1,  # Minimum 1 instance
            max_replica_count=3  # Auto-scale up to 3 instances
        )
        
        print(f"✓ Model deployed to endpoint!")
        print(f"  Endpoint ID: {endpoint.resource_name}")
        
        # Step 4: Test predictions
        print("\nStep 4: Testing predictions...")
        # Sample features (must match your preprocessor output)
        test_data = [[1, 0, 1, 0, 12, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 65.3, 0]]
        
        predictions = endpoint.predict(instances=[test_data])
        print(f"✓ Predictions returned successfully!")
        print(f"  Sample prediction: {predictions.predictions[0]}")
        
        # Final summary
        print("\n" + "="*80)
        print("✓ DEPLOYMENT SUCCESSFUL!")
        print("="*80)
        print(f"\nEndpoint Details:")
        print(f"  Name: {endpoint.display_name}")
        print(f"  Region: {REGION}")
        print(f"  Project: {PROJECT_ID}")
        print(f"\nYour model is now live and ready to serve predictions!")
        print(f"Endpoint URI: {endpoint.resource_name}")
        print("="*80 + "\n")
        
        return endpoint
    
    except Exception as e:
        print(f"\n✗ Deployment failed: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Verify GCP credentials are set up")
        print("  2. Check that Vertex AI API is enabled")
        print("  3. Ensure service account has required permissions")
        sys.exit(1)


if __name__ == "__main__":
    # Set credentials
    credentials_path = "config/service-account-key.json"
    if not os.path.exists(credentials_path):
        print(f"✗ Credentials not found at {credentials_path}")
        print("  Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        sys.exit(1)
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    
    # Deploy
    deploy_to_vertex_ai()
