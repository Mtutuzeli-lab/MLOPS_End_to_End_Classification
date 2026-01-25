"""
Simple Vertex AI Endpoint Setup
Creates a basic endpoint for batch prediction testing
"""

import os
import sys
from google.cloud import aiplatform

PROJECT_ID = "mlops-churn-prediction-484819"
REGION = "us-central1"
ENDPOINT_DISPLAY_NAME = "telco-churn-endpoint"

def setup_endpoint():
    """Create or get Vertex AI endpoint"""
    try:
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        print("\n" + "="*80)
        print("SETTING UP VERTEX AI ENDPOINT")
        print("="*80 + "\n")
        
        # Check if endpoint exists
        print("Checking for existing endpoint...")
        existing_endpoints = aiplatform.Endpoint.list(
            filter=f"display_name={ENDPOINT_DISPLAY_NAME}"
        )
        
        if existing_endpoints:
            endpoint = existing_endpoints[0]
            print(f"✓ Using existing endpoint: {endpoint.display_name}")
            print(f"  Resource name: {endpoint.resource_name}")
        else:
            print("Creating new endpoint...")
            endpoint = aiplatform.Endpoint.create(
                display_name=ENDPOINT_DISPLAY_NAME,
                project=PROJECT_ID,
                location=REGION
            )
            print(f"✓ Endpoint created: {endpoint.display_name}")
            print(f"  Resource name: {endpoint.resource_name}")
        
        print("\n" + "="*80)
        print("✓ ENDPOINT READY")
        print("="*80)
        print(f"\nEndpoint Details:")
        print(f"  Display Name: {endpoint.display_name}")
        print(f"  Region: {REGION}")
        print(f"  Project: {PROJECT_ID}")
        print(f"  Resource: {endpoint.resource_name}")
        print("\n✓ Ready for batch predictions!")
        print("="*80 + "\n")
        
        return endpoint
    
    except Exception as e:
        print(f"\n✗ Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    credentials_path = "config/service-account-key.json"
    if not os.path.exists(credentials_path):
        print(f"✗ Credentials not found at {credentials_path}")
        sys.exit(1)
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    setup_endpoint()
