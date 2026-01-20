"""
BigQuery Data Loader
Helper functions to load data from BigQuery for notebooks and pipeline components
"""
import os
import pandas as pd
from google.cloud import bigquery

# Configuration
PROJECT_ID = "mlops-churn-prediction-484819"
DATASET_ID = "telco_churn_dataset"
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "service-account-key.json")


def get_bigquery_client():
    """Initialize and return BigQuery client"""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
    return bigquery.Client(project=PROJECT_ID)


def load_raw_data():
    """Load raw customer churn data from BigQuery"""
    client = get_bigquery_client()
    query = f"""
    SELECT * 
    FROM `{PROJECT_ID}.{DATASET_ID}.customer_churn_raw`
    """
    print("Loading raw data from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_cleaned_data():
    """Load cleaned customer churn data from BigQuery"""
    client = get_bigquery_client()
    query = f"""
    SELECT * 
    FROM `{PROJECT_ID}.{DATASET_ID}.customer_churn_cleaned`
    """
    print("Loading cleaned data from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_custom_query(query):
    """Execute custom SQL query on BigQuery"""
    client = get_bigquery_client()
    print(f"Executing query on BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"✓ Query returned {len(df)} rows")
    return df


# Example usage
if __name__ == "__main__":
    # Test the loader
    print("Testing BigQuery loader...")
    df = load_cleaned_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
