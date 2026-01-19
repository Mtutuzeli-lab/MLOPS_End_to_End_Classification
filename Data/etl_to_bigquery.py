import pandas as pd
from google.cloud import bigquery
import os

# ========== CONFIGURATION ==========
PROJECT_ID = "mlops-churn-prediction-484819"  # Your GCP Project ID
DATASET_ID = "telco_churn_dataset"
CREDENTIALS_PATH = "../config/service-account-key.json"  # Go up one level to config

# ========== SETUP ==========
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
client = bigquery.Client(project=PROJECT_ID)

# ========== CREATE DATASET ==========
dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
try:
    client.get_dataset(dataset_ref)
    print(f"Dataset '{DATASET_ID}' already exists")
except:
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    client.create_dataset(dataset)
    print(f"Created dataset: {dataset_ref}")

# ========== UPLOAD RAW DATA ==========
print("\nUploading raw data...")
df_raw = pd.read_csv("Telco-Customer-Churn.csv")  # Same folder

# Fix data type issues for BigQuery
# Convert TotalCharges: remove empty strings and convert to numeric
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')

# Define schema explicitly to avoid inference issues
from google.cloud.bigquery import SchemaField
schema = [
    SchemaField("customerID", "STRING"),
    SchemaField("gender", "STRING"),
    SchemaField("SeniorCitizen", "INTEGER"),
    SchemaField("Partner", "STRING"),
    SchemaField("Dependents", "STRING"),
    SchemaField("tenure", "INTEGER"),
    SchemaField("PhoneService", "STRING"),
    SchemaField("MultipleLines", "STRING"),
    SchemaField("InternetService", "STRING"),
    SchemaField("OnlineSecurity", "STRING"),
    SchemaField("OnlineBackup", "STRING"),
    SchemaField("DeviceProtection", "STRING"),
    SchemaField("TechSupport", "STRING"),
    SchemaField("StreamingTV", "STRING"),
    SchemaField("StreamingMovies", "STRING"),
    SchemaField("Contract", "STRING"),
    SchemaField("PaperlessBilling", "STRING"),
    SchemaField("PaymentMethod", "STRING"),
    SchemaField("MonthlyCharges", "FLOAT"),
    SchemaField("TotalCharges", "FLOAT"),
    SchemaField("Churn", "STRING"),
]

table_ref = f"{PROJECT_ID}.{DATASET_ID}.customer_churn_raw"
job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(df_raw, table_ref, job_config=job_config)
job.result()
print(f"Uploaded {len(df_raw)} rows to: {table_ref}")

# ========== UPLOAD CLEANED DATA (if exists) ==========
if os.path.exists("Telco-Customer-Churn-cleaned.csv"):
    print("\nUploading cleaned data...")
    df_cleaned = pd.read_csv("Telco-Customer-Churn-cleaned.csv")  # Same folder
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.customer_churn_cleaned"
    job = client.load_table_from_dataframe(df_cleaned, table_ref)
    job.result()
    print(f"Uploaded {len(df_cleaned)} rows to: {table_ref}")

print("\nDone! Check BigQuery Console to verify your data.")