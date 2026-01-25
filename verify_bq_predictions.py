import os
from google.cloud import bigquery

PROJECT_ID = "mlops-churn-prediction-484819"
DATASET_ID = "telco_churn_dataset"
TABLE_ID = "churn_predictions"

cred_path = os.path.join("config", "service-account-key.json")
if os.path.exists(cred_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

client = bigquery.Client(project=PROJECT_ID)

print("Checking BigQuery predictions table...")
count_query = f"SELECT COUNT(*) AS total FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
count_df = client.query(count_query).to_dataframe()
print(f"Total rows: {int(count_df['total'].iloc[0])}")

sample_query = f"""
SELECT gender, SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
       churn_probability, churn_prediction, prediction_timestamp
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
ORDER BY prediction_timestamp DESC
LIMIT 5
"""
sample_df = client.query(sample_query).to_dataframe()
print("Sample rows:")
print(sample_df.to_string(index=False))
