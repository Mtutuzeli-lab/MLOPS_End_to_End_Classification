# Batch Prediction Guide

## Overview

This guide explains how to use the **batch prediction** pipeline to make predictions for multiple customers at once.

---

## Pipeline Architecture

### Training Pipeline (One-time or scheduled)
```
train_pipeline.py → Models to GCS → Registered in Vertex AI
```

### Deployment (One-time)
```
deploy_to_vertex_ai.py → Model endpoint ready for predictions
```

### Batch Prediction Pipeline (Daily/Hourly)
```
batch_prediction.py → Load customers → Predict churn → Save results
```

---

## How Batch Prediction Works

```
STEP 1: Load Customer Data
├─ Reads from BigQuery table
├─ Format: customer features (gender, tenure, MonthlyCharges, etc.)
└─ Output: DataFrame with N customer records

STEP 2: Preprocess Data
├─ Applies same transformations as training
├─ Encodes categorical features
├─ Scales numerical features
└─ Output: Ready for model input

STEP 3: Make Predictions
├─ Calls deployed Vertex AI endpoint
├─ Processes in batches (configurable size)
├─ Gets churn probability for each customer
└─ Output: Array of predictions [0.1, 0.9, 0.3, ...]

STEP 4: Save Results
├─ Creates CSV file with predictions
├─ Saves to BigQuery for analysis
├─ Summary: "500/1000 customers predicted to churn (50%)"
└─ Ready for business action (send emails, offers, etc.)
```

---

## Usage

### Prerequisites

Before running batch prediction, ensure:

1. ✅ **Training Pipeline Completed**
   ```bash
   python train_pipeline.py
   ```
   Creates: `final_model/preprocessor.pkl` and `final_model/model.pkl`

2. ✅ **Model Deployed**
   ```bash
   python deploy_to_vertex_ai.py
   ```
   Creates: Vertex AI Endpoint (get endpoint ID from output)

3. ✅ **Customer Data in BigQuery**
   - Table: `telco_churn_dataset.customer_data`
   - Must have same columns as training data

### Running Batch Prediction

**For testing (1000 records):**
```bash
python batch_prediction.py
```

**For production (all records):**
Edit `batch_prediction.py` line with `load_batch_data(limit=1000)` → remove `limit=1000`

### Output

The script creates:
1. **CSV File**: `batch_predictions/predictions_YYYYMMDD_HHMMSS.csv`
2. **BigQuery Table**: `telco_churn_dataset.churn_predictions`
3. **Console Summary**: Churn rate and statistics

---

## Scheduling Batch Prediction (Cloud Scheduler)

To run batch prediction automatically daily at 2 AM:

```bash
# Create Cloud Scheduler job
gcloud scheduler jobs create app-engine batch-churn-prediction \
  --schedule="0 2 * * *" \
  --timezone="UTC" \
  --http-method=POST \
  --uri=https://YOUR_CLOUD_FUNCTION_URL/batch-predict \
  --oidc-service-account-email=YOUR_SA@PROJECT.iam.gserviceaccount.com
```

Or use Cloud Functions to wrap the script:

```python
def batch_predict(request):
    import subprocess
    result = subprocess.run(['python', 'batch_prediction.py'], capture_output=True)
    return result.stdout.decode()
```

---

## Interpreting Results

### CSV Output Columns

| Column | Description | Example |
|--------|-------------|---------|
| gender | Customer gender | M, F |
| tenure | Months as customer | 24 |
| MonthlyCharges | Monthly bill | $65.50 |
| churn_probability | 0-1 likelihood of churn | 0.87 |
| churn_prediction | 0=stay, 1=churn | 1 |
| prediction_timestamp | When prediction was made | 2026-01-25T20:35:00 |

### Example Analysis

**High Churn Risk (probability > 0.8):**
- Target with retention offers
- Proactive customer service outreach
- Special discounts

**Medium Churn Risk (0.5-0.8):**
- Monitor closely
- Improve satisfaction

**Low Churn Risk (< 0.5):**
- Maintain service quality
- Upsell opportunities

---

## Common Issues

### Issue: "Preprocessor not found"
**Solution:** Run `python train_pipeline.py` first to create preprocessor

### Issue: "Endpoint not found"
**Solution:** Run `python deploy_to_vertex_ai.py` to deploy the model

### Issue: "BigQuery timeout"
**Solution:** 
- Check network connection
- Reduce batch size
- Use smaller dataset limit

### Issue: "Authentication error"
**Solution:**
- Ensure `config/service-account-key.json` exists
- Check GCP permissions for your service account

---

## Advanced: Custom Query

To predict for specific customers, edit the query in `batch_prediction.py`:

```python
# Example: Only predict for customers with high monthly charges
query = f"""
SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
WHERE MonthlyCharges > 100
"""
```

---

## Next Steps

1. ✅ Complete training pipeline
2. ✅ Deploy model to Vertex AI
3. ✅ Run batch predictions daily
4. 🆕 **Set up Looker/Power BI dashboards** to visualize churn predictions
5. 🆕 **Create email alerts** for high-risk customers
6. 🆕 **Schedule Cloud Function** to automate daily predictions

---

## Questions?

For more info:
- Training pipeline: See `train_pipeline.py`
- Deployment: See `deploy_to_vertex_ai.py`
- Model details: See `CI_CD_SETUP.md`
