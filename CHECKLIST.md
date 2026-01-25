# 📋 MLOps Setup Checklist

Use this checklist to track your progress through the MLOps pipeline.

---

## ✅ Phase 1: Setup (Already Done!)

- [x] Created GCS bucket: `gs://mlops-churn-models`
- [x] Set up Vertex AI in `us-central1`
- [x] Installed Google Cloud SDK
- [x] Created service account with credentials
- [x] Updated `requirements.txt` with GCP packages
- [x] Fixed model pusher for GCS/Vertex AI integration

---

## 📚 Phase 2: Documentation (Already Done!)

- [x] Created `BATCH_PREDICTION_GUIDE.md` — Full batch prediction guide
- [x] Created `BATCH_PREDICTION_QUICKSTART.md` — 2-minute quick start
- [x] Created `MLOPS_WORKFLOW.md` — Complete architecture
- [x] Created `SUMMARY_WHAT_WE_BUILT.md` — Today's summary
- [x] Created `batch_prediction.py` — Cloud batch prediction script
- [x] Created `batch_prediction_local.py` — Local testing script

---

## 🚀 Phase 3: Test Your Pipeline (Do This Now!)

### Step 1: Test Local Batch Prediction (5 minutes)
```bash
python batch_prediction_local.py
```
- [ ] Script runs without errors
- [ ] CSV file created in `batch_predictions/` folder
- [ ] Can see sample predictions in console
- [ ] Understand the churn_probability column

**Expected output:**
```
✅ Batch Prediction Summary
   Total records: 4220
   Predicted churn: 850
   Predicted retention: 3370
   Churn rate: 20.14%
   Output file: batch_predictions/predictions_local_test_YYYYMMDD_HHMMSS.csv
```

### Step 2: Deploy Model to Cloud (One-time, 2 minutes)
```bash
python deploy_to_vertex_ai.py
```
- [ ] Model endpoint created successfully
- [ ] Endpoint is in DEPLOYED state
- [ ] Can see endpoint URL in output
- [ ] Note down the endpoint ID

**Expected output:**
```
✅ Deployment Summary
   Endpoint Name: telco-churn-endpoint
   Endpoint ID: 1234567890
   Status: DEPLOYED
   Ready for predictions: YES
```

### Step 3: Cloud Batch Prediction (5+ minutes)
```bash
python batch_prediction.py
```
- [ ] Data loaded from BigQuery
- [ ] Predictions completed for all records
- [ ] CSV file created with predictions
- [ ] Results saved to BigQuery table
- [ ] Can access `telco_churn_dataset.churn_predictions` table

**Expected output:**
```
✅ Batch Prediction Summary
   Total records: 21096
   Predicted churn: 4200
   Predicted retention: 16896
   Churn rate: 19.89%
   Output file: batch_predictions/predictions_YYYYMMDD_HHMMSS.csv
   BigQuery table: telco_churn_dataset.churn_predictions
```

---

## 🔄 Phase 4: Integration (Next Week)

### Set Up Automation with Cloud Scheduler
- [ ] Enable Cloud Scheduler API
- [ ] Create scheduled job to run `batch_prediction.py` daily
- [ ] Set time (e.g., 8 AM UTC)
- [ ] Verify job runs successfully
- [ ] Set up notifications for failures

**Command:**
```bash
gcloud scheduler jobs create app-engine batch-churn-prediction \
  --schedule="0 8 * * *" \
  --timezone="UTC" \
  --http-method=POST \
  --uri=https://YOUR_CLOUD_FUNCTION_URL/batch-predict
```

### Set Up BigQuery Alerts
- [ ] Create view of high-risk customers
- [ ] Export CSV for CRM import
- [ ] Set up Data Transfer to export daily to Cloud Storage

**Example query:**
```sql
SELECT *
FROM `mlops-churn-prediction-484819.telco_churn_dataset.churn_predictions`
WHERE churn_probability > 0.8
ORDER BY churn_probability DESC
LIMIT 100
```

---

## 📊 Phase 5: Reporting (Next Week)

### Power BI / Looker Dashboard
- [ ] Connect to BigQuery `churn_predictions` table
- [ ] Create dashboard showing:
  - [ ] Daily churn rate trend
  - [ ] High-risk customer count
  - [ ] Risk distribution chart
  - [ ] Model performance metrics
- [ ] Set up data refresh schedule (daily)
- [ ] Share dashboard with stakeholders

### Business Actions
- [ ] Set up email alerts for high-risk customers
- [ ] Create retention offer templates
- [ ] Brief sales team on usage
- [ ] Track campaign effectiveness

---

## 🔐 Phase 6: Maintenance (Ongoing)

### Weekly Maintenance
- [ ] Check batch prediction logs in `logs/` folder
- [ ] Review BigQuery `churn_predictions` table growth
- [ ] Monitor GCS storage usage
- [ ] Check Cloud Scheduler job runs successfully

### Monthly Tasks
- [ ] Review model performance metrics
- [ ] Plan next retraining (add new data)
- [ ] Review budget/costs in Cloud Console
- [ ] Collect feedback from sales/support teams

### Quarterly Tasks
- [ ] Retrain model with accumulated new data
- [ ] Compare new model vs. old model
- [ ] Deploy new model version if better
- [ ] Update documentation with learnings

---

## 🎓 Understanding the Pipeline

### Data Flow
```
BigQuery (Customer Data)
    ↓
batch_prediction.py (Score customers)
    ↓
CSV + BigQuery Table (Results)
    ↓
Dashboard (Visualization)
    ↓
Sales/Support Team (Take action)
```

### Key Concepts
- **churn_probability**: 0-1 score (0.0 = stay, 1.0 = leave)
- **churn_prediction**: Binary (0 = stay, 1 = leave) based on 0.5 threshold
- **Batch size**: Number of records processed at once (100 is default)
- **Model version**: Timestamp when model was created

### Interpretation Examples
| Probability | Risk Level | Action |
|-------------|-----------|--------|
| 0.9 | CRITICAL | Call immediately, offer discount |
| 0.7 | HIGH | Send email, offer special deal |
| 0.4 | MEDIUM | Monitor, improve service |
| 0.1 | LOW | Regular service, upsell |

---

## 🐛 Troubleshooting Checklist

### Local Batch Prediction Fails
- [ ] Check if `final_model/model.pkl` exists
- [ ] Check if `final_model/preprocessor.pkl` exists
- [ ] Check if `artifacts/test.csv` exists
- [ ] Run: `python train_pipeline.py` to create missing files
- [ ] Check Python version: `python --version` (need 3.9+)

### Deployment Fails
- [ ] Check credentials: `cat config/service-account-key.json`
- [ ] Verify project ID is correct in `deploy_to_vertex_ai.py`
- [ ] Check Vertex AI API enabled: `gcloud services list --enabled`
- [ ] Check GCS bucket exists: `gsutil ls gs://mlops-churn-models`
- [ ] Run: `gcloud auth list` to verify credentials

### Cloud Batch Prediction Fails
- [ ] Check BigQuery table exists: `bq ls telco_churn_dataset`
- [ ] Verify customer_data table has data
- [ ] Check endpoint is deployed: `gcloud ai endpoints list`
- [ ] Check BigQuery permissions on service account
- [ ] Check firewall isn't blocking connections

### Data Issues
- [ ] Verify BigQuery connection: `python test_data_ingestion.py`
- [ ] Check network connection to cloud
- [ ] Verify service account has BigQuery permissions
- [ ] Check quota limits in Cloud Console

---

## 📞 Getting Help

### Resource Files
- **Quick Start**: `BATCH_PREDICTION_QUICKSTART.md`
- **Full Guide**: `BATCH_PREDICTION_GUIDE.md`
- **Architecture**: `MLOPS_WORKFLOW.md`
- **Summary**: `SUMMARY_WHAT_WE_BUILT.md`

### Log Files
- Training logs: `logs/01_25_2026_*.log`
- Check last log for errors: `tail -100 logs/01_25_2026_20_*.log`

### Test Scripts
```bash
python test_inference.py          # Test model prediction
python test_data_ingestion.py    # Test BigQuery connection
python test_model_pusher.py      # Test GCS/Vertex AI
```

---

## ✨ Success Indicators

You'll know everything is working when:

- ✅ `python batch_prediction_local.py` creates CSV in <2 minutes
- ✅ `python deploy_to_vertex_ai.py` completes without errors
- ✅ `python batch_prediction.py` saves results to BigQuery
- ✅ `batch_predictions/` folder has prediction CSVs
- ✅ Dashboard shows live churn predictions
- ✅ Sales team receives daily email alerts
- ✅ Retention offers are being sent based on predictions

---

## 🎯 Your Next Action

**Right now, run:**
```bash
python batch_prediction_local.py
```

This will:
1. Load test data locally
2. Make predictions (fast, no cloud calls)
3. Create CSV with results
4. Show you exactly what batch prediction does

**Then:**
1. Review the CSV file
2. Read `BATCH_PREDICTION_QUICKSTART.md`
3. Plan your cloud deployment

---

## 📊 Progress Tracker

```
[████████████████████] 100% - Setup & Documentation Complete
[████████████████████] 100% - Batch Prediction Scripts Ready
[████░░░░░░░░░░░░░░░░]  20% - Local Testing (You are here!)
[░░░░░░░░░░░░░░░░░░░░]   0% - Cloud Deployment
[░░░░░░░░░░░░░░░░░░░░]   0% - Automation & Monitoring
[░░░░░░░░░░░░░░░░░░░░]   0% - Dashboard & Business Integration
```

**Status:** 🟡 In Progress (Local testing phase)
**Next Milestone:** ☁️ Cloud Deployment
**Est. Completion:** This week!

---

**Last Updated:** January 25, 2026
**Created by:** MLOps Team
**Status:** ✅ Ready for Action
