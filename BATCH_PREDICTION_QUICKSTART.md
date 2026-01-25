# Quick Start Guide: Batch Prediction

**Get predictions for your customers in 3 simple steps!**

---

## 🚀 Quick Start (2 Minutes)

### Step 1: Local Testing
```bash
python batch_prediction_local.py
```
✅ Makes predictions on test data (no cloud costs)
✅ Outputs: `batch_predictions/predictions_local_test_*.csv`

### Step 2: Deploy Model (One-time)
```bash
python deploy_to_vertex_ai.py
```
✅ Model deployed and ready
✅ Endpoint created in Vertex AI

### Step 3: Score Your Customers
```bash
python batch_prediction.py
```
✅ Scores all customers in BigQuery
✅ Outputs: CSV + BigQuery table

---

## 📊 Understanding the Output

**CSV File Columns:**
- `churn_probability` — Risk score (0-1)
- `churn_prediction` — 0=Stay, 1=Leave
- `prediction_timestamp` — When prediction was made
- (All your original customer columns)

**Example:**
```
gender  tenure  MonthlyCharges  churn_probability  churn_prediction
M       24      $65            0.12               0
F       3       $95            0.89               1
M       48      $45            0.05               0
```

---

## 💡 What To Do With Predictions

### High Risk (0.8-1.0) → ACT NOW
- Send special retention offer
- Call customer immediately
- Offer discount or upgrade

### Medium Risk (0.5-0.8) → MONITOR
- Improve service quality
- Reach out proactively
- Gather feedback

### Low Risk (0.0-0.5) → NURTURE
- Maintain service
- Upsell opportunities
- Build loyalty

---

## ⚙️ Configuration

Edit `batch_prediction.py` to change:

```python
# Load only 1000 records (for testing)
df = load_batch_data(limit=1000)

# Change to None for all records
df = load_batch_data(limit=None)

# Change batch size for performance
predictions = make_batch_predictions(X_processed, batch_size=100)
```

---

## ❓ Common Questions

**Q: How often should I run batch prediction?**
A: Daily or weekly, depending on business needs. Set up Cloud Scheduler to automate.

**Q: Can I predict for specific customers?**
A: Yes! Edit the BigQuery query in batch_prediction.py to filter by criteria.

**Q: How accurate are the predictions?**
A: Model F1-Score: 98.5% (very accurate!)

**Q: What if deployment fails?**
A: Run `python batch_prediction_local.py` instead - doesn't need deployment.

**Q: Can I run this on a schedule?**
A: Yes! Use Cloud Scheduler to run the script hourly/daily/weekly.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Preprocessor not found" | Run `python train_pipeline.py` first |
| "Endpoint not found" | Run `python deploy_to_vertex_ai.py` |
| "Authentication error" | Check `config/service-account-key.json` exists |
| "BigQuery timeout" | Reduce batch size or data limit |

---

## 📈 Next: Set Up Automation

To run predictions automatically daily at 8 AM:

```bash
gcloud scheduler jobs create app-engine batch-churn \
  --schedule="0 8 * * *" \
  --timezone="UTC" \
  --http-method=POST \
  --uri=https://YOUR_CLOUD_FUNCTION_URL
```

---

## 📝 Summary

| Task | Command | Time | Output |
|------|---------|------|--------|
| Test locally | `python batch_prediction_local.py` | 1 min | CSV file |
| Deploy model | `python deploy_to_vertex_ai.py` | 2 min | Active endpoint |
| Score customers | `python batch_prediction.py` | 5+ min | CSV + BigQuery |

**Total time to first predictions: ~10 minutes!**

---

## 🎯 Example Use Case

**Monday 8 AM:**
```bash
python batch_prediction.py
# Results: 500 of 10,000 customers at high churn risk

Sales Team Actions:
- Email: "We value you! Here's 20% off this month"
- Call: Reach out to top 50 highest-risk customers
- Offer: Free service upgrade for 3 months

Friday:
- Measure: Did they stay? Offer working? Adjust strategy
```

---

For more details, see:
- 📖 [BATCH_PREDICTION_GUIDE.md](BATCH_PREDICTION_GUIDE.md) — Full guide
- 🏗️ [MLOPS_WORKFLOW.md](MLOPS_WORKFLOW.md) — Architecture
- 🔗 [CI_CD_SETUP.md](CI_CD_SETUP.md) — Automation
