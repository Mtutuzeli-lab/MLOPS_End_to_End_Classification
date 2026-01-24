# CI/CD Setup Guide for MLOps Pipeline

## Overview

This document explains how to set up **Continuous Integration & Continuous Deployment (CI/CD)** for your MLOps project using GitHub Actions or Google Cloud Build.

---

## 📋 Table of Contents

1. [What is CI/CD?](#what-is-cicd)
2. [Option 1: GitHub Actions (Easiest)](#option-1-github-actions-easiest)
3. [Option 2: Google Cloud Build (More Control)](#option-2-google-cloud-build-more-control)
4. [Pipeline Workflow](#pipeline-workflow)
5. [Monitoring & Logs](#monitoring--logs)

---

## What is CI/CD?

### CI = Continuous Integration
**Automatically test code changes before merging**

```
You Push Code → GitHub Detects Change → Runs Tests → Notifies You
```

Benefits:
- Catch bugs immediately
- Ensure code quality
- Prevent broken code from reaching production

### CD = Continuous Deployment
**Automatically deploy validated code to production**

```
Tests Pass → Train Model → Push to Cloud → Deploy Endpoint → LIVE
```

Benefits:
- No manual deployment steps
- Always serving the latest model
- Reduces human error
- Fast iteration cycles

---

## Option 1: GitHub Actions (Easiest)

### Setup Steps

#### Step 1: Create GitHub Repository

```bash
git init
git add .
git commit -m "Initial commit: MLOps pipeline"
git remote add origin https://github.com/YOUR_USERNAME/MLOPS_End_to_End_Classification.git
git push -u origin main
```

#### Step 2: Add GitHub Secrets (Store Credentials Safely)

Go to: **GitHub Repo → Settings → Secrets and variables → Actions**

Click **New repository secret** and add:

**Secret Name:** `GCP_SA_KEY`
**Value:** (Copy entire contents of `config/service-account-key.json`)

This stores your GCP credentials securely (GitHub encrypts it).

#### Step 3: Workflow Already Created!

The file `.github/workflows/mlops-pipeline.yml` is already in your repo.

GitHub will automatically detect it and run on:
- **Every push to `main` or `develop` branches**
- **Daily at 2 AM UTC** (automated retraining)
- **Manual trigger** (click "Run workflow" on GitHub UI)

#### Step 4: Monitor Pipeline

1. Go to your GitHub repo
2. Click **Actions** tab
3. See all runs: **Successful ✓ or Failed ✗**
4. Click on a run to see detailed logs

### What Happens Automatically

```
┌─────────────────────────────────────────────┐
│ You Push Code to GitHub                     │
└────────────────┬────────────────────────────┘
                 ↓
         GitHub Detects Push
                 ↓
┌─────────────────────────────────────────────┐
│ GitHub Actions Workflow Triggers            │
└────────────────┬────────────────────────────┘
                 ↓
         1️⃣ Check out code
                 ↓
         2️⃣ Setup Python 3.11
                 ↓
         3️⃣ Install dependencies
                 ↓
         4️⃣ Setup GCP credentials (from secrets)
                 ↓
         5️⃣ Run test_inference.py
                 ↓
      ❌ FAIL? → Stop & Notify You
                 ↓
         6️⃣ Run train_pipeline.py
                 ↓
      ❌ FAIL? → Stop & Notify You
                 ↓
         7️⃣ Model pushed to GCS ✓
                 ↓
         8️⃣ Deploy to Vertex AI (main branch only)
                 ↓
┌─────────────────────────────────────────────┐
│ ✓ PIPELINE COMPLETE                         │
│ Your model is now LIVE!                     │
└─────────────────────────────────────────────┘
```

---

## Option 2: Google Cloud Build (More Control)

Google Cloud Build runs directly on GCP infrastructure (no need to push to GitHub first).

### Setup Steps

#### Step 1: Enable Cloud Build API

```bash
gcloud services enable cloudbuild.googleapis.com
```

#### Step 2: Connect GitHub Repository

```bash
gcloud builds connect --repository-name=MLOPS_End_to_End_Classification \
  --repository-owner=YOUR_GITHUB_USERNAME \
  --region=us-central1
```

#### Step 3: Create Build Trigger

```bash
gcloud builds triggers create github \
  --repo-name=MLOPS_End_to_End_Classification \
  --repo-owner=YOUR_GITHUB_USERNAME \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml \
  --name="mlops-train-deploy"
```

#### Step 4: Monitor Builds

```bash
# View recent builds
gcloud builds list --region=us-central1

# View logs of a specific build
gcloud builds log BUILD_ID --region=us-central1
```

---

## Pipeline Workflow

### What Each Step Does

| Step | Purpose | Triggers |
|------|---------|----------|
| **Test Inference** | Verify saved model + preprocessor work | Every push |
| **Train Pipeline** | Train all 5 models, select best | Every push |
| **Push to GCS** | Upload model artifact to cloud storage | After training succeeds |
| **Register in Vertex AI** | Add to model registry with version | After GCS push |
| **Deploy to Endpoint** | Make model live for predictions | Only on `main` branch |

### Failure Handling

If any step fails:

```
GitHub Actions:
  ❌ Test fails → Pipeline stops
     → Sends failure notification to your email
     → You fix code and push again

Cloud Build:
  ❌ Training fails → Logs saved to Cloud Logging
     → Deployment doesn't happen
     → You review logs, fix, re-trigger manually
```

---

## Monitoring & Logs

### GitHub Actions

1. Go to **Actions** tab in your repo
2. Click on a workflow run
3. See real-time logs as it executes
4. Download artifacts (trained models, logs)

### Cloud Build

```bash
# View all builds
gcloud builds list

# View detailed logs
gcloud builds log BUILD_ID

# Follow logs in real-time
gcloud builds log BUILD_ID --stream
```

---

## Deployment Script

The file `deploy_to_vertex_ai.py` does final deployment:

```bash
python deploy_to_vertex_ai.py
```

This:
1. Gets latest model from GCS
2. Creates Vertex AI Endpoint (if needed)
3. Deploys model to endpoint
4. Tests with sample predictions
5. Returns endpoint URI for your app

---

## Testing Locally Before CI/CD

Always test locally first:

```bash
# Test inference
python test_inference.py

# Test model pusher
python test_model_pusher.py

# Test deployment
python deploy_to_vertex_ai.py
```

Only push to GitHub after all tests pass!

---

## Environment Variables

If using secrets or environment variables in your code:

### GitHub Actions
Add to workflow YAML:
```yaml
env:
  GCP_PROJECT_ID: mlops-churn-prediction-484819
  GCS_BUCKET: mlops-churn-models
```

### Cloud Build
Set in `cloudbuild.yaml`:
```yaml
substitutions:
  _GCP_PROJECT: "mlops-churn-prediction-484819"
  _GCS_BUCKET: "mlops-churn-models"
```

---

## Troubleshooting

### "Pipeline Failed: Authentication Error"
- Check GCP credentials in GitHub Secrets
- Verify service account has required roles:
  - `roles/aiplatform.admin`
  - `roles/storage.admin`
  - `roles/bigquery.dataEditor`

### "Model Not Found in GCS"
- Verify model was saved successfully
- Check bucket name is correct
- Ensure GCP credentials have `storage.buckets.get` permission

### "Endpoint Deployment Failed"
- Check Vertex AI API is enabled
- Verify model container image is correct
- Check region is available

---

## Best Practices

✓ **DO:**
- Test locally before pushing
- Use meaningful commit messages
- Deploy to `develop` first, then `main`
- Monitor pipeline logs regularly
- Set up email notifications

✗ **DON'T:**
- Commit credentials or secrets
- Push directly to `main` without tests
- Ignore pipeline failures
- Use production endpoint for testing

---

## Next Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add CI/CD pipeline"
   git push origin main
   ```

2. **Watch the first run**
   - Go to Actions tab
   - See pipeline execute automatically

3. **Make a small change**
   - Update a comment
   - Push again
   - Watch the full pipeline run

4. **Monitor in DagsHub**
   - Your MLflow experiments sync automatically
   - See all runs with metrics

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Google Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Vertex AI Deployment](https://cloud.google.com/vertex-ai/docs/predictions)
- [MLflow DagsHub Integration](https://www.dagshub.com/docs/mlflow)
