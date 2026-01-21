# MLOps Project - Documentation Resources Guide

**Project:** Telco Customer Churn Prediction MLOps Pipeline  
**Date:** January 21, 2026  
**Purpose:** Comprehensive documentation links for all components and libraries used

---

## Table of Contents
1. [Data Ingestion Component](#1-data-ingestion-component)
2. [Data Validation Component](#2-data-validation-component)
3. [Data Transformation Component](#3-data-transformation-component)
4. [Model Training Component](#4-model-training-component)
5. [Python Core Libraries](#5-python-core-libraries)
6. [MLOps Design Patterns](#6-mlops-design-patterns)
7. [Churn Prediction Specific](#7-churn-prediction-specific)
8. [Tutorials & Courses](#8-tutorials--courses)
9. [Code Quality & Best Practices](#9-code-quality--best-practices)
10. [Tools & Frameworks](#10-tools--frameworks)
11. [Learning Path](#learning-path-recommended-order)

---

## 1. Data Ingestion Component

### Pandas (Data Loading)
- **read_csv:** https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
- **DataFrame methods:** https://pandas.pydata.org/docs/reference/frame.html
- **to_csv:** https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
- **Main Documentation:** https://pandas.pydata.org/docs/

### Train-Test Split
- **train_test_split:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- **User Guide:** https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
- **Best Practices:** https://scikit-learn.org/stable/common_pitfalls.html

### Google BigQuery (Your Data Source)
- **BigQuery Python Client:** https://cloud.google.com/python/docs/reference/bigquery/latest
- **pandas-gbq:** https://pandas-gbq.readthedocs.io/en/latest/
- **Query to DataFrame:** https://cloud.google.com/bigquery/docs/pandas-gbq-migration
- **Authentication:** https://cloud.google.com/bigquery/docs/authentication

---

## 2. Data Validation Component

### Statistical Tests (Data Drift Detection)
- **scipy.stats.ks_2samp:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
  - Kolmogorov-Smirnov test for drift detection
- **Understanding KS Test:** https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
- **SciPy Stats Module:** https://docs.scipy.org/doc/scipy/reference/stats.html

### Data Quality Checks
- **DataFrame.dtypes:** https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html
- **DataFrame.columns:** https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html
- **DataFrame.isna():** https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
- **DataFrame.dropna():** https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html

### YAML Files (Reports)
- **PyYAML:** https://pyyaml.org/wiki/PyYAMLDocumentation
- **yaml.dump:** https://pyyaml.org/wiki/PyYAMLDocumentation#dumping-yaml
- **yaml.safe_load:** https://pyyaml.org/wiki/PyYAMLDocumentation#loading-yaml

---

## 3. Data Transformation Component

### KNN Imputation (Missing Values)
- **KNNImputer:** https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
- **Imputation Guide:** https://scikit-learn.org/stable/modules/impute.html
- **Comparison of Imputers:** https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html
- **Missing Data Strategies:** https://scikit-learn.org/stable/modules/impute.html#univariate-vs-multivariate-imputation

### Pipeline (Preprocessing)
- **Pipeline:** https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- **Pipeline User Guide:** https://scikit-learn.org/stable/modules/compose.html
- **Pipeline Visualization:** https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
- **ColumnTransformer:** https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html

### NumPy Operations
- **np.c_ (column stack):** https://numpy.org/doc/stable/reference/generated/numpy.c_.html
- **np.save:** https://numpy.org/doc/stable/reference/generated/numpy.save.html
- **np.load:** https://numpy.org/doc/stable/reference/generated/numpy.load.html
- **NumPy Basics:** https://numpy.org/doc/stable/user/basics.html

### Encoding & Scaling
- **OneHotEncoder:** https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- **StandardScaler:** https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- **Preprocessing Guide:** https://scikit-learn.org/stable/modules/preprocessing.html

---

## 4. Model Training Component

### SMOTE (Imbalanced Data Handling)
- **SMOTE:** https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- **Over-sampling Guide:** https://imbalanced-learn.org/stable/over_sampling.html
- **Imbalanced-Learn User Guide:** https://imbalanced-learn.org/stable/user_guide.html
- **SMOTE Research Paper:** https://arxiv.org/abs/1106.1813
- **Why SMOTE for Churn:** https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

### Classification Models

#### Logistic Regression
- **LogisticRegression API:** https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- **User Guide:** https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- **Tutorial:** https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html

#### Decision Tree
- **DecisionTreeClassifier API:** https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- **User Guide:** https://scikit-learn.org/stable/modules/tree.html
- **Visualization:** https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html

#### Random Forest
- **RandomForestClassifier API:** https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- **User Guide:** https://scikit-learn.org/stable/modules/ensemble.html#forest
- **Feature Importance:** https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

#### Gradient Boosting
- **GradientBoostingClassifier API:** https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
- **User Guide:** https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
- **Tutorial:** https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html

#### AdaBoost
- **AdaBoostClassifier API:** https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
- **User Guide:** https://scikit-learn.org/stable/modules/ensemble.html#adaboost
- **Example:** https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

### Evaluation Metrics

#### Accuracy
- **accuracy_score:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- **When to use:** https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

#### Precision, Recall, F1-Score
- **precision_score:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
- **recall_score:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
- **f1_score:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- **Classification Metrics Guide:** https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- **Confusion Matrix:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

#### ROC-AUC (Receiver Operating Characteristic)
- **roc_auc_score:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- **roc_curve:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
- **Understanding ROC:** https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
- **ROC Guide:** https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics

### Hyperparameter Tuning
- **GridSearchCV:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- **RandomizedSearchCV:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
- **Tuning Guide:** https://scikit-learn.org/stable/modules/grid_search.html

---

## 5. Python Core Libraries

### Dataclasses
- **@dataclass decorator:** https://docs.python.org/3/library/dataclasses.html
- **PEP 557 (Proposal):** https://peps.python.org/pep-0557/
- **Tutorial:** https://realpython.com/python-data-classes/

### Pickle (Object Serialization)
- **pickle module:** https://docs.python.org/3/library/pickle.html
- **pickle.dump:** https://docs.python.org/3/library/pickle.html#pickle.dump
- **pickle.load:** https://docs.python.org/3/library/pickle.html#pickle.load
- **Security Considerations:** https://docs.python.org/3/library/pickle.html#module-pickle

### OS & File Operations
- **os.path:** https://docs.python.org/3/library/os.path.html
- **os.makedirs:** https://docs.python.org/3/library/os.html#os.makedirs
- **pathlib (modern alternative):** https://docs.python.org/3/library/pathlib.html
- **File I/O:** https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files

### Type Hints
- **Type Hints:** https://docs.python.org/3/library/typing.html
- **Tuple:** https://docs.python.org/3/library/typing.html#typing.Tuple
- **Dict:** https://docs.python.org/3/library/typing.html#typing.Dict
- **PEP 484:** https://peps.python.org/pep-0484/

---

## 6. MLOps Design Patterns

### Google MLOps
- **MLOps Principles:** https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- **ML Pipeline Components:** https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build
- **Best Practices:** https://cloud.google.com/architecture/ml-on-gcp-best-practices

### Microsoft Azure MLOps
- **MLOps Overview:** https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment
- **ML Pipelines:** https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines
- **Best Practices:** https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-technical-paper

### AWS MLOps
- **SageMaker Pipelines:** https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html
- **MLOps Best Practices:** https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlops.html
- **ML Workflow:** https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-mlconcepts.html

### Papers & Research
- **Hidden Technical Debt in ML Systems (Google):** https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
- **MLOps: Continuous delivery and automation:** https://arxiv.org/abs/2205.02302
- **Machine Learning Operations (MLOps):** https://ml-ops.org/

---

## 7. Churn Prediction Specific

### Telco Churn Dataset
- **Kaggle Dataset:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Dataset Description:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

### Churn Prediction Resources
- **Feature Engineering for Churn:** https://towardsdatascience.com/predict-customer-churn-with-machine-learning-18fb2b7c8ab
- **Handling Imbalanced Churn Data:** https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
- **Churn Prediction Tutorial:** https://www.kaggle.com/code/bandiatindra/telecom-churn-prediction

### Business Context
- **Customer Churn Analysis:** https://www.optimove.com/resources/learning-center/customer-churn-rate
- **Churn Prevention Strategies:** https://www.zendesk.com/blog/customer-churn/

---

## 8. Tutorials & Courses

### Scikit-Learn
- **Official Tutorials:** https://scikit-learn.org/stable/tutorial/index.html
- **Getting Started:** https://scikit-learn.org/stable/getting_started.html
- **User Guide:** https://scikit-learn.org/stable/user_guide.html
- **Examples Gallery:** https://scikit-learn.org/stable/auto_examples/index.html

### MLOps
- **Made With ML (MLOps):** https://madewithml.com/#mlops
- **Full Stack Deep Learning:** https://fullstackdeeplearning.com/
- **MLOps Zoomcamp:** https://github.com/DataTalksClub/mlops-zoomcamp
- **Coursera MLOps:** https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops

### Production ML
- **Designing ML Systems Book:** https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/
- **ML Design Patterns Book:** https://www.oreilly.com/library/view/machine-learning-design/9781098115777/
- **Building ML Powered Applications:** https://www.oreilly.com/library/view/building-machine-learning/9781492045106/

### Python & Data Science
- **Real Python:** https://realpython.com/
- **Python Data Science Handbook:** https://jakevdp.github.io/PythonDataScienceHandbook/
- **Kaggle Learn:** https://www.kaggle.com/learn

---

## 9. Code Quality & Best Practices

### Python Style
- **PEP 8 (Style Guide):** https://peps.python.org/pep-0008/
- **Type Hints (PEP 484):** https://peps.python.org/pep-0484/
- **Docstring Conventions (PEP 257):** https://peps.python.org/pep-0257/
- **Python Enhancement Proposals:** https://peps.python.org/

### Clean Code
- **Clean Code Principles:** https://www.freecodecamp.org/news/clean-coding-for-beginners/
- **SOLID Principles:** https://en.wikipedia.org/wiki/SOLID
- **Code Review Best Practices:** https://google.github.io/eng-practices/review/

### Testing
- **pytest:** https://docs.pytest.org/
- **unittest:** https://docs.python.org/3/library/unittest.html
- **Testing Best Practices:** https://realpython.com/python-testing/

---

## 10. Tools & Frameworks

### Logging
- **Python logging:** https://docs.python.org/3/library/logging.html
- **Logging HOWTO:** https://docs.python.org/3/howto/logging.html
- **Logging Cookbook:** https://docs.python.org/3/howto/logging-cookbook.html

### Exception Handling
- **Python Exceptions:** https://docs.python.org/3/tutorial/errors.html
- **Custom Exceptions:** https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
- **Best Practices:** https://realpython.com/python-exceptions/

### Version Control
- **Git Documentation:** https://git-scm.com/doc
- **GitHub Guides:** https://guides.github.com/
- **Git Best Practices:** https://www.git-tower.com/learn/git/ebook/en/command-line/appendix/best-practices

### Environment Management
- **venv:** https://docs.python.org/3/library/venv.html
- **conda:** https://docs.conda.io/en/latest/
- **pip:** https://pip.pypa.io/en/stable/

---

## Learning Path (Recommended Order)

### Beginner (Weeks 1-4)
1. **Python Fundamentals**
   - https://docs.python.org/3/tutorial/
   - https://realpython.com/learning-paths/python3-introduction/

2. **Pandas & NumPy**
   - https://pandas.pydata.org/docs/getting_started/intro_tutorials/
   - https://numpy.org/doc/stable/user/absolute_beginners.html

3. **Scikit-Learn Basics**
   - https://scikit-learn.org/stable/tutorial/basic/tutorial.html

### Intermediate (Weeks 5-8)
4. **Classification Metrics & Evaluation**
   - https://scikit-learn.org/stable/modules/model_evaluation.html

5. **Pipeline & Preprocessing**
   - https://scikit-learn.org/stable/modules/compose.html

6. **Model Selection & Tuning**
   - https://scikit-learn.org/stable/modules/grid_search.html

### Advanced (Weeks 9-12)
7. **Imbalanced Learning & SMOTE**
   - https://imbalanced-learn.org/stable/user_guide.html

8. **MLOps Patterns & Architecture**
   - https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

9. **Production ML Systems**
   - https://madewithml.com/#mlops

### Expert (Weeks 13+)
10. **Advanced MLOps**
    - Kubeflow: https://www.kubeflow.org/docs/
    - MLflow: https://mlflow.org/docs/latest/index.html
    - DVC: https://dvc.org/doc

---

## Additional Resources

### Communities
- **MLOps Community:** https://mlops.community/
- **Kaggle:** https://www.kaggle.com/
- **Stack Overflow:** https://stackoverflow.com/questions/tagged/scikit-learn
- **Reddit r/MachineLearning:** https://www.reddit.com/r/MachineLearning/
- **Reddit r/MLOps:** https://www.reddit.com/r/mlops/

### Blogs & Newsletters
- **Towards Data Science:** https://towardsdatascience.com/
- **Machine Learning Mastery:** https://machinelearningmastery.com/
- **Google AI Blog:** https://ai.googleblog.com/
- **Netflix Tech Blog:** https://netflixtechblog.com/

### YouTube Channels
- **StatQuest:** https://www.youtube.com/@statquest
- **3Blue1Brown (Math):** https://www.youtube.com/@3blue1brown
- **Krish Naik:** https://www.youtube.com/@krishnaik06

---

## Project-Specific Files Reference

### Your Current Components
1. **Data Ingestion:** `Networksecurity/Components/data_ingestion.py`
2. **Data Validation:** `Networksecurity/Components/data_validation.py`
3. **Data Transformation:** `Networksecurity/Components/data_transformation.py`
4. **Model Trainer:** `Networksecurity/Components/model_trainer.py`

### Configuration Files
- **Constants:** `Networksecurity/Constants/training_pipeline.py`
- **Config Entities:** `Networksecurity/Entity/config_entity.py`
- **Artifact Entities:** `Networksecurity/Entity/artifact_entity.py`

### Utilities
- **Main Utils:** `Networksecurity/utils/main_utils/utils.py`
- **Exception Handler:** `Networksecurity/exception/exception.py`
- **Logger:** `Networksecurity/logging/logger.py`

---

## Quick Reference Cheat Sheets

### Common Scikit-Learn Operations
```python
# Load data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, f1_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Common Pandas Operations
```python
import pandas as pd

# Read CSV
df = pd.read_csv('data.csv')

# Basic info
df.shape
df.info()
df.describe()

# Handle missing values
df.isna().sum()
df.dropna()
df.fillna(value)

# Save
df.to_csv('output.csv', index=False)
```

---

**Last Updated:** January 21, 2026  
**Maintained By:** MLOps Team  
**Project Repository:** MLOps_End_to_End_Classification

---

## Notes

- All links verified as of January 2026
- Bookmark this file for quick reference
- Check official documentation for latest updates
- Some links may require internet connection
- Consider saving offline documentation for key libraries

---

**Happy Learning! 🚀**
