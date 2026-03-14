# MLOps Production Pipeline (Fraud Detection)

An end-to-end MLOps pipeline for building, training, evaluating, and deploying a robust Fraud Detection model.

## Business Case: Fraud Detection

Financial institutions lose billions annually to fraudulent transactions. This project implements a production-ready pipeline to:
- **Minimize False Negatives**: Catch fraudulent transactions before they are processed.
- **Maintain Low False Positives**: Avoid frustrating legitimate customers.
- **Automate Retraining**: Ensure model performance doesn't drift over time.

## MLOps Pipeline Overview

1. **Data Ingestion**: Automated cleaning and handling of class imbalance using SMOTE.
2. **Feature Engineering**: Automated scaling, encoding, and correlation-based selection.
3. **Hyperparameter Tuning**: Bayesian optimization using `Optuna`.
4. **Experiment Tracking**: Full tracking of metrics, parameters, and models using `MLflow`.
5. **Model Evaluation**: In-depth analysis using Precision-Recall curves, Confusion Matrix, and Fairness/Bias detection.
6. **Real-time Inference**: High-performance API using `FastAPI`.
7. **Monitoring**: Model health and data drift monitoring using `Evidently`.

## Performance Metrics

- **Primary Metric**: F1-Score (balance between Precision and Recall).
- **Secondary Metric**: Area Under Precision-Recall Curve (AUPRC), crucial for imbalanced datasets.
- **Business Metric**: Reduction in Fraud Loss ($) vs. Customer Friction (False Positive Rate).

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

1. **Train Model**:
   ```bash
   python src/models/trainer.py
   ```

2. **Run Inference API**:
   ```bash
   uvicorn app.inference:app --reload
   ```

3. **Run Tests**:
   ```bash
   pytest tests/
   ```
