import logging
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Handles model evaluation and bias detection.
    Following SOLID: Liskov Substitution Principle.
    """
    def __init__(self):
        logger.info("Initialized ModelEvaluator.")

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Performs comprehensive evaluation of the model.
        """
        logger.info("Evaluating model performance.")
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        
        precision, recall, _ = precision_recall_curve(y_test, probs)
        auprc = auc(recall, precision)
        
        logger.info(f"Model evaluation complete. AUPRC: {auprc}")
        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "auprc": auprc
        }

    def detect_bias(self, df: pd.DataFrame, target: str, predictions: pd.Series, sensitive_col: str):
        """
        Simple bias detection for sensitive columns.
        """
        logger.info(f"Detecting bias in sensitive column: {sensitive_col}")
        bias_df = df.copy()
        bias_df['preds'] = predictions
        bias_df['target'] = target
        
        # Calculate FPR/FNR per group in sensitive column
        for group in bias_df[sensitive_col].unique():
            group_df = bias_df[bias_df[sensitive_col] == group]
            cm = confusion_matrix(group_df['target'], group_df['preds'])
            logger.info(f"Group: {group}, Confusion Matrix: {cm}")
            # Add more sophisticated bias metrics here (e.g., disparate impact)
