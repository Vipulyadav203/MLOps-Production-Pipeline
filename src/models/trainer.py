import logging
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from typing import Dict, Any
import pandas as pd
from sklearn.metrics import f1_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles model training, hyperparameter tuning, and MLflow tracking.
    Following SOLID: Open/Closed Principle.
    """
    def __init__(self, experiment_name: str = "Fraud_Detection"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.info(f"Initialized ModelTrainer with experiment: {experiment_name}")

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials: int = 50) -> Dict[str, Any]:
        """
        Runs hyperparameter optimization using Optuna.
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials.")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
        
        logger.info(f"Hyperparameter tuning completed. Best trial f1-score: {study.best_value}")
        return study.best_params

    def train_and_log(self, X_train, y_train, X_test, y_test, params: Dict[str, Any]):
        """
        Trains the final model with best params and logs to MLflow.
        """
        logger.info(f"Training final model with params: {params}")
        with mlflow.start_run():
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Log params
            mlflow.log_params(params)
            
            # Evaluation
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            logger.info(f"Final model trained and logged to MLflow with f1-score: {f1}")
            return model
