import logging
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles data loading, cleaning, and sampling.
    Following SOLID: Single Responsibility.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        logger.info(f"Initialized DataIngestion with random_state {random_state}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic data cleaning.
        """
        logger.info("Cleaning data: removing duplicates and handling missing values.")
        df = df.drop_duplicates()
        df = df.fillna(df.median(numeric_only=True))
        return df

    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Addresses class imbalance using SMOTE.
        """
        logger.info(f"Handling class imbalance using SMOTE. Initial distribution: {y.value_counts(normalize=True).to_dict()}")
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"Resampling complete. New distribution: {pd.Series(y_resampled).value_counts(normalize=True).to_dict()}")
        return X_resampled, y_resampled

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Splits data into train and test sets.
        """
        logger.info(f"Splitting data into train/test sets with test_size: {test_size}")
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)
