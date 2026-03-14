import logging
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles feature transformation and selection.
    Following SOLID: Interface Segregation.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.selected_features: List[str] = []
        logger.info("Initialized FeatureEngineer with StandardScaler.")

    def scale_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applies standard scaling to the specified columns.
        """
        logger.info(f"Scaling features: {columns}")
        df_scaled = df.copy()
        df_scaled[columns] = self.scaler.fit_transform(df[columns])
        return df_scaled

    def select_features_by_correlation(self, df: pd.DataFrame, target: str, threshold: float = 0.8) -> List[str]:
        """
        Identifies features based on their correlation with the target and each other.
        Reduces multicollinearity and noise.
        """
        logger.info(f"Selecting features with correlation threshold: {threshold}")
        corr_matrix = df.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        logger.info(f"Dropping highly correlated features: {to_drop}")
        features = [col for col in df.columns if col not in to_drop and col != target]
        self.selected_features = features
        return features
