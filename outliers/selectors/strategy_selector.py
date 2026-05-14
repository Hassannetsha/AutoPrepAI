import numpy as np
from scipy.stats import skew

from outliers.strategies.zscore_strategy import ZScoreStrategy
from outliers.strategies.iqr_strategy import IQRStrategy
from outliers.strategies.isolation_forest_strategy import IsolationForestStrategy
class StrategySelector:
    @staticmethod
    def choose_strategy(df):
        numeric_df = df.select_dtypes(include=[np.number])
        n_samples = numeric_df.shape[0]
        n_features = numeric_df.shape[1]

        # Calculate absolute skewness and take the average or max as the dataset representative
        skewness_values = numeric_df.apply(lambda col: skew(col.dropna()))
        max_skewness = np.max(np.abs(skewness_values))

        # 1. High-Dimensional
        if n_features > 4: 
            return IsolationForestStrategy()

        # 2. Z-Score Policy: Approx Normal (skew < 0.5) 
        if max_skewness < 0.5:
            return ZScoreStrategy()

        # 3. IQR Policy: Moderately Skewed (0.5 - 2.0) 
        if 0.5 <= max_skewness <= 2.0:
            return IQRStrategy()

        # 4. Isolation Forest Policy: Highly Skewed (> 2.0) or Multimodal 
        return IsolationForestStrategy()