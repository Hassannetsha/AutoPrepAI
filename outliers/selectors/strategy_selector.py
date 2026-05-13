import numpy as np

from scipy.stats import skew

from outliers.strategies.zscore_strategy import ZScoreStrategy
from outliers.strategies.iqr_strategy import IQRStrategy
from outliers.strategies.isolation_forest_strategy import IsolationForestStrategy


class StrategySelector:

    @staticmethod
    def choose_strategy(df):

        numeric_df = df.select_dtypes(include=[np.number])

        n_features = numeric_df.shape[1]

        skewness_values = numeric_df.apply(
            lambda col: skew(col.dropna())
        )

        max_skewness = np.max(np.abs(skewness_values))

        print(f"Max skewness: {max_skewness:.3f}")
        print(f"Feature count: {n_features}")

        # approximately normal
        if max_skewness < 0.5 and n_features < 10:

            return ZScoreStrategy()

        # moderately skewed
        elif max_skewness < 2.0:

            return IQRStrategy()

        # highly skewed / high-dimensional
        else:

            return IsolationForestStrategy()