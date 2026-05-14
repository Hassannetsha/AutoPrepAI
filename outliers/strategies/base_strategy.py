from abc import ABC, abstractmethod
import pandas as pd


class OutlierStrategy(ABC):

    @abstractmethod
    def detect(
    self,
    df: pd.DataFrame,
    target_column: str = None
    ) -> pd.Series:
        """
        Returns:
            True  -> inlier
            False -> outlier
        """
        pass