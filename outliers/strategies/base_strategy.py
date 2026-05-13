from abc import ABC, abstractmethod
import pandas as pd


class OutlierStrategy(ABC):

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns:
            True  -> inlier
            False -> outlier
        """
        pass