import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Strategy Interface

class ScalingStrategy(ABC):

    @abstractmethod
    def scale(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        pass


# Strategies
class StandardScalingStrategy(ScalingStrategy):

    def scale(self, df, columns=None):

        df_out = df.copy().reset_index(drop=True)

        numeric_cols = df_out.select_dtypes(include=['number']).columns.tolist()

        cols = columns if columns else numeric_cols

        scaler = StandardScaler()

        df_out[cols] = scaler.fit_transform(df_out[cols])

        print("Standard Scaler applied")

        return df_out


class MinMaxScalingStrategy(ScalingStrategy):

    def __init__(self, feature_range=(0,1)):
        self.feature_range = feature_range


    def scale(self, df, columns=None):

        df_out = df.copy().reset_index(drop=True)

        numeric_cols = df_out.select_dtypes(include=['number']).columns.tolist()

        cols = columns if columns else numeric_cols

        scaler = MinMaxScaler(feature_range=self.feature_range)

        df_out[cols] = scaler.fit_transform(df_out[cols])

        print("MinMax Scaler applied")

        return df_out


class RobustScalingStrategy(ScalingStrategy):

    def scale(self, df, columns=None):

        df_out = df.copy().reset_index(drop=True)

        numeric_cols = df_out.select_dtypes(include=['number']).columns.tolist()

        cols = columns if columns else numeric_cols

        scaler = RobustScaler()

        df_out[cols] = scaler.fit_transform(df_out[cols])

        print("Robust Scaler applied")

        return df_out


# Context Class
class Scaler:

    def __init__(self):
        self.strategies = {
            "standard": StandardScalingStrategy(),
            "minmax": MinMaxScalingStrategy(),
            "robust": RobustScalingStrategy()
        }

    def scale(self, df: pd.DataFrame, method: str = "standard", columns=None):
        strategy = self.strategies.get(method.lower())

        if not strategy:
            raise ValueError(f"Unknown scaling method: {method}")

        return strategy.scale(df, columns)





    def set_strategy(self, strategy: ScalingStrategy):

        self.strategy = strategy


# main test

if __name__ == "__main__":

    df = pd.read_csv("data.csv")

    print("Original:")
    print(df.head())

    scaler = Scaler()

    df_standard = scaler.scale(df, method="standard")

    df_minmax = scaler.scale(df, method="minmax")

    df_robust = scaler.scale(df, method="robust")

    print("\nStandard:")
    print(df_standard.head())

    print("\nMinMax:")
    print(df_minmax.head())

    print("\nRobust:")
    print(df_robust.head())
