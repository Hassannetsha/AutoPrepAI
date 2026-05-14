import pandas as pd

from outliers.selectors.strategy_selector import (StrategySelector)


class OutliersService:

    def __init__(self, dataframe: pd.DataFrame):

        self.dataframe = dataframe

        self.strategy = None

        self.cleaned_df = None

    def process(self):

        self.strategy = StrategySelector.choose_strategy(
            self.dataframe
        )

        mask = self.strategy.detect(self.dataframe)

        self.cleaned_df = self.dataframe[mask].copy()

        print(
            f"Selected Strategy: "
            f"{self.strategy.__class__.__name__}"
        )

        print(
            f"Removed rows: "
            f"{len(self.dataframe) - len(self.cleaned_df)}"
        )

        return self.cleaned_df

    def get_strategy_name(self):

        if self.strategy:
            return self.strategy.__class__.__name__

        return None