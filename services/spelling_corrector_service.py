"""
Spelling Corrector using SymSpell

This module provides a SpellingCorrector class that uses SymSpell for 
detecting and correcting spelling mistakes in text data.
"""

from importlib.resources import files
from typing import List, Optional, Union

import pandas as pd
from symspellpy import SymSpell
from tqdm import tqdm

class SpellingCorrectorService:

    def __init__(self, max_edit_distance=2, prefix_length=7):
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length

        self.symspell = SymSpell(max_edit_distance, prefix_length)

        dictionary_path = files("symspellpy").joinpath(
            "frequency_dictionary_en_82_765.txt"
        )

        self.symspell.load_dictionary(
            str(dictionary_path),
            term_index=0,
            count_index=1,
        )

        self._is_dictionary_built = True

    
    def correct_word(
        self, 
        word: str, 
        max_edit_distance: Optional[int] = None
    ) -> str:
        """
        Correct a single word.
        
        Args:
            word (str): The word to correct.
            max_edit_distance (int, optional): Maximum edit distance for this correction.
                If None, uses the instance's max_edit_distance.
        
        Returns:
            str: The corrected word, or the original if no correction found.
        """
        if not self._is_dictionary_built:
            raise ValueError("Dictionary not built. Call one of the build_dictionary_* methods first.")
        
        edit_distance = max_edit_distance if max_edit_distance is not None else self.max_edit_distance
        
        suggestions = self.symspell.lookup_compound(
            word,
            max_edit_distance=edit_distance
        )
        
        if suggestions:
            return suggestions[0].term
        return word
    
    def correct_list(
        self, 
        words: List[str],
        max_edit_distance: Optional[int] = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        Correct a list of words.
        
        Args:
            words (List[str]): List of words to correct.
            max_edit_distance (int, optional): Maximum edit distance for corrections.
            show_progress (bool): Whether to show progress bar. Default is True.
        
        Returns:
            List[str]: List of corrected words.
        """
        if not self._is_dictionary_built:
            raise ValueError("Dictionary not built. Call one of the build_dictionary_* methods first.")
        
        edit_distance = max_edit_distance if max_edit_distance is not None else self.max_edit_distance
        corrected = []
        
        iterator = tqdm(words, desc="Correcting words") if show_progress else words
        
        for word in iterator:
            if not word or not str(word).strip():
                corrected.append(word)
                continue

            suggestions = self.symspell.lookup_compound(
                str(word),
                max_edit_distance=edit_distance
            )

            corrected.append(suggestions[0].term if suggestions else word)
        
        return corrected
    
    def correct_dataframe_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        max_edit_distance: Optional[int] = None,
        show_progress: bool = True,
        inplace: bool = False,
        new_column_name: Optional[str] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Correct spelling in a DataFrame column.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            column_name (str): Name of the column to correct.
            max_edit_distance (int, optional): Maximum edit distance for corrections.
            show_progress (bool): Whether to show progress bar. Default is True.
            inplace (bool): If True, modifies the DataFrame in place. Default is False.
            new_column_name (str, optional): Name for the corrected column. 
                If None and inplace is True, overwrites the original column.
                If None and inplace is False, returns a Series.
        
        Returns:
            Union[pd.DataFrame, pd.Series]: 
                - If inplace is True: Returns the modified DataFrame.
                - If inplace is False and new_column_name is provided: Returns DataFrame with new column.
                - If inplace is False and new_column_name is None: Returns corrected Series.
        """
        edit_distance = (
            max_edit_distance
            if max_edit_distance is not None
            else self.max_edit_distance
        )
        corrected_col = []

        words = df[column_name].fillna("").astype(str)

        iterator = (
            tqdm(words, desc=f"Correcting column '{column_name}'")
            if show_progress
            else words
        )

        for word in iterator:
            if not word.strip():
                corrected_col.append(word)
                continue

            suggestions = self.symspell.lookup_compound(
                word,
                max_edit_distance=edit_distance
            )

            corrected_col.append(
                suggestions[0].term if suggestions else word
            )

        corrected_series = pd.Series(corrected_col, index=df.index)

        if inplace:
            target_column = new_column_name or column_name
            df[target_column] = corrected_series
            return df

        elif new_column_name:
            result_df = df.copy()
            result_df[new_column_name] = corrected_series
            return result_df

        return corrected_series
    
    def evaluate_corrections(
        self,
        df: pd.DataFrame,
        misspelled_column: str,
        correct_column: str,
        corrected_column: str
    ) -> dict:
        """
        Evaluate the accuracy of spelling corrections.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            misspelled_column (str): Name of the column with misspelled words.
            correct_column (str): Name of the column with correct words.
            corrected_column (str): Name of the column with corrected words.
        
        Returns:
            dict: Dictionary containing evaluation metrics:
                - total: Total number of corrections
                - correct: Number of correct corrections
                - incorrect: Number of incorrect corrections
                - accuracy: Accuracy as a percentage
                - incorrect_df: DataFrame with incorrect corrections
        """
        incorrect_corrections = df[df[corrected_column] != df[correct_column]]
        
        total = len(df)
        correct = total - len(incorrect_corrections)
        incorrect = len(incorrect_corrections)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        return {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'incorrect_df': incorrect_corrections[[misspelled_column, corrected_column, correct_column]]
        }
    
    def is_dictionary_built(self) -> bool:
        """
        Check if a dictionary has been built.
        
        Returns:
            bool: True if dictionary is built, False otherwise.
        """
        return self._is_dictionary_built
    
    def get_dictionary_size(self) -> int:
        """
        Get the size of the current dictionary.
        
        Returns:
            int: Number of words in the dictionary.
        """
        return self.symspell.word_count

