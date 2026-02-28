"""
Spelling Corrector using SymSpell

This module provides a SpellingCorrector class that uses SymSpell for 
detecting and correcting spelling mistakes in text data.
"""

import pandas as pd
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm
from typing import List, Union, Optional


class SpellingCorrectorService:
    """
    A class to correct spelling mistakes using SymSpell algorithm.
    
    This class can build a custom dictionary from various sources (DataFrame, list, file)
    and use it to correct spelling mistakes in text data.
    
    Attributes:
        max_edit_distance (int): Maximum edit distance for dictionary lookup.
        prefix_length (int): Length of word prefixes used for dictionary lookup.
        symspell (SymSpell): The underlying SymSpell object.
    """
    
    def __init__(self, max_edit_distance: int = 2, prefix_length: int = 7):
        """
        Initialize the SpellingCorrector.
        
        Args:
            max_edit_distance (int): Maximum edit distance for corrections. Default is 2.
            prefix_length (int): Prefix length for dictionary. Default is 7.
        """
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self.symspell = SymSpell(max_edit_distance, prefix_length)
        self._is_dictionary_built = False
    
    def build_dictionary_from_dataframe(
        self, 
        df: pd.DataFrame, 
        column_name: str,
        show_progress: bool = True
    ) -> 'SpellingCorrectorService':
        """
        Build SymSpell dictionary from a DataFrame column.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing the text data.
            column_name (str): Name of the column to build dictionary from.
            show_progress (bool): Whether to show progress bar. Default is True.
        
        Returns:
            SpellingCorrectorService: Self reference for method chaining.
        """
        # Reset the dictionary
        self.symspell = SymSpell(self.max_edit_distance, self.prefix_length)
        
        # Prepare list of words
        words = df[column_name].astype(str).str.strip().fillna("").tolist()
        
        # Add each word to SymSpell dictionary
        iterator = tqdm(words, desc="Building SymSpell dictionary") if show_progress else words
        
        for word in iterator:
            if word:  # skip empty strings
                self.symspell.create_dictionary_entry(word, 1)
        
        self._is_dictionary_built = True
        return self
    
    def build_dictionary_from_list(
        self, 
        words: List[str],
        show_progress: bool = True
    ) -> 'SpellingCorrectorService':
        """
        Build SymSpell dictionary from a list of words.
        
        Args:
            words (List[str]): List of words to build dictionary from.
            show_progress (bool): Whether to show progress bar. Default is True.
        
        Returns:
            SpellingCorrector: Self reference for method chaining.
        """
        # Reset the dictionary
        self.symspell = SymSpell(self.max_edit_distance, self.prefix_length)
        
        iterator = tqdm(words, desc="Building SymSpell dictionary") if show_progress else words
        
        for word in iterator:
            if word:  # skip empty strings
                self.symspell.create_dictionary_entry(word, 1)
        
        self._is_dictionary_built = True
        return self
    
    def build_dictionary_from_file(
        self,
        file_path: str,
        corpus_file: bool = True,
        separator: str = " ",
        show_progress: bool = True
    ) -> 'SpellingCorrectorService':
        """
        Build SymSpell dictionary from a file.
        
        Args:
            file_path (str): Path to the dictionary file.
            corpus_file (bool): Whether the file is a corpus file. Default is True.
            separator (str): Separator used in the file. Default is space.
            show_progress (bool): Whether to show progress (only for custom parsing).
        
        Returns:
            SpellingCorrector: Self reference for method chaining.
        """
        # Reset the dictionary
        self.symspell = SymSpell(self.max_edit_distance, self.prefix_length)
        
        if corpus_file:
            # Use built-in corpus loading
            self.symspell.load_dictionary(file_path, 0, 1, separator)
        else:
            # Load as plain text file
            with open(file_path, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
            
            iterator = tqdm(words, desc="Building SymSpell dictionary") if show_progress else words
            
            for word in iterator:
                if word:
                    self.symspell.create_dictionary_entry(word, 1)
        
        self._is_dictionary_built = True
        return self
    
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
        
        suggestions = self.symspell.lookup(
            word, 
            verbosity=Verbosity.CLOSEST, 
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
            suggestions = self.symspell.lookup(
                word, 
                verbosity=Verbosity.CLOSEST, 
                max_edit_distance=edit_distance
            )
            if suggestions:
                corrected.append(suggestions[0].term)
            else:
                corrected.append(word)
        
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
        if not self._is_dictionary_built:
            raise ValueError("Dictionary not built. Call one of the build_dictionary_* methods first.")
        
        edit_distance = max_edit_distance if max_edit_distance is not None else self.max_edit_distance
        corrected_col = []
        
        words = df[column_name].astype(str).fillna("")
        iterator = tqdm(words, desc=f"Correcting column '{column_name}'") if show_progress else words
        
        for word in iterator:
            suggestions = self.symspell.lookup(
                word, 
                verbosity=Verbosity.CLOSEST, 
                max_edit_distance=edit_distance
            )
            if suggestions:
                corrected_col.append(suggestions[0].term)
            else:
                corrected_col.append(word)
        
        corrected_series = pd.Series(corrected_col, index=df.index)
        
        if inplace:
            target_column = new_column_name if new_column_name else column_name
            df[target_column] = corrected_series
            return df
        elif new_column_name:
            result_df = df.copy()
            result_df[new_column_name] = corrected_series
            return result_df
        else:
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


# Example usage
if __name__ == "__main__":
    # Example 1: Using with DataFrame
    print("Example 1: Correcting DataFrame column")
    print("-" * 50)
    
    # Create sample data
    df = pd.DataFrame({
        'misspelled': ['helo', 'wrld', 'pyhton', 'machne', 'lerning'],
        'correct': ['hello', 'world', 'python', 'machine', 'learning']
    })
    
    # Initialize corrector
    corrector = SpellingCorrectorService(max_edit_distance=2, prefix_length=7)
    
    # Build dictionary from correct words
    corrector.build_dictionary_from_dataframe(df, 'correct', show_progress=False)
    
    # Correct the misspelled column
    df['corrected'] = corrector.correct_dataframe_column(
        df, 'misspelled', show_progress=False
    )
    
    # Evaluate
    results = corrector.evaluate_corrections(df, 'misspelled', 'correct', 'corrected')
    print(f"\nTotal: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print("\nDataFrame:")
    print(df)
    
    # Example 2: Using with list
    print("\n\nExample 2: Correcting a list of words")
    print("-" * 50)
    
    correct_words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    misspelled_words = ['aple', 'bannana', 'chery', 'dat', 'elderberr']
    
    corrector2 = SpellingCorrectorService(max_edit_distance=2)
    corrector2.build_dictionary_from_list(correct_words, show_progress=False)
    
    corrected_words = corrector2.correct_list(misspelled_words, show_progress=False)
    
    print("\nOriginal -> Corrected:")
    for orig, corr in zip(misspelled_words, corrected_words):
        print(f"{orig:15} -> {corr}")
