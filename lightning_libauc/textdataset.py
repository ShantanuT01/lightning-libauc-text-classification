"""Dataset definitions for LibAUC text classification."""

from torch.utils.data import Dataset
import torch
import numpy as np

class TextDataset(Dataset):
    """Dataset wrapper for text inputs and binary labels."""

    def __init__(self, dataframe, text_col, label_col):
        """Initialize dataset fields from a dataframe.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            text_col (str): Column name containing text.
            label_col (str): Column name containing labels.
        """
        self.len = len(dataframe)
        self.data = dataframe
        self.text_col = text_col
        self.targets = self.data[label_col].to_numpy().astype(np.float32)
        self.texts = self.data[text_col]

    def __getitem__(self, index):
        """Return a single sample by index.

        Args:
            index (int): Sample index.

        Returns:
            item (Tuple[str, float, int]): Text, label, and index.
        """
        text_inputs = self.texts[index]
        targets = self.targets[index]
        return text_inputs, targets, index    
    

    def __len__(self):
        """Return dataset length.

        Returns:
            int (int): Number of samples.
        """
        return self.len
