from torch.utils.data import Dataset
import torch
import numpy as np

class TextDataset(Dataset):
    def __init__(self, dataframe, text_col, label_col):
        self.len = len(dataframe)
        self.data = dataframe
        self.text_col = text_col
        self.targets = self.data[label_col].to_numpy().astype(np.float32)
        self.texts = self.data[text_col]

    def __getitem__(self, index):
        text_inputs = self.texts[index]
        targets = self.targets[index]
        return text_inputs, targets, index    
    

    def __len__(self):
        return self.len
