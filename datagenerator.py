# Import necessary libraries
from torch.utils.data import Dataset
import torch


"""
Sanskriti Singh
Define my custom dataset class for handling the data that was created in data processing for pytorch model
"""
class CSVDataset(Dataset):
    def __init__(self, x_data, y_data):
        # Convert input features (x_data) and labels (y_data) to PyTorch tensors
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

        # one hot encoding
        self.y_data = torch.nn.functional.one_hot(self.y_data, num_classes=8).to(float)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        # Retrieve a single sample and its corresponding label based on the index
        x = self.x_data[idx]
        y = self.y_data[idx]

        # Return the tuple (input features, label) for the specified index
        return x, y
