import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SignalDataset(Dataset):
    def __init__(self, hdf5_directory, transform = None):
        self.hdf5_files = [os.path.join(hdf5_directory, file) for file in os.listdir(hdf5_directory) if file.endswith('.h5')]
        self.data_indices = []
        self.transform = transform
        self._create_index()

    def _create_index(self):
        for file_idx, file in enumerate(self.hdf5_files):
            with h5py.File(file, 'r') as f:
                num_signals = f['/rxSignals/real'].shape[1]  # Should be 500
                self.data_indices.extend([(file_idx, i) for i in range(num_signals)])

    def __len__(self):
        return len(self.data_indices)
    def __getitem__(self, idx):
        file_idx, signal_idx = self.data_indices[idx]
        with h5py.File(self.hdf5_files[file_idx], 'r') as f:
            real_part = f['/rxSignals/real'][:, signal_idx]
            imag_part = f['/rxSignals/imag'][:, signal_idx]
            signal = np.stack((real_part, imag_part), axis=0)
            signal = torch.from_numpy(signal).float()

            # Since the prompts dataset is 1x16, treat it as a 1D array
            prompt = str(f['/prompts'][0, signal_idx].decode('utf-8'))  # Access the 1st row, signal_idx-th column

            if self.transform:
                signal = self.transform(signal)

        return signal, prompt

    # Set the directory where your .mat files are stored
data_dir = '/ext/trey/experiment_diffusion/experiment_rfdiffusion/dataset/data'  # Adjust the path accordingly

# Create the dataset
signal_dataset = SignalDataset(data_dir)

# Create a DataLoader
signal_loader = DataLoader(signal_dataset, batch_size=1, shuffle=True)

# Example of iterating through the DataLoader
for batch in signal_loader:
    print(batch)  # This will print the signals in the batch
