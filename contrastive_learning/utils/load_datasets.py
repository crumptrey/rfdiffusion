import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image
import pickle
import numpy as np
import h5py
import json
import random
from os.path import dirname, join as pjoin
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
class PowerNormalization(torch.nn.Module):
    def __init__(self):
        super(PowerNormalization, self).__init__()

    def forward(self, signals):
        """
        Perform L2 normalization on a signal array of shape (batch_size, 2, N).

        Parameters:
            signals (torch.Tensor): 3D tensor of shape (batch_size, 2, N).

        Returns:
            torch.Tensor: Normalized signals with each row normalized to have a unit Euclidean norm.
        """
        # Compute the L2 norm along the second dimension (dim=1).
        norms = torch.norm(signals, p=2, dim=1, keepdim=True)
        
        # Perform L2 normalization with broadcasting.
        normalized_signals = signals / norms

        return normalized_signals

def plot_class_distribution(dataset, dataset_name, directory):
    modulations = [item[1] for item in dataset]
    snrs = [item[2] for item in dataset]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(modulations, bins=len(set(modulations)), color='skyblue', edgecolor='black')
    plt.title('Modulation Types Distribution for ' + dataset_name)
    plt.xlabel('Modulation Type')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(snrs, bins=len(set(snrs)), color='salmon', edgecolor='black')
    plt.title('SNR Distribution for ' + dataset_name)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    plt.savefig("{0}/dataset_{1}.png".format(directory, dataset_name))
    plt.close()  # Clear the plot to release memory


def balanced_split(data, test_data, split, train_modulations, train_SNRs, TRAIN_MOD_TYPES,
                   test_modulations, test_SNRs, TEST_MOD_TYPES):
    """
    Split the data into training and validation datasets with balanced modulation classes and SNRs.

    Parameters:
        data (list): List of tuples (data, mod_type, snr).
        split (list): List containing the proportions for training, validation, and test datasets.

    Returns:
        tuple: Tuple containing training and validation datasets.
    """
    train_size = int(split[0] * len(data) / (len(train_modulations) * len(train_SNRs)))
    valid_size = int(split[1] * len(data) / (len(train_modulations) * len(train_SNRs)))
    test_size = int(split[2] * len(data) / (len(test_modulations) * len(test_SNRs)))
    print(train_size)
    print(valid_size)
    print(test_size)
    # Shuffle the data
    random.shuffle(data)
    random.shuffle(test_data)

    # Separate modulation classes and SNRs
    modulations = set(item[1] for item in data)
    snrs = set(item[2] for item in data)

    test_mods = set(item[1] for item in test_data)
    test_snrs = set(item[2] for item in test_data)

    # Initialize dictionaries to store selected samples for each dataset
    train_samples = {mod: {snr: [] for snr in snrs} for mod in modulations}
    valid_samples = {mod: {snr: [] for snr in snrs} for mod in modulations}
    test_samples = {mod: {snr: [] for snr in test_snrs} for mod in test_mods}
    # Iterate over data and distribute samples evenly among datasets
    for item in data:
        mod_type, snr = item[1], item[2]
        if len(train_samples[mod_type][snr]) < train_size:
            train_samples[mod_type][snr].append(item)
        elif len(valid_samples[mod_type][snr]) < valid_size:
            valid_samples[mod_type][snr].append(item)
    for item in test_data:
        mod_type, snr = item[1], item[2]
        if len(test_samples[mod_type][snr]) < test_size:
            test_samples[mod_type][snr].append(item)
    # Print number of samples per modulation type and SNR
    print("Train samples per modulation type and SNR:")
    for mod in train_modulations:
        for snr in train_SNRs:
            mods = TRAIN_MOD_TYPES.index(mod)
            num_samples = len(train_samples[mods][snr])
            print(f"Modulation: {mod}, SNR: {snr}, Num Samples: {num_samples}")

    print("\nValid samples per modulation type and SNR:")
    for mod in train_modulations:
        for snr in train_SNRs:
            mods = TRAIN_MOD_TYPES.index(mod)
            num_samples = len(valid_samples[mods][snr])
            print(f"Modulation: {mod}, SNR: {snr}, Num Samples: {num_samples}")

    print("\nTest samples per modulation type and SNR:")
    for mod in test_modulations:
        for snr in test_SNRs:
            print(mod)
            mods = TEST_MOD_TYPES.index(mod)
            num_samples = len(test_samples[mods][snr])
            print(f"Modulation: {mod}, SNR: {snr}, Num Samples: {num_samples}")
    # Flatten the dictionaries to get the final datasets
    train_dataset = [item for sublist in train_samples.values() for subsublist in sublist.values() for item in
                     subsublist]
    valid_dataset = [item for sublist in valid_samples.values() for subsublist in sublist.values() for item in
                     subsublist]
    test_dataset = [item for sublist in test_samples.values() for subsublist in sublist.values() for item in
                    subsublist]
    return train_dataset, valid_dataset, test_dataset

class SignalDataset(Dataset):
    def __init__(self, hdf5_file, transform):
        # Open the HDF5 file
        self.modulationType = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
        self.hdf5_file = hdf5_file
        with h5py.File(hdf5_file, 'r') as hdf:
            # Load datasets
            self.X = hdf['/X'][:]  # Shape: [2, 1024, total_samples]
            self.modulation_classes = hdf['/mod'][:]  # Shape: [total_samples]
            self.snr_levels = hdf['/snr'][:]  # Shape: [total_samples]
        self.transform = transform

    def __len__(self):
        # Return the number of samples
        return self.X.shape[0]  # Total samples

    def __getitem__(self, index):
        # Get the signal and its corresponding modulation class and SNR level
        signal = self.X[index]  # Shape: [2, 1024]
        mod_class = self.modulation_classes[:,index]  # Get modulation class
        snr_level = self.snr_levels[:,index]  # Get SNR level
        mod_type = int(mod_class) - 1
        mod_type = self.modulationType[mod_type]
        snr_level = int(snr_level)
        prompt = '{0} modulated signal at {1} dB SNR'.format(mod_type, snr_level)
        signal = torch.from_numpy(signal)
        signal = signal.permute(1,0)
        signal = signal.float()
        if self.transform:
            signal = self.transform(signal)
        return signal, prompt

def getDataset(train_name, test_name, train_modulations, train_SNRs, test_modulations, test_SNRs, split, directory,
               train_transforms, test_transforms):
    if split is None:
        split = [0.75, 0.125, 0.125]

    train_dataset = None
    valid_dataset = None
    test_dataset = None

    if train_name in ["2018.01A", "2016.10A", "2016.04C"]:
        # First load unfiltered datasets
        if train_name == "2018.01A":
            dataDirectory = "/home/trey/projectModClass/datasets/2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5"
            TRAIN_MOD_TYPES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                               '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
                               'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
            dataset = DeepSig2018Dataset(dataDirectory, train_transforms)

        elif train_name == "2016.10A":
            TRAIN_MOD_TYPES = ['AM-SSB', 'CPFSK', 'QPSK', 'GFSK', 'PAM4', 'QAM16', 'WBFM', '8PSK', 'QAM64', 'AM-DSB',
                               'BPSK']
            dataDirectory = "/home/trey/projectModClass/datasets/2016.10A/RML2016.10a_dict.pkl"
            dataset = DeepSig2016Dataset(dataDirectory, train_transforms)

        elif train_name == "2016.04C":
            TRAIN_MOD_TYPES = ['AM-SSB', 'CPFSK', 'QPSK', 'GFSK', 'PAM4', 'QAM16', 'WBFM', '8PSK', 'QAM64', 'AM-DSB',
                               'BPSK']
            dataDirectory = "/home/trey/projectModClass/datasets/2016.04C/RML2016.04C.multisnr.pkl"
            dataset = DeepSig2016Dataset(dataDirectory, train_transforms)


    # Perform balanced splitting of the training dataset
    #train_dataset, valid_dataset, test_dataset = balanced_split(filtered_train_data, filtered_test_data, split,
    #                                                            train_modulations, train_SNRs, TRAIN_MOD_TYPES,
     #                                                           test_modulations, test_SNRs, TEST_MOD_TYPES)
    #plot_class_distribution(train_dataset, 'Train', directory)
    #plot_class_distribution(valid_dataset, 'Valid', directory)
    #plot_class_distribution(test_dataset, 'Test', directory)
    return dataset


class MATLABDataset(Dataset):
    def __init__(self, file_dir, transform=None):
        self.file_dir = file_dir
        self.transform = transform
        self.mat_files = [file for file in os.listdir(file_dir) if file.endswith('.mat')]

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mat_file_path = os.path.join(self.file_dir, self.mat_files[idx])
        mat_data = sio.loadmat(mat_file_path)  # Load .mat file
        frame = mat_data['frame']  # Access frame data from the loaded .mat file
        label = mat_data['label']  # Access label from the loaded .mat file
        SNR = mat_data['SNR']  # Access SNR from the loaded .mat file
        if self.transform:
            frame = self.transform(frame)
        return frame, label, SNR


def dataset_split(data,
                  modulations_classes,
                  modulations, snrs,
                  target_modulations,
                  mode,
                  target_snrs, train_proportion=0.7,
                  valid_proportion=0.2,
                  test_proportion=0.1,
                  seed=48):
    np.random.seed(seed)
    train_split_index = int(train_proportion * 4096)
    valid_split_index = int((valid_proportion + train_proportion) * 4096)
    test_split_index = int((test_proportion + valid_proportion + train_proportion) * 4096)
    X_output = []
    Y_output = []
    Z_output = []

    target_modulation_indices = [modulations_classes.index(modu) for modu in target_modulations]

    for modu in target_modulation_indices:
        for snr in target_snrs:
            snr_modu_indices = np.where((modulations == modu) & (snrs == snr))[0]

            np.random.shuffle(snr_modu_indices)
            train, valid, test, remaining = np.split(snr_modu_indices,
                                                     [train_split_index, valid_split_index,
                                                      test_split_index])
            if mode == 'train':
                X_output.append(data[np.sort(train)])
                Y_output.append(modulations[np.sort(train)])
                Z_output.append(snrs[np.sort(train)])
            elif mode == 'valid':
                X_output.append(data[np.sort(valid)])
                Y_output.append(modulations[np.sort(valid)])
                Z_output.append(snrs[np.sort(valid)])
            elif mode == 'test':
                X_output.append(data[np.sort(test)])
                Y_output.append(modulations[np.sort(test)])
                Z_output.append(snrs[np.sort(test)])
            else:
                raise ValueError(f'unknown mode: {mode}. Valid modes are train, valid and test')
    X_array = np.vstack(X_output)
    Y_array = np.concatenate(Y_output)
    Z_array = np.concatenate(Z_output)
    for index, value in enumerate(np.unique(np.copy(Y_array))):
        Y_array[Y_array == value] = index
    return X_array, Y_array, Z_array


class DeepSig2018Dataset_O(Dataset):
    def __init__(self, file_dir, mode: str, seed=48):
        nf_train = 1024
        nf_valid = 1024
        nf_test = 1024
        self.file_dir = file_dir
        self.modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                               '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
                               'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        # load data
        hdf5_file = h5py.File(self.file_dir, 'r')
        self.X = hdf5_file['X']
        self.Y = np.argmax(hdf5_file['Y'], axis=1)
        print(self.X.shape[0])
        self.Z = hdf5_file['Z'][:, 0]
        train_proportion = (24 * 26 * nf_train) / self.X.shape[0]
        valid_proportion = (24 * 26 * nf_valid) / self.X.shape[0]
        test_proportion = (24 * 26 * nf_test) / self.X.shape[0]
        #self.target_modulations = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '256QAM']
        self.target_modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                               '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
                               'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        self.target_snrs = np.unique(self.Z)

        self.X_data, self.Y_data, self.Z_data = dataset_split(
            data=self.X,
            modulations_classes=self.modulation_classes,
            modulations=self.Y,
            snrs=self.Z,
            mode=mode,
            train_proportion=0.75,
            valid_proportion=0.125,
            test_proportion=0.125,
            target_modulations=self.target_modulations,
            target_snrs=self.target_snrs,
            seed=seed
        )

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x, y, z = self.X_data[idx], self.Y_data[idx], self.Z_data[idx]
        x, y, z = torch.Tensor(x), y, z
        return x, y, z


class DeepSig2016Dataset(Dataset):
    def __init__(self, file_dir, transform=None):
        self.file_dir = file_dir
        self.transform = transform
        with open(self.file_dir, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.pickle = u.load()
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.pickle.keys())))), [1, 0])
        self.MOD_TYPES = {'AM-SSB', 'CPFSK', 'QPSK', 'GFSK', 'PAM4', 'QAM16', 'WBFM', '8PSK', 'QAM64', 'AM-DSB',
                          'BPSK'}
        X = []
        self.lbl = []
        for mod in mods:
            for snr in snrs:
                X.append(self.pickle[(mod, snr)])
                for i in range(self.pickle[(mod, snr)].shape[0]):
                    self.lbl.append((mod, snr))
        self.X = np.vstack(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        mod = self.lbl[idx][0]
        mod_type = list(self.MOD_TYPES).index(mod)
        snr = self.lbl[idx][1]
        train_SNRs = np.arange(-20, 19, 2)
        snr = np.where(train_SNRs == snr)[0].item()
        if self.transform:
            x = self.transform(x)
        #x = torch.permute(x, [1, 0])
        prompt = '{0} modulated signal at {1} dB SNR'.format(mod_type, snr)
        return x, prompt


class DeepSig2018Dataset(Dataset):
    def __init__(self, file_dir, transform=None):
        self.file_dir = file_dir
        self.transform = transform
        hdf5_file = h5py.File(self.file_dir, 'r')
        #self.X = hdf5_file['X']
        #self.Y = np.argmax(hdf5_file['Y'], axis=1)
        #self.Z = hdf5_file['Z'][:, 0]
        self.TRAIN_MOD_TYPES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                               '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
                               'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        Z = hdf5_file['Z'][:, 0]
        snr_min = 0
        if snr_min is not None:
            indices = np.where((Z >= snr_min))[0]
        else:
            indices = np.arange(len(Z))
        self.X = hdf5_file['X'][indices]
        self.Y = np.argmax(hdf5_file['Y'][indices], axis=1)
        self.Z = Z[indices]
        Z = Z[indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y, z = self.X[idx], self.Y[idx], self.Z[idx]
        x, y, z = torch.Tensor(x), y, z
        if self.transform:
            x = self.transform(x)
        x = torch.permute(x, [1, 0])
        y = self.TRAIN_MOD_TYPES[y]
        #prompt = '{0} modulated signal at {1} dB SNR'.format(y,z)
        prompt = '{0} modulated signal at {1} dB SNR'.format(y, z)
        return x, prompt

class DeepSig2018Dataset_MOD(Dataset):
    def __init__(self, file_dir, transform=None):
        self.file_dir = file_dir
        self.transform = transform
        hdf5_file = h5py.File(self.file_dir, 'r')
        Z = hdf5_file['Z'][:, 0]
        self.TRAIN_MOD_TYPES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                               '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
                               'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        snr_min = 0
        if snr_min is not None:
            indices = np.where((Z >= snr_min))[0]
        else:
            indices = np.arange(len(Z))

        self.X = hdf5_file['X'][indices]
        self.Y = np.argmax(hdf5_file['Y'][indices], axis=1)
        self.Z = Z[indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y, z = self.X[idx], self.Y[idx], self.Z[idx]
        x, y, z = torch.Tensor(x), y, z
        if self.transform:
            x = self.transform(x)
        x = torch.permute(x, [1, 0])
        #y = self.TRAIN_MOD_TYPES[y]
        return x, y, z
