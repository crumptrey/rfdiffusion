from audio_diffusion_pytorch.audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from audio_encoders_pytorch.audio_encoders_pytorch import AutoEncoder1d, VariationalBottleneck, Encoder1d, Decoder1d, TanhBottleneck, MelE1d
import utils.load_datasets
import utils.training
import utils.logging
from networks import *
import networks.transforms as net_transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import wandb
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

def prefix_dict(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """Prefixes all keys in a dictionary with a given string."""
    return {f"{prefix}{k}": v for k, v in d.items()}

# Hyperparameters
batch_size = 128
num_epochs = 100
learning_rate = 1e-4
latent_dim = 256  # This might need adjustment
in_channels = 2
channels = 256  # Starting channel count
multipliers = [1, 2, 2, 2, 4, 4, 4]  # To achieve [256, 512, 512, 512, 1024, 1024, 1024]
factors = [2, 2, 2, 2, 2, 2]  # Downsampling factors
num_blocks = [2, 2, 2, 2, 2, 2]  # Repetitions of ResNet blocks
patch_size = 1  # Start with no patching
project = 'rfdiffusion_2018.01a_vae'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# Initialize wandb
wandb.init(project=project,
           config={
               "learning_rate": learning_rate,
               "epochs": num_epochs,
               "batch_size": batch_size,
               "optimizer": "Adam",
               "latent_dim": latent_dim,
               "channels": channels,
               "multipliers": multipliers,
               "factors": factors,
               "num_blocks":num_blocks,
               "patch_size":patch_size
           })
model = AutoEncoder1d(
    in_channels=in_channels,
    channels=channels,
    multipliers=multipliers,
    factors=factors,
    num_blocks=num_blocks,
    patch_size=patch_size,
    resnet_groups=8,
    bottleneck=TanhBottleneck(),
).to(device)
# Dataset and DataLoader setup (using your existing code)
train_modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                     '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
                     'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
train_SNRs = np.arange(-20, 32, 2)
dataset_train_name = '2018.01A'
dataset_test_name = '2018.01A'
dataDir = '/home/trey/experiment_rfdiffusion/models/saved_models/{0}'.format(project)
model_save_dir = '/home/trey/experiment_rfdiffusion/models/saved_models/{0}'.format(project)

utils.training.create_directory(dataDir)

split = [0.75, 0.05, 0.20]
train_transforms = transforms.Compose([net_transforms.PowerNormalization()])
test_transforms = train_transforms

train_dataset = utils.load_datasets.getDataset(
    dataset_train_name, dataset_test_name, train_modulations, train_SNRs, train_modulations, train_SNRs,
    split, dataDir, train_transforms, test_transforms
)

data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set device


# Initialize model
# Parameters for the VAE


# Initialize optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    for batch_idx, (waveform, mod, snr) in enumerate(data_loader):
        waveform = waveform.to(device)


        y = model(waveform, with_info=False)
        loss = criterion(y, waveform)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    average_loss = np.mean(epoch_losses)
    print(f'Epoch: {epoch}, Average Loss: {average_loss}')

    # Log to wandb
    wandb.log({"epoch": epoch, "average_loss": average_loss})

    # Save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': average_loss,
    }, os.path.join(model_save_dir, f'vae_model_epoch_{epoch}.pth'))

print("Training complete and models saved.")

# Finish the wandb run
wandb.finish()