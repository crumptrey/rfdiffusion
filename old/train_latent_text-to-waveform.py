from audio_diffusion_pytorch.audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from audio_encoders_pytorch.audio_encoders_pytorch import AutoEncoder1d, VariationalBottleneck, Encoder1d, Decoder1d, TanhBottleneck
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
# Example usage

# Hyperparameters
batch_size = 128
num_epochs = 100
learning_rate = 1e-4
latent_dim = 256
in_channels = 2
channels = 64
multipliers = [1, 2, 4]
factors = [2, 2]
num_blocks = [2, 2]
patch_size = 4
project = 'rfdiffusion_2018.01a_vae+diffusion'
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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Initialize model
# Parameters for the VAE
in_channels = 2
channels = 64
multipliers = [1, 2, 4]
factors = [2, 2]
num_blocks = [2, 2]
patch_size = 4
model = DiffusionModel(
    net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
    in_channels=64,  # U-Net: number of input/output (audio) channels
    channels=[64, 128, 256, 512],  # U-Net: channels at each layer
    factors=[2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 2],  # U-Net: number of repeating items at each layer
    attentions=[1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    cross_attentions=[0, 0, 0, 0],
    context_channels=[0, 0, 0, 0],
    attention_heads=4,  # U-Net: number of attention heads per attention item
    attention_features=32,  # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion,  # The diffusion method used
    sampler_t=VSampler,  # The diffusion sampler used
    use_text_conditioning=True,  # U-Net: enables text conditioning (default T5-base)
    use_embedding_cfg=True,  # U-Net: enables classifier free guidance
    embedding_max_length=64,  # U-Net: text embedding maximum length (default for T5-base)
    embedding_features=768,  # U-Net: text embedding features (default for T5-base)
).to(device)

ae = AutoEncoder1d(
    in_channels=in_channels,
    channels=channels,
    multipliers=multipliers,
    factors=factors,
    num_blocks=num_blocks,
    patch_size=patch_size,
    resnet_groups=8,
    bottleneck=TanhBottleneck(),
).to(device)

# Load the checkpoint
checkpoint_path = '/home/trey/experiment_rfdiffusion/models/saved_models/rfdiffusion_2018.01a_vae/vae_model_epoch_15.pth'
checkpoint = torch.load(checkpoint_path)

# Load the state dictionary into the model
ae.load_state_dict(checkpoint['model_state_dict'])

# Set model to training mode
model.train()

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)
# Initialize list to store average loss per epoch
average_loss_per_epoch = []

# Training loop
for epoch in range(num_epochs):
    epoch_losses = []
    for batch_idx, (x, mod, snr) in enumerate(data_loader):
        # Generate prompts for each example in the batch
        prompts = []
        #actual_snr = train_SNRs[snr]

        for i in range(x.size()[0]):
            modulation = train_modulations[mod[i].item()]
            #prompt = f"{modulation} modulated waveform at {actual_snr[i]} dB SNR"
            prompt = f"{modulation} modulated waveform at {snr[i]} dB SNR"
            prompts.append(prompt)

        # Create random input data
        x = x.to(device)  # Move to GPU if available
        print(x.size())
        latent = ae.encode(x)
        print(latent.size())
        # Calculate loss
        loss = model(latent, text=prompts, embedding_mask_proba=0.1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Store batch loss
        epoch_losses.append(loss.item())

    # Calculate and store average loss for the epoch
    average_loss = np.mean(epoch_losses)
    average_loss_per_epoch.append(average_loss)
    print(f'Epoch: {epoch}, Average Loss: {average_loss}')

    # Log average loss to wandb
    wandb.log({"epoch": epoch, "average_loss": average_loss})

    # Save the model and optimizer state dictionaries after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, os.path.join(model_save_dir, f'model_epoch_{epoch}.pth'))

print("Training complete and models saved.")

# Plot average loss per epoch using wandb
wandb.log({"average_loss_per_epoch": average_loss_per_epoch})

# Finish the wandb run
wandb.finish()