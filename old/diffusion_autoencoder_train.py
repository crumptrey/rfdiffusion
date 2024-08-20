from audio_diffusion_pytorch.audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler
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
from torch.optim import Adam
import os
from torchsummary import summary
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
batch_size = 128
num_epochs = 100
learning_rate = 1e-4
# Initialize wandb

hps = {
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "optimizer": "Adam"
           }
accelerator.init_trackers('rfdiffusion_custom', config=hps)

dataDir = '/home/trey/Documents/MATLAB/ModulationClassificationWithDeepLearningExample/data'
adam_betas = (0.9, 0.999)
model_save_dir = '/home/trey/experiment_rfdiffusion/models/saved_models/custom'

# Create the directory if it does not exist
os.makedirs(model_save_dir, exist_ok=True)
# Define data transformations
train_transforms = transforms.Compose([net_transforms.PowerNormalization()])
test_transforms = train_transforms

# Load datasets
train_dataset = utils.load_datasets.SignalDataset(
    dataDir, train_transforms
)

# Create data loader
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
model = DiffusionAE(
    encoder=MelE1d( # The encoder used, in this case a mel-spectrogram encoder
        in_channels=2,
        channels=512,
        multipliers=[1, 1],
        factors=[2],
        num_blocks=[12],
        out_channels=32,
        mel_channels=80,
        mel_sample_rate=48000,
        mel_normalize_log=True,
        bottleneck=TanhBottleneck(),
    ),
    inject_depth=6,
    net_t=UNetV0, # The model type used for diffusion upsampling
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
)

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=learning_rate, betas=adam_betas)

# Prepare everything with accelerator
model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
num_training_steps = num_epochs * len(data_loader)

progress_bar = tqdm(range(num_training_steps))
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Set model to training mode
model.train()

# Log hyperparameters with wandb

# Initialize list to store average loss per epoch
average_loss_per_epoch = []
step = 0
# Training loop
for epoch in range(num_epochs):
    epoch_losses = []
    for x, prompt in data_loader:
        optimizer.zero_grad()

        loss = model(x)

        accelerator.backward(loss)
        optimizer.step()

        progress_bar.update(1)
        accelerator.log({"training_loss": loss}, step=step)
        step += 1

    if accelerator.is_main_process:  # Ensure only the main process saves
        checkpoint_path = os.path.join(model_save_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
accelerator.end_training()
print("Training complete and models saved.")

