from audio_diffusion_pytorch.audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion_Inpainted, VSampler
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

batch_size = 256
num_epochs = 50
learning_rate = 1e-4
# Initialize wandb
wandb.init(project='rfdiffusion_impainting',
           config = {
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "optimizer": "Adam"
           })

# Define hyperparameters
train_modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                     '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
                     'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
train_SNRs = np.arange(-20, 32, 2)
test_modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                    '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
                    'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
test_SNRs = np.arange(-20, 30, 2)
dataset_train_name = '2016.10A'
dataset_test_name = '2016.10A'
dataDir = '/home/trey/experiment_rfdiffusion/models/saved_models/impainting'
adam_betas = (0.9, 0.999)
train_modulations = ['AM-SSB', 'CPFSK', 'QPSK', 'GFSK', 'PAM4', 'QAM16', 'WBFM', '8PSK', 'QAM64', 'AM-DSB', 'BPSK']
train_SNRs = np.arange(-20, 19, 2)
model_save_dir = '/home/trey/experiment_rfdiffusion/models/saved_models/impainting'
# Create directories if not exist
utils.training.create_directory(dataDir)

# Define data split ratios
split = [0.75, 0.05, 0.20]

# Define data transformations
train_transforms = transforms.Compose([net_transforms.PowerNormalization()])
test_transforms = train_transforms

# Load datasets
train_dataset = utils.load_datasets.getDataset(
    dataset_train_name, dataset_test_name, train_modulations, train_SNRs, test_modulations, test_SNRs, split, dataDir, train_transforms, test_transforms
)

# Create data loader
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = DiffusionModel(
    net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
    in_channels=2,  # U-Net: number of input/output (audio) channels
    channels=[64, 128, 256, 512],  # U-Net: channels at each layer
    factors=[2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 2],  # U-Net: number of repeating items at each layer
    attentions=[1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    attention_heads=4,  # U-Net: number of attention heads per attention item
    attention_features=32,  # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion_Inpainted,  # The diffusion method used
    sampler_t=VSampler,  # The diffusion sampler used
    use_text_conditioning=False,  # U-Net: enables text conditioning (default T5-base)
    use_embedding_cfg=False,  # U-Net: enables classifier free guidance
).to(device)
# Initialize model
# Initialize optimizer
optimizer = Adam(net.parameters(), lr=learning_rate, betas=adam_betas)
print(f"Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

# Set model to training mode
net.train()

# Log hyperparameters with wandb

# Initialize list to store average loss per epoch
average_loss_per_epoch = []

# Training loop
for epoch in range(num_epochs):
    epoch_losses = []
    for batch_idx, (x, mod, snr) in enumerate(data_loader):
        # Generate prompts for each example in the batch

        # Create random input data
        x = x.to(device)  # Move to GPU if available

        # Calculate loss
        loss = net(x)
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
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, os.path.join(model_save_dir, f'model_epoch_{epoch}.pth'))

print("Training complete and models saved.")

# Plot average loss per epoch using wandb
wandb.log({"average_loss_per_epoch": average_loss_per_epoch})

# Finish the wandb run
wandb.finish()
