#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, random_split
from audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck, NoiserBottleneck, VariationalBottleneck
import utils.load_datasets
import torch.nn.functional as F
import os
from tqdm import tqdm
from accelerate import Accelerator
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from torchaudio.transforms import Spectrogram
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer


os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"
import wandb
def plot_latent_space(latent_representations, labels, idx=0, save_dir="plots", save_name="latent_space"):
    """
    Plot the latent space of the encoded signals using different methods.

    Parameters:
    - latent_representations: The encoded latent representations.
    - idx: Index for distinguishing multiple plots if needed.
    """
    latent_representations = latent_representations.cpu().detach().numpy()
    # Flatten latent representations if necessary
    if latent_representations.ndim > 2:
        num_samples = latent_representations.shape[0]
        num_features = np.prod(latent_representations.shape[1:])
        latent_representations = latent_representations.reshape(num_samples, num_features)

    os.makedirs(save_dir, exist_ok=True)

    # Create a color map for unique labels
    unique_labels = list(set(labels))
    color_map = plt.cm.get_cmap('tab20')  # You can change 'tab20' to any other colormap
    color_dict = {label: color_map(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

    # Create a 2D histogram (heatmap) from the latent space data
    plt.figure(figsize=(10, 8))
    heatmap, xedges, yedges = np.histogram2d(latent_representations[:, 0], latent_representations[:, 1], bins=50,
                                             range=[[-1, 1], [-1, 1]])

    # Plot heatmap
    sns.heatmap(heatmap.T, cmap='viridis', cbar=True, xticklabels=50, yticklabels=50)

    plt.title('Latent Space Heatmap')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig(os.path.join(save_dir, f"{save_name}_heatmap_{idx}.jpg"), format="jpg")
    plt.close()

    max_samples = 10000  # Adjust this based on memory constraints
    if latent_representations.shape[0] > max_samples:
        indices = np.random.choice(latent_representations.shape[0], max_samples, replace=False)
        latent_representations = latent_representations[indices]
        labels = [labels[i] for i in indices]

    pca_result = PCA(n_components=2).fit_transform(latent_representations)
    tsne_result = TSNE(n_components=2, random_state=42).fit_transform(latent_representations)

    # Plot configurations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Function to create scatter plot with unique colors
    def scatter_with_legend(ax, x, y, labels, title):
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(x[mask], y[mask], c=[color_dict[label]], label=label, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True)
        ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Simple plot (first two dimensions)
    scatter_with_legend(axes[0], latent_representations[:, 0], latent_representations[:, 1], labels,
                        f'Latent Space - Simple (Pair {idx + 1})')

    # PCA plot
    scatter_with_legend(axes[1], pca_result[:, 0], pca_result[:, 1], labels,
                        f'Latent Space - PCA (Pair {idx + 1})')

    # t-SNE plot
    scatter_with_legend(axes[2], tsne_result[:, 0], tsne_result[:, 1], labels,
                        f'Latent Space - t-SNE (Pair {idx + 1})')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_name}_{idx}.jpg"), format="jpg", bbox_inches='tight')
    plt.close()


def plot_waveform_and_spectrogram(input_signal, decoded_signal, idx, save_dir="plots",
                                  save_name="waveform_spectrogram"):
    """
    Plot the waveform and spectrogram of the input and decoded signals.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Time-domain signals
    axes[0, 0].plot(input_signal.cpu().numpy().flatten(), color='blue')
    axes[0, 0].set_title(f'Input Signal - Pair {idx + 1}')

    axes[0, 1].plot(decoded_signal.cpu().numpy().flatten(), color='red')
    axes[0, 1].set_title(f'Decoded Signal - Pair {idx + 1}')

    # Spectrograms
    spectrogram_transform = Spectrogram(n_fft=1024).to(input_signal.device)

    input_spectrogram = spectrogram_transform(input_signal).log2()[0, :, :].detach().cpu().numpy()
    decoded_spectrogram = spectrogram_transform(decoded_signal).log2()[0, :, :].detach().cpu().numpy()

    axes[1, 0].imshow(input_spectrogram, aspect='auto', origin='lower')
    axes[1, 0].set_title(f'Input Spectrogram - Pair {idx + 1}')

    axes[1, 1].imshow(decoded_spectrogram, aspect='auto', origin='lower')
    axes[1, 1].set_title(f'Decoded Spectrogram - Pair {idx + 1}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_name}_{idx}.jpg"), format="jpg")
    plt.close()


def compute_spectrogram_loss(input_signal, decoded_signal):
    """
    Compute the reconstruction loss based on spectrograms of the input and decoded signals.

    Parameters:
    - input_signal: The original input signal.
    - decoded_signal: The signal reconstructed by the autoencoder.
    - config: Configuration dictionary with spectrogram parameters.

    Returns:
    - spectrogram_loss: The loss between the spectrograms of input and decoded signals.
    """
    spectrogram_transform = Spectrogram(n_fft=1024).to(input_signal.device)

    input_spectrogram = spectrogram_transform(input_signal)
    decoded_spectrogram = spectrogram_transform(decoded_signal)

    spectrogram_loss = F.mse_loss(decoded_spectrogram, input_spectrogram)
    return spectrogram_loss


def setup_dataloader(config, val_split=0.2):
    dataset = utils.load_datasets.DeepSig2018Dataset_MOD(config["dataset_path"])
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True,
                              num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=config["num_workers"])

    return train_loader, val_loader

def setup_model(config):
    return AutoEncoder1d(
        in_channels=config['ae_in_channels'],
        channels=config['ae_channels'],
        multipliers=config['ae_multipliers'],
        factors=config['ae_factors'],
        num_blocks=config['ae_num_blocks'],
        patch_size=config['ae_patch_size'],
        resnet_groups=config['ae_resnet_groups'],
        bottleneck=VariationalBottleneck(channels=config["vae_channels"], loss_weight=config["vae_loss_scale"])
    )



def setup_training(config, model):
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], betas=tuple(config['adam_betas']), weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['gamma'])
    criterion = nn.MSELoss()
    return optimizer, criterion, scheduler

def setup_accelerator(config):
    accelerator = Accelerator(log_with="wandb")
    run_name = str(random.randint(0, 10e5))
    accelerator.init_trackers(
        config['project_name'],
        config=config,
        init_kwargs={"wandb": {"name": run_name}}
    )
    return accelerator, run_name

def evaluate_model(model, data_loader, accelerator):
    model.eval()
    total_time_loss = 0.0
    total_spectrogram_loss = 0.0
    num_samples = 0

    # To collect all latent representations
    all_latent_representations = []
    all_labels = []

    # Get the underlying model if it's wrapped in DistributedDataParallel
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(data_loader):
            x = x.to(accelerator.device)
            y = model.encode(x)
            y_decoded = model.decode(y)
            # labels = labels.to(accelerator.device)  # Assuming labels are provided

            # Time-domain reconstruction loss
            time_loss = F.mse_loss(y_decoded, x)
            total_time_loss += time_loss.item() * x.size(0)

            # Spectrogram-based reconstruction loss
            spectrogram_loss = compute_spectrogram_loss(x, y_decoded)
            total_spectrogram_loss += spectrogram_loss.item() * x.size(0)
            all_labels.extend(labels)
            num_samples += x.size(0)
            # Collect latent representations for visualization
            all_latent_representations.append(y.cpu())
            # Visualize the first 5 pairs in the batch
            if batch_idx < 3:
                for i in range(min(len(x), 1)):
                    plot_waveform_and_spectrogram(x[i], y_decoded[i], i)

    avg_time_loss = total_time_loss / num_samples
    avg_spectrogram_loss = total_spectrogram_loss / num_samples
    # Concatenate all latent representations into a single tensor
    all_latent_representations = torch.cat(all_latent_representations, dim=0)
    plot_latent_space(all_latent_representations, all_labels)
    # Log the losses to wandb
    accelerator.log({
        "eval_loss": avg_time_loss,
        "eval_spectrogram_loss": avg_spectrogram_loss
    })

    return avg_time_loss, avg_spectrogram_loss


def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, accelerator, config):
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    num_training_steps = config['epochs'] * len(train_loader)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    step = 1
    best_val_loss = float('inf')
    patience = 0
    early_stopping_patience = config['early_stopping_patience']

    for epoch in range(config['epochs']):
        for x, _ in train_loader:
            y = model(x)
            loss = criterion(y, x)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)

            accelerator.log({"training_loss": loss, "learning_rate": scheduler.get_last_lr()[0]}, step=step)
            step += 1
        eval_loss, freq_loss = evaluate_model(model, val_loader, accelerator)
        if epoch % config['save_every'] == 0 and accelerator.is_main_process:
            save_checkpoint(model, optimizer, epoch, config['model_save_dir'], f'model_epoch_{epoch}.pth')

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            patience = 0
            if accelerator.is_main_process:
                save_checkpoint(model, optimizer, epoch, config['model_save_dir'], f'model_epoch_{epoch}.pth')
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        if epoch % config['save_every'] == 0 and accelerator.is_main_process:
            save_checkpoint(model, optimizer, epoch, config['model_save_dir'], f'model_epoch_{epoch}.pth')

    accelerator.end_training()


def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(model, checkpoint_path)


def main():
    # Load the sweep configuration
    sweep_config = json.load(open('autoencoder_sweep.json'))

    # Load the fixed parameters from the separate JSON file
    with open('autoencoder_sweep_fixed.json', 'r') as f:
        fixed_params = json.load(f)

    # Initialize the wandb sweep
    print(f"Initializing wandb with project name: {fixed_params['project_name']}")
    # Initialize the wandb sweep
    wandb.init(config={**sweep_config["parameters"], **fixed_params})
    config = wandb.config

    # Construct model_save_dir
    config['model_save_dir'] = os.path.join(config['base_save_dir'], config['project_name'])
    os.makedirs(config['model_save_dir'], exist_ok=True)
    # Choose tokenizer based on the text encoder type
    accelerator, run_name = setup_accelerator(config)

    model = setup_model(config)
    optimizer, criterion, scheduler = setup_training(config, model)
    train_loader, val_loader = setup_dataloader(config)

    print(f"Training on {accelerator.num_processes} GPUs")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Models will be saved in: {config['model_save_dir']}")

    train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, accelerator, config)

    if accelerator.is_main_process:
        final_checkpoint_path = os.path.join(config['model_save_dir'], f'model_{run_name}.pth')
        save_checkpoint(accelerator.unwrap_model(model), optimizer, config['epochs'], config['model_save_dir'],
                        final_checkpoint_path)

    print("Training complete and models saved.")



if __name__ == "__main__":
    main()
