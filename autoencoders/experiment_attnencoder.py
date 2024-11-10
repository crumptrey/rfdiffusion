#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, random_split
from audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck, NoiserBottleneck
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
import torch
import torch.nn as nn

# If you need the full transformer architecture
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

# Embedding layers and other helpers
from torch.nn import Embedding

# Optional: If you're defining custom positional encodings
import math

os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"
import wandb



def plot_latent_space(latent_representations, labels, idx=0, save_dir="plots_fixed", save_name="latent_space",
                          max_samples=10000):
        """
        Plot the latent space of the encoded signals using t-SNE, with different subplots for specific modulation groups.
        Limit the number of samples to speed up t-SNE computation.
        """
        latent_representations = latent_representations.cpu().detach().numpy()
        if latent_representations.ndim > 2:
            num_samples = latent_representations.shape[0]
            num_features = np.prod(latent_representations.shape[1:])
            latent_representations = latent_representations.reshape(num_samples, num_features)

        os.makedirs(save_dir, exist_ok=True)

        # Limit the number of samples
        if latent_representations.shape[0] > max_samples:
            indices = np.random.choice(latent_representations.shape[0], max_samples, replace=False)
            latent_representations = latent_representations[indices]
            labels = [labels[i] for i in indices]

        # Define modulation groups
        analog_mods = ['AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM']
        digital_mods = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                        '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'GMSK', 'OQPSK']
        am_mods = ['AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC']
        fm_mods = ['FM']
        ask_mods = ['OOK', '4ASK', '8ASK']
        psk_mods = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK']
        qam_mods = ['16QAM', '32QAM', '64QAM', '128QAM', '256QAM']

        # Perform t-SNE
        print(f"Performing t-SNE on {latent_representations.shape[0]} samples...")
        tsne_result = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(latent_representations)
        print("t-SNE complete.")

        # Create color map
        unique_labels = sorted(set(labels))
        color_map = plt.cm.get_cmap('tab20')
        color_dict = {label: color_map(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

        # Function to create scatter plot
        def scatter_with_legend(ax, x, y, plot_labels, title, mod_group):
            for label in mod_group:
                mask = np.array(plot_labels) == label
                if np.any(mask):  # Only plot if there are points for this label
                    ax.scatter(x[mask], y[mask], c=[color_dict[label]], label=label, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend(title="Modulations", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

        # Analog vs Digital
        scatter_with_legend(axes[0, 0], tsne_result[:, 0], tsne_result[:, 1], labels, 'Analog Modulations', analog_mods)
        scatter_with_legend(axes[0, 1], tsne_result[:, 0], tsne_result[:, 1], labels, 'Digital Modulations',
                            digital_mods)

        # AM vs FM
        scatter_with_legend(axes[1, 0], tsne_result[:, 0], tsne_result[:, 1], labels, 'AM Modulations', am_mods)
        scatter_with_legend(axes[1, 1], tsne_result[:, 0], tsne_result[:, 1], labels, 'FM Modulation', fm_mods)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_name}_analog_digital_{idx}.jpg"), format="jpg", bbox_inches='tight')
        plt.close()

        # ASK vs PSK vs QAM
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        scatter_with_legend(axes[0], tsne_result[:, 0], tsne_result[:, 1], labels, 'ASK Modulations', ask_mods)
        scatter_with_legend(axes[1], tsne_result[:, 0], tsne_result[:, 1], labels, 'PSK Modulations', psk_mods)
        scatter_with_legend(axes[2], tsne_result[:, 0], tsne_result[:, 1], labels, 'QAM Modulations', qam_mods)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_name}_ask_psk_qam_{idx}.jpg"), format="jpg", bbox_inches='tight')
        plt.close()

        def plot_grouped_scatter(grouped_labels, group_names, title, filename):
            fig, ax = plt.subplots(figsize=(15, 15))
            color_map = plt.cm.get_cmap('tab10')
            color_dict = {group: color_map(i) for i, group in enumerate(group_names) if group != 'Other'}

            for group in group_names:
                if group != 'Other':  # Exclude 'Other' category
                    mask = np.array(grouped_labels) == group
                    if np.any(mask):
                        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1], c=[color_dict[group]], label=group,
                                   alpha=0.7)

            ax.set_title(title)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend(title="Modulations", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{save_name}_{filename}_{idx}.jpg"), format="jpg", bbox_inches='tight')
            plt.close()

        # 1. Digital vs. Analog (grouped)

        digital_analog_labels = ['Digital' if label in digital_mods else 'Analog' for label in labels]
        plot_grouped_scatter(digital_analog_labels, ['Digital', 'Analog'], 'Digital vs Analog Modulations',
                             'digital_vs_analog')

        # 2. AM vs. FM (grouped)
        am_fm_labels = ['AM' if label in am_mods else 'FM' if label in fm_mods else 'Other' for label in labels]
        plot_grouped_scatter(am_fm_labels, ['AM', 'FM'], 'AM vs FM Modulations', 'am_vs_fm')

        # 3. ASK vs. PSK vs. QAM (grouped)
        ask_psk_qam_labels = ['ASK' if label in ask_mods else 'PSK' if label in psk_mods
        else 'QAM' if label in qam_mods else 'Other' for label in labels]
        plot_grouped_scatter(ask_psk_qam_labels, ['ASK', 'PSK', 'QAM'], 'ASK vs PSK vs QAM Modulations',
                             'ask_psk_qam')


def plot_waveform_and_spectrogram(input_signal, decoded_signal, idx, save_dir="plots_fixed",
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
# Define custom input embedding layer to handle multi-channel signal (I/Q data)
class IQEmbedding(nn.Module):
    def __init__(self, input_channels=2, d_model=512):
        super(IQEmbedding, self).__init__()
        # Project the two-channel (I/Q) data into the d_model dimension
        self.linear = nn.Linear(input_channels, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, input_channels)
        return self.linear(x)  # Output: (batch_size, seq_len, d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderForIQ(nn.Module):
    def __init__(self, input_channels=2, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerEncoderForIQ, self).__init__()
        
        # Embedding for I/Q data
        self.iq_embedding = IQEmbedding(input_channels, d_model)
        
        # Positional encoding to add temporal/positional information
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Bottleneck layer for latent space (compression)
        self.bottleneck = nn.Linear(d_model, d_model // 2)  # Compress to latent representation

    def forward(self, src):
        # src: (batch_size, seq_len, input_channels)
        embedded = self.iq_embedding(src)  # (batch_size, seq_len, d_model)
        encoded = self.positional_encoding(embedded)
        print(encoded.size())
        encoded = self.transformer_encoder(encoded)
        print(encoded.size())
        latent_representation = self.bottleneck(encoded)  # Compressed representation
        print(latent_representation.size())
        return latent_representation


class TransformerDecoderForIQ(nn.Module):
    def __init__(self, latent_dim=256, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerDecoderForIQ, self).__init__()
        
        # Expand latent space back to the original dimensionality
        self.expander = nn.Linear(latent_dim, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Final reconstruction layer to map back to the original 2-channel input
        self.reconstruction_layer = nn.Linear(d_model, 2)  # Reconstruct the two-channel I/Q data

    def forward(self, latent):
        expanded = self.expander(latent)  # Expand latent representation
        decoded = self.transformer_decoder(expanded)
        output = self.reconstruction_layer(decoded)  # Output: (batch_size, seq_len, input_channels)
        return output

class TransformerAutoencoderForIQ(nn.Module):
    def __init__(self, input_channels=2, d_model=512, latent_dim=256, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerAutoencoderForIQ, self).__init__()
        
        # Transformer Encoder
        self.encoder = TransformerEncoderForIQ(input_channels, d_model, nhead, num_layers, dropout)
        
        # Transformer Decoder
        self.decoder = TransformerDecoderForIQ(latent_dim, d_model, nhead, num_layers, dropout)

    def forward(self, src):
        # src: (batch_size, seq_len, input_channels)
        latent = self.encoder(src)  # Compressed representation
        reconstruction = self.decoder(latent)  # Reconstructed two-channel data
        return reconstruction

def setup_training(config, model):
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], betas=tuple(config['adam_betas']), weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()
    return optimizer, criterion

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
            x = x.permute(0, 2, 1)  # Change from (1024, 2, 1024) to (1024, 1024, 2)
            y = model(x)
            
            # Time-domain reconstruction loss
            time_loss = F.mse_loss(y, x)
            total_time_loss += time_loss.item() * x.size(0)

            # Collect latent representations for visualization
            all_latent_representations.append(y.cpu())
            all_labels.extend(labels)
            num_samples += x.size(0)
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


def train_model(model, optimizer, criterion, train_loader, val_loader, accelerator, config):
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    num_training_steps = config['epochs'] * len(train_loader)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    step = 1

    for epoch in range(config['epochs']):
        for x, _ in train_loader:
            # Transformer expects (sequence_length, batch_size, d_model)
            # In your data loading/preprocessing step
            x = x.permute(0, 2, 1)  # Change from (1024, 2, 1024) to (1024, 1024, 2)
            y = model(x)  # Forward pass through the Transformer

            y = y.permute(0, 2, 1)  # Change from (1024, 2, 1024) to (1024, 1024, 2)
            # Rearrange the output for loss computation
            loss = criterion(y, x)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            accelerator.log({"training_loss"}, step=step)
        if epoch == 25:
            eval_loss, freq_loss = evaluate_model(model, val_loader, accelerator)
        if epoch % config['save_every'] == 0 and accelerator.is_main_process:
            save_checkpoint(model, optimizer, epoch, config['model_save_dir'], f'model_epoch_{epoch}.pth')

    accelerator.end_training()


def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), checkpoint_path)


def main():
    # Load the configuration from a single JSON file
    with open('config_transformer.json', 'r') as f:
        config = json.load(f)

    # Initialize wandb
    print(f"Initializing wandb with project name: {config['project_name']}")
    wandb.init(project=config['project_name'], config=config)

    # Construct model_save_dir
    config['model_save_dir'] = os.path.join(config['base_save_dir'], config['project_name'])
    os.makedirs(config['model_save_dir'], exist_ok=True)

    accelerator, run_name = setup_accelerator(config)
    model = TransformerAutoencoderForIQ(input_channels=2, d_model=256, latent_dim=256)
    optimizer, criterion = setup_training(config, model)
    train_loader, val_loader = setup_dataloader(config)

    print(f"Training on {accelerator.num_processes} GPUs")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Models will be saved in: {config['model_save_dir']}")

    train_model(model, optimizer, criterion, train_loader, val_loader, accelerator, config)

    if accelerator.is_main_process:
        final_checkpoint_path = os.path.join(config['model_save_dir'], f'model_{run_name}.pth')
        save_checkpoint(accelerator.unwrap_model(model), optimizer, config['epochs'], config['model_save_dir'],
                        final_checkpoint_path)

    print("Training complete and models saved.")



if __name__ == "__main__":
    main()
