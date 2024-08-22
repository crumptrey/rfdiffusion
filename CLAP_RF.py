import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, random_split
from audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck
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
import torch.nn.functional as F

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
    dataset = utils.load_datasets.DeepSig2018Dataset(config["dataset_path"])
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True,
                              num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=config["num_workers"])

    return train_loader, val_loader

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.signal_encoder = AutoEncoder1d(
            in_channels=config['ae_in_channels'],
            channels=config['ae_channels'],
            multipliers=config['ae_multipliers'],
            factors=config['ae_factors'],
            num_blocks=config['ae_num_blocks'],
            patch_size=config['ae_patch_size'],
            resnet_groups=config['ae_resnet_groups'],
            bottleneck=TanhBottleneck()
        )
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.text_projection_head = MLP(768, config['mlp_hidden_dim'], config['mlp_projection_dim'])
        self.signal_projection_head = MLP(32 * 256, config['mlp_hidden_dim'], config['mlp_projection_dim'])
        #self.text_projection_head = nn.Linear(768, config['mlp_projection_dim'])
        #self.signal_projection_head = nn.Linear(32 * 256,  config['mlp_projection_dim'])
    def forward(self, signal, text):
        signal_embedding = self.signal_encoder.encode(signal)
        # Reshape and project signal embedding
        signal_embedding = signal_embedding.view(signal_embedding.size(0), -1)  # Shape: [512, 8192]
        signal_embedding = self.signal_projection_head(signal_embedding)  # Shape: [512, 768]

        text_outputs = self.text_encoder(**text)
        text_embedding = self.text_projection_head(text_outputs.last_hidden_state[:, 0, :])
        return signal_embedding, text_embedding


class ContrastiveLossWithLearnableTemp(nn.Module):
    def __init__(self, initial_temperature=0.07):
        super(ContrastiveLossWithLearnableTemp, self).__init__()
        # Initialize the temperature parameter
        self.temperature = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32))

    def forward(self, embedding_x, embedding_y):
        # Normalize embeddings
        embedding_x = F.normalize(embedding_x, dim=1)
        embedding_y = F.normalize(embedding_y, dim=1)

        # Compute similarity logits
        similarity_x_y = torch.matmul(embedding_x, embedding_y.t()) / self.temperature
        similarity_y_x = torch.matmul(embedding_y, embedding_x.t()) / self.temperature

        # Create labels
        labels = torch.arange(similarity_x_y.size(0), device=similarity_x_y.device)

        # Compute loss components
        loss_x_y = F.cross_entropy(similarity_x_y, labels, reduction='mean')
        loss_y_x = F.cross_entropy(similarity_y_x, labels, reduction='mean')

        # Compute symmetric loss
        loss = (loss_x_y + loss_y_x) / 2

        return loss

def setup_model(config):
    return ContrastiveModel(config)


def setup_training(config, model):
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], betas=tuple(config['adam_betas']), weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['gamma'])
    criterion = ContrastiveLossWithLearnableTemp(initial_temperature=0.07)
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

def cosine_similarity(signal_embed, text_embed, batch_size=32):
    similarity_matrix = []

    for i in range(0, signal_embed.size(0), batch_size):
        signal_batch = signal_embed[i:i+batch_size]
        text_batch = text_embed[i:i+batch_size]
        if signal_batch.size(0) < batch_size:
            continue
        similarity_matrix.append(F.cosine_similarity(signal_batch.unsqueeze(1), text_batch.unsqueeze(0), dim=-1))

    return torch.cat(similarity_matrix, dim=0)


def mean_reciprocal_rank(similarity_matrix):
    # Get the rank of the correct match (i.e., the diagonal elements)
    ranks = torch.argsort(torch.argsort(similarity_matrix, dim=1, descending=True), dim=1, descending=False)
    correct_ranks = torch.diag(ranks)

    # Compute the reciprocal of the ranks and take the mean
    reciprocal_ranks = 1.0 / (correct_ranks.float() + 1.0)
    return reciprocal_ranks.mean().item()

def top_k_accuracy(similarity_matrix, k=1):
    # Get the indices of the top-k matches
    top_k_matches = torch.topk(similarity_matrix, k=k, dim=1).indices
    # Check if the correct match (i.e., diagonal index) is within the top-k matches
    correct_matches = torch.arange(similarity_matrix.size(0)).unsqueeze(1).to(similarity_matrix.device)
    correct_in_top_k = (top_k_matches == correct_matches).any(dim=1)
    return correct_in_top_k.float().mean().item()

def evaluate_model(model, criterion, data_loader, accelerator, tokenizer):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    signal_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for batch_idx, (signals, prompts) in enumerate(data_loader):
            signals = signals.to(accelerator.device)
            text_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(accelerator.device)

            signal_embed, text_embed = model(signals, text_inputs)
            loss = criterion(signal_embed, text_embed)

            total_loss += loss.item() * signals.size(0)
            num_samples += signals.size(0)
            # Store embeddings for metrics calculation
            signal_embeddings.append(signal_embed)
            text_embeddings.append(text_embed)

    # Concatenate all embeddings
    signal_embeddings = torch.cat(signal_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(signal_embeddings, text_embeddings)

    # Calculate metrics
    avg_loss = total_loss / num_samples
    avg_cosine_similarity = torch.mean(torch.diag(similarity_matrix)).item()
    mrr = mean_reciprocal_rank(similarity_matrix)
    top_1_accuracy = top_k_accuracy(similarity_matrix, k=1)
    top_30_accuracy = top_k_accuracy(similarity_matrix, k=30)

    # Log metrics
    accelerator.log({
        "eval_loss": avg_loss,
        "avg_cosine_similarity": avg_cosine_similarity,
        "mean_reciprocal_rank": mrr,
        "top_1_accuracy": top_1_accuracy,
        "top_30_accuracy": top_30_accuracy,
    })

    return avg_loss


def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, accelerator, config, tokenizer):
    model, optimizer,criterion, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, criterion, train_loader, val_loader, scheduler
    )
    num_training_steps = config['epochs'] * len(train_loader)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    step = 1

    for epoch in range(config['epochs']):
        for signals, labels in train_loader:
            text_inputs = tokenizer(labels, padding=True, truncation=True, return_tensors="pt").to(accelerator.device)

            signal_embed, text_embed = model(signals, text_inputs)
            loss = criterion(signal_embed, text_embed)

            accelerator.backward(loss)
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)

            accelerator.log({"training_loss": loss, "learning_rate": scheduler.get_last_lr()[0]}, step=step)
            step += 1

        eval_loss = evaluate_model(model, criterion, val_loader, accelerator, tokenizer)
        if epoch % config['save_every'] == 0 and accelerator.is_main_process:
            save_checkpoint(model, optimizer, epoch, config['model_save_dir'], f'model_epoch_{epoch}.pth')

    accelerator.end_training()


def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(model, checkpoint_path)


def main():
    config_path = 'config_contrastive.json'  # Specify your JSON file path here
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Construct model_save_dir
    config['model_save_dir'] = os.path.join(config['base_save_dir'], config['project_name'])
    os.makedirs(config['model_save_dir'], exist_ok=True)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    accelerator, run_name = setup_accelerator(config)

    model = setup_model(config)
    optimizer, criterion, scheduler = setup_training(config, model)
    train_loader, val_loader = setup_dataloader(config)

    print(f"Training on {accelerator.num_processes} GPUs")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Models will be saved in: {config['model_save_dir']}")

    train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, accelerator, config, tokenizer)

    if accelerator.is_main_process:
        final_checkpoint_path = os.path.join(config['model_save_dir'], f'model_{run_name}.pth')
        save_checkpoint(accelerator.unwrap_model(model), optimizer, config['epochs'], config['model_save_dir'],
                        final_checkpoint_path)

    print("Training complete and models saved.")


if __name__ == "__main__":
    main()
