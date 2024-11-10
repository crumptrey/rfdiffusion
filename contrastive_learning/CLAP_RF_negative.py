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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from accelerate import DistributedDataParallelKwargs
from sklearn.random_projection import GaussianRandomProjection
from torch.utils.data import Sampler
# Function to set the fixed seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set your seed value here
SEED = 0
set_seed(SEED)
class BalancedSubsetRandomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_indices = defaultdict(list)
        
        # Precompute class indices
        for idx in range(len(dataset)):
            _, prompt = dataset[idx]
            class_key = self.get_class_key(prompt)
            self.class_indices[class_key].append(idx)
        
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        
        # Precompute number of batches
        self.num_batches = min(len(indices) for indices in self.class_indices.values()) // batch_size
        self.total_samples = self.num_batches * self.num_classes * batch_size

    def get_class_key(self, prompt):
        mod_type, snr = prompt.split(' modulated signal at ')
        snr = float(snr.split(' dB SNR')[0])
        snr_bin = round(snr / 2) * 2
        return (mod_type, snr_bin)

    def __iter__(self):
        class_samples = {
            class_key: np.random.permutation(indices).tolist()
            for class_key, indices in self.class_indices.items()
        }
        
        for _ in range(self.num_batches):
            batch = []
            for class_key in self.classes:
                batch.extend(class_samples[class_key][:self.batch_size])
                class_samples[class_key] = class_samples[class_key][self.batch_size:]
            
            # Shuffle the batch
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.total_samples

def setup_balanced_dataloaders(config, val_split=0.2):
    dataset = utils.load_datasets.DeepSig2018Dataset(config["dataset_path"])
    
    # Create train and validation splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create balanced samplers
    train_sampler = BalancedSubsetRandomSampler(train_dataset, config["batch_size"])
    val_sampler = BalancedSubsetRandomSampler(val_dataset, config["batch_size"])
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        sampler=train_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        sampler=val_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
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

class ContrastiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['ae_bottleneck'] == 0:
            bottleneck = [TanhBottleneck()]
        elif config['ae_bottleneck'] == 1:
            bottleneck = [TanhBottleneck(), NoiserBottleneck(config['ae_sigma'])]
        elif config['ae_bottleneck'] == 2:
            bottleneck = [NoiserBottleneck(config['ae_sigma']), TanhBottleneck()]
        self.signal_encoder = AutoEncoder1d(
            in_channels=config['ae_in_channels'],
            channels=config['ae_channels'],
            multipliers=config['ae_multipliers'],
            factors=config['ae_factors'],
            num_blocks=config['ae_num_blocks'],
            patch_size=config['ae_patch_size'],
            resnet_groups=config['ae_resnet_groups'],
            bottleneck=bottleneck)

        # Load pretrained weights if a path is provided
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection_head = nn.Linear(768, config['mlp_projection_dim'])
        self.signal_projection_head = nn.Linear(64 * 256,  config['mlp_projection_dim'])
        self.temperature = nn.Parameter(torch.ones([])*np.log(1/0.07))
    def forward(self, signal, text):
        signal_embedding = self.signal_encoder.encode(signal)
        # Reshape and project signal embedding
        signal_embedding = signal_embedding.view(signal_embedding.size(0), -1)  # Shape: [512, 8192]
        signal_embedding = self.signal_projection_head(signal_embedding)  # Shape: [512, 768]

        text_outputs = self.text_encoder(**text)
        text_embedding = self.text_projection_head(text_outputs.last_hidden_state[:, 0, :])
        return signal_embedding, text_embedding, self.temperature.exp()

class ContrastiveLossWithLearnableTemp(nn.Module):
    def __init__(self):
        super(ContrastiveLossWithLearnableTemp, self).__init__()
        # Initialize the temperature parameter

    def forward(self, embedding_x, embedding_y,temperature):
        # Normalize embeddings
        embedding_x = F.normalize(embedding_x, dim=1)
        embedding_y = F.normalize(embedding_y, dim=1)

        # Compute similarity logits and scale by temperature
        similarity_x_y = torch.matmul(embedding_x, embedding_y.t()) * temperature
        similarity_y_x = torch.matmul(embedding_y, embedding_x.t()) * temperature

        # Clip the logits to a maximum value of 100 to prevent instability
        similarity_x_y = torch.clamp(similarity_x_y, max=100)
        similarity_y_x = torch.clamp(similarity_y_x, max=100)

        # Create labels (for contrastive learning)
        labels = torch.arange(similarity_x_y.size(0), device=similarity_x_y.device)

        # Compute the cross-entropy loss
        loss_x_y = F.cross_entropy(similarity_x_y, labels)
        loss_y_x = F.cross_entropy(similarity_y_x, labels)

        # Symmetric loss (averaging the two directions)
        loss = (loss_x_y + loss_y_x) / 2

        return loss

def setup_model(config):
    return ContrastiveModel(config)


def setup_training(config, model):
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], betas=tuple(config['adam_betas']), weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    criterion = ContrastiveLossWithLearnableTemp()
    return optimizer, criterion, scheduler

def setup_accelerator(config):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
    run_name = str(random.randint(0, 10e5))
    accelerator.init_trackers(
        config['project_name'],
        config=config,
        init_kwargs={"wandb": {"name": run_name}}
    )
    return accelerator, run_name

def cosine_similarity(signal_embed, text_embed, batch_size=32):
    signal_embed = F.normalize(signal_embed, p=2, dim=-1)
    text_embed = F.normalize(text_embed, p=2, dim=-1)
    
    similarity_matrix = []
    for i in range(0, signal_embed.size(0), batch_size):
        signal_batch = signal_embed[i:i+batch_size]
        text_batch = text_embed[i:i+batch_size]
        if signal_batch.size(0) < batch_size:
            continue
        similarity_matrix.append(F.cosine_similarity(signal_batch.unsqueeze(1), text_batch.unsqueeze(0), dim=-1))

    return torch.cat(similarity_matrix, dim=0)

def average_precision(similarity_scores, k=None):
    """
    Calculate Average Precision for a single query.

    Args:
    similarity_scores (torch.Tensor): A tensor of similarity scores for a single query.
    k (int): The number of top results to consider. If None, consider all results.

    Returns:
    float: The Average Precision score.
    """
    if k is None:
        k = len(similarity_scores)

    # Sort similarity scores in descending order
    _, indices = torch.sort(similarity_scores, descending=True)

    # The correct match should be at index 0 (assuming diagonal elements are correct matches)
    positives = (indices[:k] == 0).float()

    # Calculate precision at each position
    precisions = torch.cumsum(positives, dim=0) / torch.arange(1, k + 1, device=positives.device)

    # Calculate AP
    ap = (precisions * positives).sum() / min(k, positives.sum().item())

    return ap.item()

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

def recall_at_k(similarity_matrix, k):
    """
    Calculate Recall@k for the given similarity matrix.

    Args:
    similarity_matrix (torch.Tensor): A 2D tensor of similarity scores.
    k (int): The rank at which to calculate recall.

    Returns:
    float: The Recall@k score.
    """
    num_queries = similarity_matrix.size(0)

    # Get the indices of the top k matches for each query
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

    # Create a tensor of correct indices (diagonal elements)
    correct_indices = torch.arange(num_queries, device=similarity_matrix.device).unsqueeze(1)

    # Check if the correct index is in the top k for each query
    correct_in_top_k = (top_k_indices == correct_indices).any(dim=1)

    # Calculate recall
    recall = correct_in_top_k.sum().float() / num_queries if num_queries > 0 else 0.0

    return recall

def precision_at_k(similarity_matrix, k):
    """
    Calculate Precision@k for the given similarity matrix.

    Args:
    similarity_matrix (torch.Tensor): A 2D tensor of similarity scores.
    k (int): The rank at which to calculate precision.

    Returns:
    float: The Precision@k score.
    """
    num_queries = similarity_matrix.size(0)

    # Get the indices of the top k matches for each query
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

    # Create a tensor of correct indices (diagonal elements)
    correct_indices = torch.arange(num_queries, device=similarity_matrix.device).unsqueeze(1)

    # Check if the correct index is in the top k for each query
    correct_in_top_k = (top_k_indices == correct_indices).any(dim=1)

    # Calculate precision
    precision = correct_in_top_k.sum().float() / (k * num_queries) if num_queries > 0 else 0.0

    return precision

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

            signal_embed, text_embed, temp = model(signals, text_inputs)
            loss = criterion(signal_embed, text_embed, temp)

            total_loss += loss.item() * signals.size(0)
            num_samples += signals.size(0)
            # Store embeddings for metrics calculation
            signal_embeddings.append(signal_embed)
            text_embeddings.append(text_embed)

    # Concatenate all embeddings
    signal_embeddings = torch.cat(signal_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    # Calculate similarity matrix
    #similarity_matrix = cosine_similarity(signal_embeddings, text_embeddings)
    print('computing similarity')
    similarity_matrix = cosine_similarity(signal_embeddings, text_embeddings)
    # Calculate metrics
    avg_loss = total_loss / num_samples
    avg_cosine_similarity = torch.mean(torch.diag(similarity_matrix)).item()
    r_at_5 = recall_at_k(similarity_matrix, 5)
    r_at_10 = recall_at_k(similarity_matrix, 10)
    p_at_5 = precision_at_k(similarity_matrix, 5)
    p_at_10 = precision_at_k(similarity_matrix, 10)
    # Calculate mAP
    ap_scores = [average_precision(similarity_matrix[i]) for i in range(similarity_matrix.shape[0])]
    map_score = sum(ap_scores) / len(ap_scores)

    mrr = mean_reciprocal_rank(similarity_matrix)

    accelerator.log({
        "eval_loss": avg_loss,
        "avg_cosine_similarity": avg_cosine_similarity,
        "mean_reciprocal_rank": mrr,
        "R@5": r_at_5,
        "R@10": r_at_10,
        "P@5": p_at_5,
        "P@10": p_at_10,
        "mAP": map_score
    })

    # Log metrics

    return avg_loss


def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, accelerator, config, tokenizer):
    model, optimizer,criterion, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, criterion, train_loader, val_loader, scheduler
    )
    num_training_steps = config['epochs'] * len(train_loader)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    model.text_encoder.train()
    step = 1
    best_eval_loss = float('inf')
    patience = 3  # Number of epochs to wait for improvement
    epochs_without_improvement = 0
    for epoch in range(config['epochs']):
        for signals, labels in train_loader:
            text_inputs = tokenizer(labels, padding=True, truncation=True, return_tensors="pt").to(accelerator.device)
            roberta_weights_before = model.text_encoder.state_dict()
            signal_embed, text_embed, temp = model(signals, text_inputs)
            loss = criterion(signal_embed, text_embed, temp)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
                        # Print RoBERTa weights after update
                        # Capture RoBERTa weights after update
            roberta_weights_after = model.text_encoder.state_dict()
            for name, param_before in roberta_weights_before.items():
                param_after = roberta_weights_after[name]
                if not torch.equal(param_before, param_after):
                    print(f"Weights updated for layer: {name}")

            progress_bar.update(1)

            accelerator.log({"training_loss": loss, "learning_rate": scheduler.get_last_lr()[0]}, step=step)
            step += 1

        eval_loss = evaluate_model(model, criterion, val_loader, accelerator, tokenizer)
        scheduler.step(eval_loss)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Evaluation Loss: {eval_loss:.4f}")
                # Check for early stopping
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            epochs_without_improvement = 0
            print(f"New best evaluation loss: {best_eval_loss:.4f}. Saving model...")
            # Save your model here (if applicable)
            if epoch % config['save_every'] == 0 and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_checkpoint(unwrapped_model, optimizer, epoch, config['model_save_dir'], f'model_epoch_{epoch}.pth')
                # Save the model (e.g., torch.save(unwrapped_model.state_dict(), 'model.pth'))
        else:
            epochs_without_improvement += 1
            print(f"No improvement in evaluation loss. Current patience: {patience - epochs_without_improvement}")

            if epochs_without_improvement >= patience:
                print("Early stopping triggered. Stopping training.")
                break  # Stop training

    accelerator.end_training()

def hard_negative_sampling(self, embeddings_x, embeddings_y, temperature, mode='within', top_k=5):
    """
    Perform hard negative sampling based on the specified mode.

    Args:
    embeddings_x: The embeddings from modality X (e.g., audio).
    embeddings_y: The embeddings from modality Y (e.g., text).
    mode: The type of hard negative sampling ('within', 'cross', 'semi-hard').
    top_k: The number of hard negatives to sample.

    Returns:
    selected_negatives: Indices of the selected hard negatives.
    """
    embedding_x = F.normalize(embedding_x, dim=1)
    embedding_y = F.normalize(embedding_y, dim=1)
    if mode == 'within':
        # Calculate within-modality scores
        within_scores_x = torch.matmul(embeddings_x, embeddings_x.t()) * temperature
        within_scores_y = torch.matmul(embeddings_y, embeddings_y.t()) * temperature

        # Select hard negatives based on maximum within-modality scores
        _, hard_negatives_x = torch.topk(within_scores_x, top_k, dim=1, largest=False)
        _, hard_negatives_y = torch.topk(within_scores_y, top_k, dim=1, largest=False)

        return hard_negatives_x, hard_negatives_y

    elif mode == 'cross':
        # Calculate cross-modality scores
        cross_scores = torch.matmul(embeddings_x, embeddings_y.t()) * temperature

        # Select hard negatives based on maximum cross-modality scores
        _, hard_negatives_x = torch.topk(cross_scores, top_k, dim=1, largest=False)
        return hard_negatives_x,torch.empty_like(hard_negatives_x)

    elif mode == 'semi-hard':
        # Calculate cross-modality scores
        cross_scores = torch.matmul(embeddings_x, embeddings_y.t()) * temperature

        # Assuming that the first element is the positive pair, find semi-hard negatives
        pos_scores = cross_scores.diag()
        semi_hard_negatives = torch.argsort(torch.abs(cross_scores - pos_scores.view(-1, 1)), dim=1)[:, 1:top_k + 1]
        return semi_hard_negatives,torch.empty_like(semi_hard_negatives)

    else:
        raise ValueError("Invalid mode selected for hard negative sampling.")

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    accelerator, run_name = setup_accelerator(config)

    model = setup_model(config)

    # Wrap the model in DistributedDataParallel with find_unused_parameters=True

# Training and evaluation logic remains the same

    optimizer, criterion, scheduler = setup_training(config, model)
    train_loader, val_loader = setup_balanced_dataloaders(config)


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
