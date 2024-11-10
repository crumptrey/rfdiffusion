import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Subset, Sampler
from sklearn.metrics import accuracy_score
import numpy as np
from collections import defaultdict
import random
import utils.load_datasets
import math

# Initialize Weights and Biases (wandb)
wandb.init(project="transformer-classifier")

# Device configuration (GPU/CPU)
device = torch.device("cuda:1")

# Define Hyperparameters
config = {
    "num_layers": 6,
    "signal_length": 1024,
    "num_classes": 24,
    "input_channels": 2,
    "embed_size": 512,
    "num_heads": 8,
    "expansion": 4,
    "learning_rate": 0.0001,
    "batch_size": 64,
    "num_epochs": 20,
    "dataset_path": "/ext/trey/experiment_diffusion/experiment_rfdiffusion/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
}

# Update wandb with config
wandb.config.update(config)

# Define Balanced Sampler
class BalancedModSNRRandomSampler(Sampler):
    def __init__(self, dataset, batch_size, snr_bin_size=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_indices = defaultdict(list)
        
        # Precompute class indices for combinations of modulation type and SNR bin
        for idx in range(len(dataset)):
            _, mod_type, snr = dataset[idx]
            snr_bin = round(snr / snr_bin_size) * snr_bin_size  # Bin SNR values
            class_key = (mod_type, snr_bin)  # Create a unique key combining mod_type and snr_bin
            self.class_indices[class_key].append(idx)
        
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        
        # Precompute number of batches based on the least common combination
        self.num_batches = min(len(indices) for indices in self.class_indices.values()) // batch_size
        self.total_samples = self.num_batches * self.num_classes * batch_size

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
            
            # Shuffle the batch to mix different class samples
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.total_samples

# Modify the function to include a test split
def setup_balanced_dataloaders(file_dir, batch_size, snr_bin_size=2, val_split=0.2, test_split=0.1, snr_min=30):
    # Load the dataset
    dataset = utils.load_datasets.DeepSig2018Dataset_MOD(file_dir)

    # Compute dataset sizes for splits
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * (dataset_size - test_size))
    train_size = dataset_size - val_size - test_size

    # Split indices
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create balanced samplers for train and val sets (balancing by mod_type and snr_bin)
    train_sampler = BalancedModSNRRandomSampler(train_dataset, batch_size, snr_bin_size=snr_bin_size)
    val_sampler = BalancedModSNRRandomSampler(val_dataset, batch_size, snr_bin_size=snr_bin_size)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

# Define Linear Embedding layer
class LinearEmbedding(nn.Module):
    def __init__(self, input_channels, embed_size):
        super(LinearEmbedding, self).__init__()
        self.linear = nn.Linear(input_channels, embed_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.linear(x)

# Define the Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, expansion):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, expansion * embed_size),
            nn.ReLU(),
            nn.Linear(expansion * embed_size, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attention_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attention_output))
        forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward_output))
        return x

# Define the Classifier
class Classifier(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # Global Average Pooling (pool along the time dimension)
        x = torch.mean(x, dim=1)
        return self.fc(x)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=1024):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Define the Transformer Model
class Transformer(nn.Module):
    def __init__(self, num_layers, signal_length, num_classes, input_channels, embed_size, num_heads, expansion):
        super().__init__()
        self.encoder = nn.ModuleList([TransformerEncoderLayer(
            embed_size=embed_size, num_heads=num_heads, expansion=expansion) for _ in range(num_layers)])
        self.classifier = Classifier(embed_size, num_classes)
        self.positional_encoding = SinusoidalPositionalEncoding(embed_size,signal_length)
        self.embedding = LinearEmbedding(input_channels, embed_size)

    def forward(self, x):

        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)

        for layer in self.encoder:
            embedded = layer(embedded)
        return self.classifier(embedded)

# Initialize the model, criterion, and optimizer
model = Transformer(
    config['num_layers'], 
    config['signal_length'], 
    config['num_classes'], 
    config['input_channels'], 
    config['embed_size'], 
    config['num_heads'], 
    config['expansion']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training and Validation Loop
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, total_correct = 0, 0
    for batch_idx, (data, targets, snr) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        print(snr)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()

        # Log to wandb
        if batch_idx % 10 == 0:
            wandb.log({
                "Train Loss": loss.item(),
                "Train Accuracy": total_correct / ((batch_idx + 1) * config['batch_size'])
            })

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    return avg_loss, avg_acc

def validate(model, loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for data, targets, snr in loader:
            data, targets = data.to(device), targets.to(device)
            print(snr)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    
    # Log to wandb
    wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": avg_acc})
    
    return avg_loss, avg_acc

# Test Evaluation Loop
def test_model(model, loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for data, targets, snr in loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)

    # Log to wandb
    wandb.log({"Test Loss": avg_loss, "Test Accuracy": avg_acc})

    return avg_loss, avg_acc

# Initialize data loaders
train_loader, val_loader, test_loader = setup_balanced_dataloaders(
    file_dir=config['dataset_path'], 
    batch_size=config['batch_size'], 
    snr_bin_size=2,  # or any other bin size you'd prefer
    val_split=0.2, 
    test_split=0.1,
    snr_min=30
)# Main training loop (unchanged)
# Main training loop
for epoch in range(config["num_epochs"]):
    print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # Log epoch results to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_acc
    })

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
# Evaluate on the test set after training is complete
test_loss, test_acc = test_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Finish the run
wandb.finish()
