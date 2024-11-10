import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel, BertModel
from torch.optim import AdamW
import numpy as np
import os
import json
import torch
from transformers import RobertaTokenizer, BertTokenizer
from accelerate import Accelerator
from audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck
import utils.load_datasets
from torch.utils.data import Sampler
from collections import defaultdict
import random
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
# Define the retrieval system
# Define the retrieval system
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

class SignalToTextRetrieval:
    def __init__(self, model_checkpoint, config, tokenizer_checkpoint='bert-base-uncased'):
        """
        Initialize the retrieval system with model and tokenizer.

        Args:
        - model_checkpoint: Path to the saved model checkpoint.
        - tokenizer_checkpoint: Name or path of the pre-trained RoBERTa tokenizer.
        - config: The configuration dictionary for the model architecture.
        """
        self.config = config
        self.model = self.load_model(model_checkpoint)  # Load the pretrained model
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_checkpoint)  # Load tokenizer

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Initialize a new instance of the model
    def load_model(self, checkpoint_path):
        """
        Load the full model from a checkpoint.

        Args:
        - checkpoint_path: Path to the model checkpoint.

        Returns:
        - model: The loaded PyTorch model.
        """
        # Load the entire model (not just the state_dict)
        model = torch.load(checkpoint_path, map_location='cpu')

        # Ensure it is the correct type
        if not isinstance(model, ContrastiveModel):
            raise TypeError(f"Expected a ContrastiveModel, got {type(model)}")

        return model
    def encode_signal(self, signal):
        """
        Encode the input signal using the signal encoder.

        Args:
        - signal: The 1D input signal to encode (tensor).

        Returns:
        - signal_embedding: Encoded signal embedding (tensor).
        """
        self.model.signal_encoder.eval()
        with torch.no_grad():
            signal_embedding = self.model.signal_encoder.encode(signal)
            signal_embedding = signal_embedding.view(signal_embedding.size(0), -1)  # Flatten the embedding
            signal_embedding = self.model.signal_projection_head(signal_embedding)  # Project to final embedding

        return signal_embedding

    def encode_text(self, text_prompts):
        self.model.text_encoder.eval()
        
        # Process each prompt individually
        individual_embeddings = []
        for prompt in text_prompts[:30]:  # Process first 10 prompts for demonstration
            #print(f"\nProcessing prompt: {prompt}")
            
            text_inputs = self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
            #print("Tokenized input:", text_inputs)
            
            with torch.no_grad():
                text_outputs = self.model.text_encoder(**text_inputs)
                #print("RoBERTa output sample:", text_outputs.last_hidden_state[0, 0, :10])
                embedding = self.model.text_projection_head(text_outputs.last_hidden_state[:, 0, :])
                individual_embeddings.append(embedding)
                    # Check if different prompts give different embeddings
        
        # Now process all prompts in batch
        all_text_inputs = self.tokenizer(text_prompts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            all_text_outputs = self.model.text_encoder(**all_text_inputs)
            all_embeddings = self.model.text_projection_head(all_text_outputs.last_hidden_state[:, 0, :])
        
        #print("\nAll embeddings shape:", all_embeddings.shape)
        #print("All embeddings sample (first 5):")
        #print(all_embeddings[:5, :10])  # Print first 10 elements of first 5 embeddings
        
        return all_embeddings

    def retrieve_best_match(self, signal_embedding, text_embeddings):
        """
        Retrieve the closest matching text for the given signal embedding using cosine similarity.

        Args:
        - signal_embedding: The embedding for the input signal (tensor).
        - text_embeddings: Encoded embeddings for all possible text prompts (tensor).

        Returns:
        - best_match_idx: Index of the closest matching text prompt.
        """
        # Compute cosine similarity between signal embedding and text embeddings
        similarity_matrix = F.cosine_similarity(signal_embedding.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)
        # Retrieve the index of the most similar text prompt
        best_match_indices = torch.argmax(similarity_matrix, dim=-1)  # Now it's [128]
        return best_match_indices

    def perform_retrieval(self, signal, text_prompts, k=5):
        """
        Perform signal-to-text retrieval by encoding the signal and text prompts and retrieving the closest matches.

        Args:
        - signal: Input signal (tensor).
        - text_prompts: List of text prompts (list of strings).
        - k: The number of top matches to retrieve.

        Returns:
        - top_k_matches: List of the top-k closest matching text prompts.
        """
        # Encode the signal and the text prompts
        signal_embedding = self.encode_signal(signal)
        #print("Signal Embedding:", signal_embedding[:5])  # Print a few samples
        text_embeddings = self.encode_text(text_prompts)
        #print("Text Embeddings:", text_embeddings[:5])

        # Retrieve the top-k matching text prompts with their similarity percentages
        top_k_indices, top_k_similarities = self.retrieve_top_k_matches(signal_embedding, text_embeddings, k=k)

        # Get the top-k matching text prompts
        top_k_matches = []
        for i in range(signal_embedding.size(0)):  # Iterate over the batch of signals
            matches = [(text_prompts[idx], sim.item()) for idx, sim in zip(top_k_indices[i], top_k_similarities[i])]
            top_k_matches.append(matches)

        return top_k_matches

    def retrieve_top_k_matches(self, signal_embedding, text_embeddings, k=5):
        """
        Retrieve the top-k closest matching text prompts for the given signal embedding using cosine similarity.

        Args:
    - signal_embedding: The embedding for the input signal (tensor).
    - text_embeddings: Encoded embeddings for all possible text prompts (tensor).
    - k: The number of top matches to return.

    Returns:
    - top_k_indices: Indices of the top-k closest matching text prompts.
    - top_k_similarities: Similarities for the top-k matches (percentage).
    """
    # Compute cosine similarity between signal embedding and text embeddings
        signal_embedding = F.normalize(signal_embedding, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        similarity_matrix = F.cosine_similarity(signal_embedding.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)
        
        # Get the top-k most similar text prompt indices and their similarities
        top_k_similarities, top_k_indices = torch.topk(similarity_matrix, k, dim=-1)
        
        # Convert similarities to percentages (0 to 100)
        top_k_similarities = top_k_similarities * 100
        
        return top_k_indices, top_k_similarities

# Define the MLP used for projection heads
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

# Define the main contrastive model
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
        self.signal_projection_head = MLP(64 * 256, config['mlp_hidden_dim'], config['mlp_projection_dim'])

    def forward(self, signal, text):
        signal_embedding = self.signal_encoder.encode(signal)
        signal_embedding = signal_embedding.view(signal_embedding.size(0), -1)  # Flatten signal embedding
        signal_embedding = self.signal_projection_head(signal_embedding)  # Project signal embedding

        text_outputs = self.text_encoder(**text)
        text_embedding = self.text_projection_head(text_outputs.last_hidden_state[:, 0, :])  # Project text embedding
        
        return signal_embedding, text_embedding

# Function to setup the accelerator for distributed or multi-GPU training
def setup_accelerator(config):
    accelerator = Accelerator()
    run_name = config.get('run_name', 'default_run')  # You can customize run_name from config
    return accelerator, run_name


def main():
    # Load configuration from JSON file
    config_path = 'config_contrastive_test.json'  # Path to the JSON configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Assuming `config` is the dictionary you use to configure the ContrastiveModel
    model_checkpoint = '/ext/trey/experiment_diffusion/experiment_rfdiffusion/models/contrastive_learning/model_885440.pth'  # Path to the saved model
    retriever = SignalToTextRetrieval(model_checkpoint, config)

    #train_loader, val_loader = setup_balanced_dataloaders(config)
    data_dir = '/ext/trey/experiment_diffusion/experiment_rfdiffusion/dataset/data/all_signals.h5'  # Adjust the path accordingly

    signal_dataset = utils.load_datasets.SignalDataset(data_dir, transform=utils.load_datasets.PowerNormalization())
    train_loader = torch.utils.data.DataLoader(
        signal_dataset, 
        batch_size=36, 
        num_workers=config.get("num_workers", 4),
    )
# Create the dataset
    # Available modulation types for the dataset
    TRAIN_MOD_TYPES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

    # Range of SNR values (you can customize the range based on your dataset)
    SNR_VALUES = np.arange(0, 31, 2)  # From -20 dB to 30 dB with a step of 5 dB

    # Generate all possible combinations of prompts
    prompts = ['{0} modulated signal at {1} dB SNR'.format(mod, snr) for mod in TRAIN_MOD_TYPES for snr in SNR_VALUES]


    # The variable `prompts` contains all the prompt combinations
    for signals, labels in train_loader:
        print(labels)
        # Example input signal (dummy tensor)
        top_k_matches = retriever.perform_retrieval(signals, prompts, k=5)

        # Print top-5 matches for the first few signals in the batch
        for i in range(min(10, len(signals))):  # Print up to 5 signals
            print(f"Signal index: {i}, True Label: {labels[i]}")
            for rank, (match, similarity) in enumerate(top_k_matches[i], start=1):
                print(f"  Rank {rank}: {match} (Similarity: {similarity:.2f}%)")


if __name__ == "__main__":
    main()
