import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils.load_datasets

def load_config(config_path): 
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_dataloader(config):
    dataset_class = getattr(utils.load_datasets, config['dataset_name'])
    train_dataset = dataset_class(config['dataset_path'])
    return DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                      num_workers=config['num_workers'])

def setup_autoencoder(config):
    autoencoder = AutoEncoder1d(
        in_channels=config['ae_in_channels'],
        channels=config['ae_channels'],
        multipliers=config['ae_multipliers'],
        factors=config['ae_factors'],
        num_blocks=config['ae_num_blocks'],
        patch_size=config['ae_patch_size'],
        resnet_groups=config['ae_resnet_groups'],
        bottleneck=TanhBottleneck()
    )

    if 'ae_name' in config:
        checkpoint = torch.load(config['ae_name'], map_location=torch.device('cpu'))
        autoencoder.load_state_dict(checkpoint['model_state_dict'])

    return autoencoder

def setup_diffusion_model(config):
    return DiffusionModel(
        net_t=UNetV0,
        in_channels=config["dm_in_channels"],
        channels=config["dm_channels"],
        factors=config["dm_factors"],
        items=config["dm_items"],
        attentions=config["dm_attentions"],
        attention_heads=config["dm_attention_heads"],
        attention_features=config["dm_attention_features"],
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        use_text_conditioning=config["use_text_conditioning"],
        use_embedding_cfg=config["use_embedding_cfg"],
        embedding_max_length=config["embedding_max_length"],
        embedding_features=config["embedding_features"],
    )

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model

def generate_samples(model, autoencoder, prompts, device,config, num_steps=1000):
    model.eval()
    autoencoder.eval()

    with torch.no_grad():
        # Generate initial noisy samples (e.g., Gaussian noise)
        batch_size = len(prompts)
        latent_shape = (batch_size, 32, 256)  # Define your latent shape
        x_noisy = torch.randn(latent_shape, device=device)

        # Run the diffusion model to denoise the latent samples
        latent_samples = model.sample(x_noisy, text=prompts, embedding_scale=5.0, num_steps=1000)

        # Decode latent samples to IQ samples
        iq_samples = autoencoder.decode(latent_samples)

    return iq_samples

def calculate_mse(real_samples, generated_samples):
    return np.mean((real_samples - generated_samples) ** 2)

def plot_comparison(real_sample, generated_sample, index, output_dir):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(real_sample[0])
    plt.title('Real I Channel')
    
    plt.subplot(2, 2, 2)
    plt.plot(generated_sample[0])
    plt.title('Generated I Channel')
    
    plt.subplot(2, 2, 3)
    plt.plot(real_sample[1])
    plt.title('Real Q Channel')
    
    plt.subplot(2, 2, 4)
    plt.plot(generated_sample[1])
    plt.title('Generated Q Channel')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{index}.png'))
    plt.close()

def main():
    # Hardcoded values
    config_path = 'diffusion_sample.json'
    checkpoint_path = "/ext/trey/experiment_diffusion/experiment_rfdiffusion/models/latent_diffusion/model_epoch_100.pth"
    output_dir = 'comparison_results'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = load_config(config_path)

    autoencoder = setup_autoencoder(config).to(device)
    model = setup_diffusion_model(config).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    
    data_loader = setup_dataloader(config)

    os.makedirs(output_dir, exist_ok=True)
    
    mse_scores = []

    print("Generating samples and comparing with real data...")
    for i, (real_samples, prompts) in enumerate(tqdm(data_loader)):
        real_samples = real_samples.to(device)
        generated_samples = generate_samples(model, autoencoder, prompts, device, config)
        
        for j in range(real_samples.shape[0]):
            real_sample = real_samples[j].cpu().numpy()
            generated_sample = generated_samples[j].cpu().numpy()
            
            mse = calculate_mse(real_sample, generated_sample)
            print(mse)
            mse_scores.append(mse)
            
            plot_comparison(real_sample, generated_sample, i * config['batch_size'] + j, output_dir)

    print(f"Average MSE: {np.mean(mse_scores)}")
    print(f"Comparison plots saved in {output_dir}")

if __name__ == "__main__":
    main()
