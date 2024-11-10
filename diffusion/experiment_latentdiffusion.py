import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck, NoiserBottleneck
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import utils.load_datasets
import os
from tqdm import tqdm
from accelerate import Accelerator
import random
import json
from accelerate import DistributedDataParallelKwargs



def parse_args():
    parser = argparse.ArgumentParser(description="Train an latent diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_autoencoder(config):
    autoencoder = AutoEncoder1d(
        in_channels=config['ae_in_channels'],
        channels=config['ae_channels'],
        multipliers=config['ae_multipliers'],
        factors=config['ae_factors'],
        num_blocks=config['ae_num_blocks'],
        patch_size=config['ae_patch_size'],
        resnet_groups=config['ae_resnet_groups'],
        bottleneck=TanhBottleneck()  # You might want to make this configurable too
    )

    # Load pretrained weights if available
    if 'ae_name' in config:
        checkpoint = torch.load(config['ae_name'], map_location=torch.device('cpu'))
        autoencoder.load_state_dict(checkpoint['model_state_dict'])

    return autoencoder

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

def setup_diffusion_model(config):
    return DiffusionModel(
        net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
        in_channels= config["dm_in_channels"],  # U-Net: number of input/output (audio) channels
        channels= config["dm_channels"],  # U-Net: channels at each layer
        factors= config["dm_factors"],  # U-Net: downsampling and upsampling factors at each layer
        items= config["dm_items"],  # U-Net: number of repeating items at each layer
        attentions=config["dm_attentions"],  # U-Net: attention enabled/disabled at each layer
        attention_heads=config["dm_attention_heads"],  # U-Net: number of attention heads per attention item
        attention_features=config["dm_attention_features"],  # U-Net: number of attention features per attention item
        diffusion_t=VDiffusion,  # The diffusion method used
        sampler_t=VSampler,  # The diffusion sampler used
        use_text_conditioning=config["use_text_conditioning"],  # U-Net: enables text conditioning (default T5-base)
        use_embedding_cfg=config["use_embedding_cfg"],  # U-Net: enables classifier free guidance
        embedding_max_length=config["embedding_max_length"],  # U-Net: text embedding maximum length (default for T5-base)
        embedding_features=config["embedding_features"],  # U-Net: text embedding features (default for T5-base)
    )

def setup_training(config, model):
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], betas=tuple(config['adam_betas']))
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['gamma'])
    return optimizer #scheduler


def setup_dataloader(config):
    dataset_class = getattr(utils.load_datasets, config['dataset_name'])
    train_dataset = dataset_class(config['dataset_path'])
    return DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                      num_workers=config['num_workers'])


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


def train_model(model, ae, optimizer, data_loader, accelerator, config):
    model, ae, optimizer, data_loader = accelerator.prepare(
        model, ae, optimizer, data_loader
    )

    num_training_steps = config['epochs'] * len(data_loader)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    ae.eval()

    step = 1

    for epoch in range(config['epochs']):
        for x, prompt in data_loader:
            # Generate latent representation
            with torch.no_grad():
                # Use .module to access the underlying model if it's wrapped in DistributedDataParallel
                if isinstance(ae, torch.nn.parallel.DistributedDataParallel):
                    z = ae.module.encode(x)
                else:
                    z = ae.encode(x)
            optimizer.zero_grad()
            # Train diffusion model on latent space
            loss = model(z, text=prompt, embedding_mask_proba=0.1)

            accelerator.backward(loss)

            optimizer.step()
            #scheduler.step()

            progress_bar.update(1)
            accelerator.log({"training_loss": loss}, step=step)
            step += 1

        if epoch % config['save_every'] == 0 and accelerator.is_main_process:
            save_checkpoint(model, optimizer, epoch, config['model_save_dir'], f'model_epoch_{epoch}.pth')

    accelerator.end_training()


def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)


def main():
    config_path = 'config_diffusion.json'  # Specify your JSON file path here
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Construct model_save_dir
    config['model_save_dir'] = os.path.join(config['base_save_dir'], config['project_name'])
    os.makedirs(config['model_save_dir'], exist_ok=True)

    accelerator, run_name = setup_accelerator(config)
    embedding_model = torch.load(checkpoint_path, map_location='cpu')
    embedder = embedding_model.text_encoder()
    ae = setup_autoencoder(config)
    model = setup_diffusion_model(config)
    optimizer = setup_training(config, model)
    data_loader = setup_dataloader(config)



    print(f"Training on {accelerator.num_processes} GPUs")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Models will be saved in: {config['model_save_dir']}")

    train_model(model, ae, optimizer, data_loader, accelerator, config)

    if accelerator.is_main_process:
        final_checkpoint_path = os.path.join(config['model_save_dir'], f'model_{run_name}.pth')
        save_checkpoint(accelerator.unwrap_model(model), optimizer, config['epochs'], config['model_save_dir'],
                        final_checkpoint_path)

    print("Training complete and models saved.")


if __name__ == "__main__":
    main()
