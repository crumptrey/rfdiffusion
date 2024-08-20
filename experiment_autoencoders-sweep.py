import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from audio_encoders_pytorch.audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck
import utils.load_datasets
import os
from tqdm import tqdm
from accelerate import Accelerator
import random
import wandb
import numpy as np
from einops import rearrange

def setup_dataloader(batch_size, num_workers, val_split=0.2):
    dataset = utils.load_datasets.DeepSig2018Dataset(
        "/ext/trey/experiment_diffusion/experiment_rfdiffusion/dataset/GOLD_XYZ_OSC.0001_1024.hdf5")
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader

def evaluate_model(model, data_loader, accelerator):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(accelerator.device)
            y = model.encode(x)
            y = model.decode(y)
            loss = torch.nn.functional.mse_loss(y, x)
            total_loss += loss.item() * x.size(0)
            num_samples += x.size(0)

    avg_loss = total_loss / num_samples
    return avg_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train an autoencoder model")
    return parser.parse_args()

def setup_model(ae_in_channels, ae_channels, ae_multipliers, ae_factors, ae_num_blocks, ae_patch_size,
                ae_resnet_groups, bottleneck):
    return AutoEncoder1d(
        in_channels=ae_in_channels,
        channels=ae_channels,
        multipliers=ae_multipliers,
        factors=ae_factors,
        num_blocks=ae_num_blocks,
        patch_size=ae_patch_size,
        resnet_groups=ae_resnet_groups,
        bottleneck=bottleneck
    )

def setup_training(learning_rate, adam_betas, model):
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=adam_betas)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999996)
    return optimizer, criterion, scheduler

def setup_accelerator(config):
    accelerator = Accelerator(log_with="wandb")
    run_name = str(random.randint(0, 10e5))
    accelerator.init_trackers(
        "autoencoder_a6000",
        config=config,
        init_kwargs={"wandb": {"name": run_name}}
    )
    return accelerator, run_name

def train_model(model, optimizer, criterion, scheduler, data_loader, accelerator,val_loader):
    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )
    num_training_steps = 10 * len(data_loader)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    step = 1

    for epoch in range(10):
        for x, _ in data_loader:
            y = model(x, with_info=False)
            loss = criterion(y, x)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            accelerator.log({"training_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]}, step=step)

            progress_bar.update(1)
            step += 1

        validation_loss = evaluate_model(model, val_loader, accelerator)
        wandb.log({"validation_loss": validation_loss})

        if epoch % 10 == 0 and accelerator.is_main_process:
            save_checkpoint(model, optimizer, epoch, "/home/trey/experiment_rfdiffusion/models/rfdiffusion_diffusion",
                            f'model_epoch_{epoch}.pth')
            wandb.log({"epoch": epoch, "training_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]}, step=step)

    accelerator.end_training()
    return validation_loss

def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

def objective():
    wandb.init()  # Ensure wandb.init() is called
    config = wandb.config

    ae_in_channels = 2
    ae_channels = config.ae_channels
    num_layers = config.num_layers

    ae_factors = [config[f'ae_factor_{i}'] for i in range(num_layers - 1)]

    ae_multipliers = [config[f'ae_multi_{i}'] for i in range(num_layers)]

    ae_num_blocks = [config[f'ae_num_block_{i}'] for i in range(num_layers - 1)]

    bottleneck = TanhBottleneck()
    ae_resnet_groups = config.ae_resnet_groups
    ae_patch_size = 1
    learning_rate = 1e-4
    batch_size = 256
    num_workers = 8

    model = setup_model(ae_in_channels, ae_channels, ae_multipliers, ae_factors, ae_num_blocks, ae_patch_size,
                        ae_resnet_groups, bottleneck)
    optimizer, criterion, scheduler = setup_training(learning_rate, (0.9, 0.999), model)
    train_loader, val_loader = setup_dataloader(batch_size, num_workers)
    accelerator, run_name = setup_accelerator(config)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    validation_loss = train_model(model, optimizer, criterion, scheduler, train_loader, accelerator, val_loader)

    if accelerator.is_main_process:
        checkpoint_dir = "/home/trey/experiment_rfdiffusion/models/autoencoder_sweep"
        checkpoint_path = os.path.join(checkpoint_dir, f"model_{run_name}.pth")
        save_checkpoint(accelerator.unwrap_model(model), optimizer, 100, checkpoint_dir, checkpoint_path)
    print(validation_loss)
    return validation_loss

def main():
    sweep_config = {
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 27
        },
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'validation_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'ae_channels': {
                'values': [32, 64, 128, 256]
            },
            'num_layers': {
                'min': 2,
                'max': 5
            },
            'ae_resnet_groups': {
                'values': [4, 8, 16]
            },
            'ae_factor_0': {
                'values': [1, 2, 4]
            },
            'ae_factor_1': {
                'values': [1, 2, 4]
            },
            'ae_factor_2': {
                'values': [1, 2, 4]
            },
            'ae_multi_0': {
                'values': [1]
            },
            'ae_multi_1': {
                'values': [1, 2, 4]
            },
            'ae_multi_2': {
                'values': [1, 2, 4]
            },
            'ae_multi_3': {
                'values': [1, 2, 4]
            },
            'ae_num_block_0': {
                'values': [1, 2, 4]
            },
            'ae_num_block_1': {
                'values': [1, 2, 4]
            },
            'ae_num_block_2': {
                'values': [1, 2, 4]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="autoencoder_project")
    wandb.agent(sweep_id, function=objective, count=100)

    print("Sweep completed! Best model and results saved.")

if __name__ == "__main__":
    main()
