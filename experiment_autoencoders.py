import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from audio_encoders_pytorch.audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck
import utils.load_datasets
import os
from tqdm import tqdm
from accelerate import Accelerator
import random
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train an autoencoder model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_model(config):
    return AutoEncoder1d(
        in_channels=config['ae_in_channels'],
        channels=config['ae_channels'],
        multipliers=config['ae_multipliers'],
        factors=config['ae_factors'],
        num_blocks=config['ae_num_blocks'],
        patch_size=config['ae_patch_size'],
        resnet_groups=config['ae_resnet_groups'],
        bottleneck=TanhBottleneck()  # You might want to make this configurable too
    )


def setup_training(config, model):
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], betas=tuple(config['adam_betas']))
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['gamma'])
    return optimizer, criterion, scheduler


def setup_dataloader(config):
    dataset_class = getattr(utils.load_datasets, config['dataset_name'])
    train_dataset = dataset_class(config['dataset_path'])
    return DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                      num_workers=config['num_workers'])


def setup_accelerator(config):
    accelerator = Accelerator(log_with="wandb")
    run_name = str(random.randint(0, 10e5))
    accelerator.init_trackers(
        config['project_name'],
        config=config,
        init_kwargs={"wandb": {"name": run_name}}
    )
    return accelerator, run_name


def train_model(model, optimizer, criterion, scheduler, data_loader, accelerator, config):
    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )
    num_training_steps = config['epochs'] * len(data_loader)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    step = 1

    for epoch in range(config['epochs']):
        for x, _ in data_loader:
            y = model(x)
            loss = criterion(y, x)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            accelerator.log({"training_loss": loss, "learning_rate": scheduler.get_last_lr()[0]}, step=step)
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
    args = parse_args()
    config = load_config(args.config)

    # Construct model_save_dir
    config['model_save_dir'] = os.path.join(config['base_save_dir'], config['project_name'])
    os.makedirs(config['model_save_dir'], exist_ok=True)

    accelerator, run_name = setup_accelerator(config)

    model = setup_model(config)
    optimizer, criterion, scheduler = setup_training(config, model)
    data_loader = setup_dataloader(config)


    print(f"Training on {accelerator.num_processes} GPUs")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Models will be saved in: {config['model_save_dir']}")

    train_model(model, optimizer, criterion, scheduler, data_loader, accelerator, config)

    if accelerator.is_main_process:
        final_checkpoint_path = os.path.join(config['model_save_dir'], f'model_{run_name}.pth')
        save_checkpoint(accelerator.unwrap_model(model), optimizer, config['epochs'], config['model_save_dir'],
                        final_checkpoint_path)

    print("Training complete and models saved.")


if __name__ == "__main__":
    main()