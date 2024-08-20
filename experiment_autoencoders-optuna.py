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
import optuna
from torch.utils.data import random_split
import numpy as np

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
    """
    Evaluate the autoencoder model on a validation dataset.

    Args:
        model (nn.Module): The autoencoder model to evaluate.
        data_loader (DataLoader): The validation data loader.
        accelerator (Accelerator): The accelerator object.

    Returns:
        float: The average reconstruction loss on the validation set.
    """
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
        bottleneck=bottleneck)


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


def train_model(model, optimizer, criterion, scheduler, data_loader, accelerator):
    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )
    num_training_steps = 10 * len(data_loader)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    step = 1

    for epoch in range(10):
        for x, _ in data_loader:
            y = model(x, with_info = False)
            loss = criterion(y, x)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            accelerator.log({"training_loss": loss, "learning_rate": scheduler.get_last_lr()[0]}, step=step)
            step += 1

        if epoch % 10 == 0 and accelerator.is_main_process:
            save_checkpoint(model, optimizer, epoch, "/home/trey/experiment_rfdiffusion/models/rfdiffusion_diffusion",
                            f'model_epoch_{epoch}.pth')

    accelerator.end_training()


def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)


def objective(trial):
    # Get the hyperparameters from the trial
    ae_in_channels = 2
    ae_channels = trial.suggest_categorical('ae_channels', [32, 64, 128, 256, 512])
    # Suggest number of layers
    num_layers = trial.suggest_int('num_layers', 2, 5)
    # Generate factors for num_layers - 1 layers
    ae_factors = [trial.suggest_int(f'ae_factor_{i}', 2, 4) for i in range(num_layers - 1)]
    # Initialize the multipliers list with the first multiplier as 1
    # Initialize the multipliers list with the first multiplier as 1
    ae_multipliers = [1]

    # Generate the remaining multipliers
    for i in range(1, num_layers):
        # Suggest an exponent that will produce a power of 2 greater than or equal to the previous multiplier
        min_exp = int(np.log2(ae_multipliers[-1]))  # Minimum exponent based on the last multiplier
        next_multiplier_exponent = trial.suggest_int(f'ae_multi_{i}', min_exp, 4)  # Max exponent corresponds to 2^4=16
        next_multiplier = 2 ** next_multiplier_exponent
        ae_multipliers.append(next_multiplier)

    # Generate ae_num_blocks with values either 1 or a factor of 2 (i.e., 2 or 4)
    ae_num_blocks = []
    for i in range(num_layers - 1):
        block_choice = trial.suggest_int(f'ae_num_block_{i}', 1, 4)
        # Adjust to ensure the value is either 1 or a factor of 2
        if block_choice == 3:  # If it was 3, default it to 2
            block_choice = 2
        ae_num_blocks.append(block_choice)


    bottleneck = TanhBottleneck()
    ae_resnet_groups = 8
    learning_rate = 1e-4
    batch_size = 512
    num_workers = 8
    ae_patch_size = 1
    config = {
        'ae_in_channels': ae_in_channels,
        'ae_channels': ae_channels,
        'num_layers': num_layers,
        'ae_multipliers': ae_multipliers,
        'ae_factors': ae_factors,
        'ae_num_blocks': ae_num_blocks,
        'ae_patch_size': ae_patch_size,
        'ae_resnet_groups': ae_resnet_groups,
        'bottleneck': bottleneck,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_workers': num_workers
    }

    # Create the model with the suggested hyperparameters


    model = setup_model(ae_in_channels, ae_channels, ae_multipliers, ae_factors, ae_num_blocks, ae_patch_size,
                    ae_resnet_groups, bottleneck)
    # Set up the training process
    optimizer, criterion, scheduler = setup_training(learning_rate, (0.9, 0.999), model)
    train_loader, val_loader = setup_dataloader(batch_size, num_workers)
    accelerator, _ = setup_accelerator(config)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    # Train the model
    train_model(model, optimizer, criterion, scheduler, train_loader, accelerator)

    # Evaluate the model on the validation set
    validation_loss = evaluate_model(model, val_loader, accelerator)
    accelerator.log({"validation_loss": validation_loss}, step=0)
    return validation_loss


def main():
    args = parse_args()

    study = optuna.create_study(direction='minimize', storage="sqlite:///db.sqlite3",
                                study_name="autoencoder_architecture", load_if_exists = True)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)
    study.optimize(objective, n_trials=100)

    print('Best hyperparameters: ', study.best_params)
    print('Best validation loss: ', study.best_value)


    best_config = {
        'ae_in_channels': 2,
        'ae_channels': study.best_params['ae_channels'],
        'num_layers': study.best_params['num_layers'],
        'ae_multipliers': [study.best_params[f'ae_multi_{i}'] for i in range(study.best_params['num_layers'])],
        'ae_factors': [study.best_params[f'ae_factor_{i}'] for i in range(study.best_params['num_layers'] - 1)],
        'ae_num_blocks': [study.best_params[f'ae_num_block_{i}'] for i in range(study.best_params['num_layers'])],
        'ae_patch_size': 1,
        'ae_resnet_groups': 8,
        'bottleneck': TanhBottleneck(),
        'learning_rate': study.best_params['learning_rate'],
        'batch_size': study.best_params['batch_size'],
        'num_workers': 8
    }

    best_model = setup_model(
        best_config['ae_in_channels'],
        best_config['ae_channels'],
        best_config['ae_multipliers'],
        best_config['ae_factors'],
        best_config['ae_num_blocks'],
        best_config['ae_patch_size'],
        best_config['ae_resnet_groups'],
        best_config['bottleneck']
    )
    accelerator, run_name = setup_accelerator(best_config)
    best_optimizer, best_criterion, best_scheduler = setup_training(
        study.best_params['learning_rate'], (0.9, 0.999), best_model
    )
    best_data_loader = setup_dataloader(
        study.best_params['batch_size'], study.best_params['num_workers']
    )

    print(f"Training on {accelerator.num_processes} GPUs")
    print(f"Number of parameters: {sum(p.numel() for p in best_model.parameters() if p.requires_grad)}")
    print(f"Models will be saved in: /home/trey/experiment_rfdiffusion/models/rfdiffusion_diffusion")

    train_model(best_model, best_optimizer, best_criterion, best_scheduler, best_data_loader, accelerator)

    if accelerator.is_main_process:
        final_checkpoint_path = os.path.join("/home/trey/experiment_rfdiffusion/models/rfdiffusion_diffusion",
                                             f'model_{run_name}.pth')
        save_checkpoint(accelerator.unwrap_model(best_model), best_optimizer, 100,
                        "/home/trey/experiment_rfdiffusion/models/rfdiffusion_diffusion", final_checkpoint_path)

    print("Training complete and models saved.")


if __name__ == "__main__":
    main()
