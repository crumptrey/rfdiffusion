import os
import sys
import inspect
import json
from torchsummary import summary
import torch


def save_model_arguments(model, model_name, file_str):
    args = inspect.signature(model.__init__).parameters
    model_args = {param: getattr(model, param) for param in args if param != 'self'}
    model_attributes = model.__dict__
    with open(file_str, 'a') as file:
        file.write('\nModel Specifics \n')
        file.write('---------------------------- \n')
        file.write(f'Model Name: {model_name} \n')
        file.write(f'Model Arguments: {json.dumps(model_args)} \n')
        file.write(f'Model Arguments: {model_attributes} \n')


def logging_function(model, model_name, dataset_train_name, train_modulations, train_SNRs,
                     dataset_test_name, test_modulations, test_SNRs,
                     batch_size, epochs, criterion,
                     optimizer, device, train_transforms, test_transforms):
    # Logging
    file_str = '/home/trey/projectModClass/models/{0}/{1}/log.txt'.format(model_name, dataset_train_name)
    # Check if the file exists
    if os.path.exists(file_str):
        # If it exists, delete it
        os.remove(file_str)
    file = open(file_str, 'w')
    file.write('Dataset Specifics \n')
    file.write('---------------------------- \n')
    file.write('Training Dataset: {0} \n'.format(dataset_train_name))
    file.write('Training Modulation Set: {0} \n'.format(train_modulations))
    file.write('Training SNR Set: {0} \n'.format(train_SNRs))
    file.write('Train Transforms: {0} \n'.format(train_transforms))
    file.write('Testing Dataset: {0} \n'.format(dataset_test_name))
    file.write('Testing Modulation Set: {0} \n'.format(test_modulations))
    file.write('Testing SNR Set: {0} \n'.format(test_SNRs))
    file.write('Test Transforms: {0} \n'.format(test_transforms))
    file.write('\n')
    file.write('Model Specifics \n')
    file.write('---------------------------- \n')
    save_model_arguments(model, model_name, file_str)
    file.close()
    # Redirect the standard output to a variable
    stdout = sys.stdout
    sys.stdout = open(file_str, 'a')
    # Call summary to print the summary to the file
    device = torch.device(device)
    summary(model, (1024, 2), device='cpu')
    # Restore the standard output
    sys.stdout.close()
    sys.stdout = stdout
    file = open(file_str, 'a')
    file.write('\n')
    file.write('Training Specifics \n')
    file.write('---------------------------- \n')
    file.write('Batch Size: {0} \n'.format(batch_size))
    file.write('Epochs: {0} \n'.format(epochs))
    file.write('Loss Criterion: {0} \n'.format(criterion))
    file.write('Optimizer: {0} \n'.format(optimizer))
    file.write('Device: {0} \n'.format(device))
    file.write('\n')
    return file
