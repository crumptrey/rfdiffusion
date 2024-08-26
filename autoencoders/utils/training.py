import os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np  # linear algebra
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import utils.load_datasets
from utils import *
from networks import *
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from utils.general import create_directory
from utils.metrics import plot_confusion_matrix, plot_pcc, save_classification_report, plot_distributions, plot_confusion_matrix_snr, plot_losses

def evaluate(model, directory, epoch, data_loader, criterion, phase, device, train_modulations, file):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    correct_labels = []
    predicted_labels = []
    snr_correct_dict = {}
    snr_total_dict = {}
    with torch.no_grad():
        for batch in data_loader:
            data, mod_types, snrs = batch
            correct_labels.append(mod_types)
            data = data.to(device)
            labels = mod_types.to(device)

            outputs = model(data)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predicted_labels.append(predicted.cpu())
            correct_predictions += (predicted == labels).sum().item()
            correct = (predicted == labels).cpu().numpy()
            total_samples += labels.size(0)
            for i, snr in enumerate(snrs):
                snr = snr.item()
                if snr not in snr_correct_dict:
                    snr_correct_dict[snr] = 0
                    snr_total_dict[snr] = 0
                snr_correct_dict[snr] += correct[i]
                snr_total_dict[snr] += 1

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    snr_accuracy_dict = {snr: (snr_correct_dict[snr] / snr_total_dict[snr]) for snr in snr_correct_dict if
                         snr_total_dict[snr] != 0}
    print('{0} - Average Loss: {1}, Accuracy: {2}'.format(phase, avg_loss, accuracy))
    file.write('{0} - Average Loss: {1}, Accuracy: {2} \n'.format(phase, avg_loss, accuracy))
    plot_pcc(snr_accuracy_dict, directory, epoch)
    if phase == 'Valid' or phase == 'Test':
        # Plot Probability of Correct Classification vs. SNR
        correct_labels = torch.cat(correct_labels)
        predicted_labels = torch.cat(predicted_labels)
        y_true = correct_labels
        y_pred = predicted_labels
        plot_confusion_matrix(y_true, y_pred, train_modulations, directory, epoch)
        return avg_loss
    elif phase == 'Test':
        # Plot Probability of Correct Classification vs. SNR
        correct_labels = torch.cat(correct_labels)
        predicted_labels = torch.cat(predicted_labels)
        y_true = correct_labels
        y_pred = predicted_labels
        plot_confusion_matrix(y_true, y_pred, train_modulations, directory, epoch)
        for snr in snr_correct_dict:
            correct_labels_snr = []  # Correct labels for the current SNR
            predicted_labels_snr = []  # Predicted labels for the current SNR

            # Collect correct and predicted labels for the current SNR
            for i, s in enumerate(snrs):
                if s.item() == snr:
                    correct_labels_snr.append(mod_types[i].item())
                    predicted_labels_snr.append(predicted[i].item())

            y_true = torch.tensor(correct_labels_snr)
            y_pred = torch.tensor(predicted_labels_snr)
            plot_confusion_matrix_snr(y_true, y_pred, train_modulations, directory, snr)
            save_classification_report(y_true, y_pred, train_modulations, directory, snr)


def train(model, model_name, dataset_train_name, train_modulations, dataset_test_name, optimizer, loss_criterion,
          epochs,
          train_dataset, val_dataset, test_dataset, batch_size, plot_every, save_every, device, file, checkpoint=None):
    print('Training on {0} and testing on {1} for {2} model'.format(dataset_train_name, dataset_test_name,
                                                                    model_name))
    file.write('Training on {0} and testing on {1} for {2} model \n'.format(dataset_train_name, dataset_test_name,
                                                                            model_name))
    file.write('Training Progress \n')
    file.write('---------------------------- \n')
    device = torch.device(device)
    model.to(device)

    # Progress Directories
    trainDir = '/home/trey/projectModClass/models/{0}/{1}'.format(model_name, dataset_train_name)
    create_directory(trainDir)
    # Validation
    validDir = "{0}/validation".format(trainDir)
    create_directory(validDir)
    # Testing
    testDir = "{0}/test_{1}".format(trainDir, dataset_test_name)
    create_directory(testDir)
    # Checkpoints
    checkpoint_dir = "{0}/checkpoints".format(trainDir)
    create_directory(checkpoint_dir)
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    epoch_hist = []
    for epoch in range(epochs):
        epoch_hist.append(epoch)
        model.train(True)
        total_train_loss = 0.0
        total_batch_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, train_batch in enumerate(train_loader):
            data, mod_types, snrs = train_batch
            data = data.to(device)
            labels = mod_types.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = loss_criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()
            total_batch_loss += loss.item()
            if (batch_idx + 1) % plot_every == 0:  # Plot every 'plot_every' batches
                avg_batch_loss = total_batch_loss / plot_every
                print(
                    'Epoch [{}/{}], Batch [{}/{}], Average Batch Loss: {:.4f}'.format(epoch + 1, epochs,
                                                                                      batch_idx + 1,
                                                                                      num_batches, avg_batch_loss))
                file.write('Epoch [{}/{}], Batch [{}/{}], Average Batch Loss: {:.4f} \n'.format(epoch + 1, epochs,
                                                                                                batch_idx + 1,
                                                                                                num_batches,
                                                                                                avg_batch_loss))
                total_batch_loss = 0.0

        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, epochs, avg_train_loss))
        file.write('Epoch [{}/{}], Train Loss: {:.4f} \n'.format(epoch + 1, epochs, avg_train_loss))
        val_loss = evaluate(model, validDir, epoch, val_loader, loss_criterion, 'Valid', device, train_modulations,
                            file)
        val_losses.append(val_loss)
        # Save model checkpoint every 'save_checkpoint_every' epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch + 1))
            torch.save(model.state_dict(), checkpoint_path)
            print('Model checkpoint saved: {}'.format(checkpoint_path))

        # Plot training and validation losses
        plt.plot(epoch_hist, train_losses, label='Train Loss')
        plt.plot(epoch_hist, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(trainDir, 'loss_plot.png'))
        plt.show()
        plt.close()
    avg_loss = evaluate(model, testDir, epochs, test_loader, loss_criterion, 'Test', device, train_modulations,
                        file)


def test(model, model_name, dataset_train_name, train_modulations, dataset_test_name, optimizer, loss_criterion,
         epochs,
         train_dataset, val_dataset, test_dataset, batch_size, plot_every, save_every, device, file, checkpoint=None):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainDir = '/home/trey/projectModClass/models/{0}/{1}'.format(model_name, dataset_train_name)
    testDir = "{0}/test_{1}".format(trainDir, dataset_test_name)
    create_directory(testDir)
    # Checkpoints
    checkpoint_path = "{0}/checkpoints/model_epoch_{1}.pth".format(trainDir, epochs)
    checkpoint = torch.load(checkpoint_path)
    device = torch.device(device)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    avg_loss = evaluate(model, testDir, epochs, test_loader, loss_criterion, 'Test', device, train_modulations,
                        file)
