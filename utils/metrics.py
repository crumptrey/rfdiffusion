import os
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

def plot_losses(epoch_hist, train_losses,val_losses, trainDir):
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

def plot_pcc(snr_accuracy_dict, directory, epoch):
    sorted_snr_accuracy = sorted(snr_accuracy_dict.items())
    snr_values = [snr for snr, _ in sorted_snr_accuracy]
    accuracy_values = [accuracy for _, accuracy in sorted_snr_accuracy]
    plt.plot(snr_values, accuracy_values, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Probability of Correct Classification')
    plt.title('Probability of Correct Classification vs. SNR')
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim(-20, 20)
    plt.grid(True)
    plt.savefig(os.path.join(directory, f'pcc_{epoch}.png'))  # Save the figure after displaying it
    plt.show()
    plt.close()  # Close the figure after saving it

# Define a function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, directory, epoch):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=False, fmt="", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'confusion_matrix_normalized_{epoch}.png'))
    plt.show()
    plt.close()

def plot_confusion_matrix_snr(y_true, y_pred, classes, directory, snr):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix
    plt.figure(figsize=(10, 8))
    plt.title('Confusion Matrix (SNR={})'.format(snr))
    sns.heatmap(cm_normalized, annot=False, fmt="", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'confusion_matrix_SNR_{snr}.png'))
    plt.close()

def save_classification_report(y_true, y_pred, classes, directory, snr):
    report = classification_report(y_true, y_pred, classes)
    with open(os.path.join(directory, f'classification_report_SNR_{snr}.txt'), 'w') as f:
        f.write(report)

def plot_distributions(data_loader, classes, snrs, title):
    modulation_counts = {modulation: {snr: 0 for snr in snrs} for modulation in classes}
    for batch in data_loader:
        data, mod_types, snr_batch = batch
        for modulation, snr in zip(mod_types, snr_batch):
            modulation = modulation.item()
            snr = snr.item()
            modulation_counts[modulation][snr] += 1

    num_snrs = len(snrs)
    fig, axes = plt.subplots(nrows=len(classes), ncols=num_snrs, figsize=(14, 8), sharex=True, sharey=True)

    for i, modulation in enumerate(classes):
        for j, snr in enumerate(snrs):
            ax = axes[i, j]
            counts = modulation_counts[modulation][snr]
            ax.bar(range(len(counts)), counts)
            ax.set_title(f'Modulation: {modulation}, SNR: {snr} dB')

    plt.tight_layout()
    plt.show()


def create_directory(directory):
    """
    Create a directory if it doesn't exist.

    Parameters:
        directory (str): The path of the directory to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")