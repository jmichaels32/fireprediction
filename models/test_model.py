import os
import torch
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

# Imports for models
from feature_rich import feature_rich
from zero_shot import zero_shot
from baseline_CNN import conv_net
from simple_baseline import simple_baseline

from utils.utils import generate_dataloader
from utils.model_utils import test

def graph_labels_vs_predictions(name, prev_day, labels, predictions):
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    num_cols = 3
    num_rows = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 8))

    for i in range(num_rows):
        prediction = (predictions[i] > 0).int()
        axes[i, 0].imshow(prev_day[i].detach().numpy(), cmap=CMAP, norm=NORM)
        axes[i, 1].imshow(labels[i].detach().numpy(), cmap=CMAP, norm=NORM)
        axes[i, 2].imshow(prediction.detach().numpy(), cmap=CMAP, norm=NORM)
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')
    
    axes[0, 0].set_title('Previous Day\nFire Segmentation\nMask', fontsize=10)
    axes[0, 1].set_title('Actual Next Day\nFire Segmentation\nMask', fontsize=10)
    axes[0, 2].set_title('Predicted Next Day\nFire Segmentation\nMask', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join('output_graphs', name + "_labels_predictions.png"))

def create_confusion_matrix(labels, predictions):
    confusion_matrix = np.zeros((2, 2))

    for i in range(len(labels)):
        label = labels[i].detach().numpy()
        prediction = (predictions[i] > 0).int().detach().numpy()

        # Update confusion matrix
        confusion_matrix[0, 0] += np.sum((label == 1) & (prediction == 1)) # True Positives
        confusion_matrix[0, 1] += np.sum((label == 0) & (prediction == 1)) # False Positives
        confusion_matrix[1, 0] += np.sum((label == 1) & (prediction == 0)) # False Negatives
        confusion_matrix[1, 1] += np.sum((label == 0) & (prediction == 0)) # True Negatives

    return confusion_matrix

def plot_confusion_matrix(name, model, data):
    confusion_matrix = np.zeros((2, 2))

    for inputs, labels in data:
        predictions = model(inputs)
        confusion_matrix += create_confusion_matrix(labels, predictions)

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['1', '0'])
    ax.set_yticklabels(['1', '0'])
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')

    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='w')

    ax.set_title('Confusion Matrix', pad=20)
    fig.tight_layout()
    plt.savefig(os.path.join('output_graphs', name + "_confusion_matrix.png"))

if __name__ == "__main__":
    # ---------------------- Parameters to change ----------------------
    model_name = 'simple_baseline'
    model = simple_baseline()
    data_name = 'test'
    # -------------------------------------------------------------------

    model_path = 'best_' + model_name + '.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data = generate_dataloader(data_name, 32)
    for datapoint in data:
        inputs, labels = datapoint
        predictions = model(inputs)
        graph_labels_vs_predictions(model_name, inputs[:, -1, :, :], labels, predictions)
        break

    plot_confusion_matrix(model_name, model, data)
    test(model, 32)
