import torch
import argparse
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from baseline_CNN import conv_net
from utils.utils import generate_dataloader

def graph_labels_vs_predictions(labels, predictions):
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    prediction = None
    index = 0
    for pred in predictions:
        index += 1
        pred_temp = (pred > 0.5).int()
        if np.any(pred_temp.detach().numpy() > 0):
            prediction = pred_temp
            break

    axes[0].imshow(labels[index].detach().numpy(), cmap=CMAP, norm=NORM)
    axes[1].imshow(prediction.detach().numpy(), cmap=CMAP, norm=NORM)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='best_baseline.pth')
    args = parser.parse_args()

    model = conv_net()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_data = generate_dataloader('test', 32)
    for datapoint in test_data:
        inputs, labels = datapoint
        predictions = model(inputs)
        graph_labels_vs_predictions(labels, predictions)
        break
