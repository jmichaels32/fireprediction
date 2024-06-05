import os 
import torch
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import colors
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def _parse_function(proto):
    feature_description = {
        'elevation': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'th': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'vs': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'tmmn': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'tmmx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sph': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'pr': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'pdsi': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'NDVI': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'population': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'erc': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'PrevFireMask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'FireMask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }

    return tf.io.parse_single_example(proto, feature_description)

def load_tfrecords(tfrecord_paths):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_paths)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def tf_dataset_to_numpy(tf_dataset):
    features = []
    for record in tf_dataset:
        feature = {key: value.numpy() for key, value in record.items()}
        features.append(feature)
    return features

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sph = torch.tensor(sample['sph'], dtype=torch.float32).view((64, 64))
        prev_fire_mask = torch.tensor(sample['PrevFireMask'], dtype=torch.float32).view((64, 64))
        population = torch.tensor(sample['population'], dtype=torch.float32).view((64, 64))
        pdsi = torch.tensor(sample['pdsi'], dtype=torch.float32).view((64, 64))
        fire_mask = torch.tensor(sample['FireMask'], dtype=torch.float32).view((64, 64))
        tmmx = torch.tensor(sample['tmmx'], dtype=torch.float32).view((64, 64))
        elevation = torch.tensor(sample['elevation'], dtype=torch.float32).view((64, 64))
        ndvi = torch.tensor(sample['NDVI'], dtype=torch.float32).view((64, 64))
        th = torch.tensor(sample['th'], dtype=torch.float32).view((64, 64))
        vs = torch.tensor(sample['vs'], dtype=torch.float32).view((64, 64))
        pr = torch.tensor(sample['pr'], dtype=torch.float32).view((64, 64))
        tmmn = torch.tensor(sample['tmmn'], dtype=torch.float32).view((64, 64))
        erc = torch.tensor(sample['erc'], dtype=torch.float32).view((64, 64))
        
        return torch.stack([
            elevation,
            sph,
            vs,
            tmmn,
            tmmx,
            th,
            pr,
            pdsi,
            ndvi,
            population,
            erc,
            prev_fire_mask,
        ], dim=0), fire_mask

def generate_dataloader(tfrecord_type, batch_size):
    tfrecord_paths = _generate_tfrecords(tfrecord_type)
    parsed_dataset = load_tfrecords(tfrecord_paths)
    numpy_data = tf_dataset_to_numpy(parsed_dataset)
    pytorch_dataset = CustomDataset(numpy_data)
    pytorch_dataloader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=True)
    return pytorch_dataloader

def _generate_tfrecords(type):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.realpath(os.path.join(dir_path, "../../data"))
    tfrecord_paths = []
    if type == 'train':
        tfrecord_paths = [os.path.join(path, f"next_day_wildfire_spread_train_{i:02d}.tfrecord") for i in range(1)]
    elif type == 'eval':
        tfrecord_paths = [os.path.join(path, f"next_day_wildfire_spread_eval_{i:02d}.tfrecord") for i in range(2)]
    elif type == 'test':
        tfrecord_paths = [os.path.join(path, f"next_day_wildfire_spread_test_{i:02d}.tfrecord") for i in range(2)]
    else:
        raise ValueError("Invalid type")
    return tfrecord_paths

def plot_samples(dataloader, n_rows: int):
    TITLES = ["Elevation", "Wind\nDirection", "Wind\nSpeed", "Min\nTemp", "Max\nTemp", "Humidity", "Precipitation", "Drought\nIndex", "Vegetation", "Population\nDensity", "Energy\nRelease\nComponent", "Prev\nFire\nMask", "Fire\nMask"]
    fig = plt.figure(figsize=(15, 6.5))
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

    inputs, labels = None, None
    for elem in dataloader:
        inputs, labels = elem
        break

    n_features = 12
    for i in range(n_rows):
        for j in range(n_features + 1):
            plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
            if i == 0:
                plt.title(TITLES[j], fontsize=13)
            if j < n_features - 1:
                plt.imshow(inputs[i, j, :, :], cmap='viridis')
            if j == n_features - 1:
                plt.imshow(inputs[i, -1, :, :], cmap=CMAP, norm=NORM)
            if j == n_features:
                plt.imshow(labels[i, :, :], cmap=CMAP, norm=NORM) 
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def graph_losses(losses, losses_dice, losses_wbce, validation_losses, subset_size=1000):
    if len(losses) > subset_size:
        indices = np.linspace(0, len(losses) - 1, subset_size).astype(int)
        losses = np.array(losses)[indices]
        losses_dice = np.array(losses_dice)[indices]
        losses_wbce = np.array(losses_wbce)[indices]
    
    val_indices = np.linspace(0, len(losses) - 1, len(validation_losses)).astype(int)

    plt.figure(figsize=(14, 8))
    
    # Plot training losses
    plt.plot(losses, label='Training Loss')
    plt.plot(losses_dice, label='Dice Loss')
    plt.plot(losses_wbce, label='WBCE Loss')
    
    # Plot validation losses on the same graph
    plt.plot(val_indices, validation_losses, label='Validation Loss', marker='o')
    
    plt.title('Training and Validation Losses')
    plt.xlabel('Training Samples / Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    
    '''if len(losses) > subset_size:
        indices = np.linspace(0, len(losses) - 1, subset_size).astype(int)
        losses = np.array(losses)[indices]
        losses_dice = np.array(losses_dice)[indices]
        losses_wbce = np.array(losses_wbce)[indices]
    
    epochs = range(1, len(validation_losses) + 1)
    
    plt.figure(figsize=(14, 8))
    
    # Plot training losses
    plt.subplot(2, 1, 1)
    plt.plot(losses, label='Training Loss')
    plt.plot(losses_dice, label='Dice Loss')
    plt.plot(losses_wbce, label='WBCE Loss')
    plt.title('Training Losses')
    plt.xlabel('Training Samples')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation losses
    plt.subplot(2, 1, 2)
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='o')
    plt.title('Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='train')
    parser.add_argument("--samples", default=5, type=int)
    args = parser.parse_args()

    dataloader = generate_dataloader(args.type, 32)
    plot_samples(dataloader, args.samples)