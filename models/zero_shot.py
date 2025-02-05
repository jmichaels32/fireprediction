import torch
import argparse
import torch.nn as nn
import torchvision.models as models

from utils.model_utils import train, test
from utils.utils import generate_dataloader

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
class zero_shot(nn.Module):
    def __init__(self):
        super(zero_shot, self).__init__()

        # Data processing (upscale then downscale)
        self.upscale = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.downscale = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64 * 64),
            Reshape(64, 64),
        )
    
    def forward(self, x):
        # Mask out features for data reduction setting
        elevation = x[:, 0, :, :]
        vegetation = x[:, 8, :, :]
        population_density = x[:, 9, :, :]
        x_iter = torch.stack([elevation, vegetation, population_density], dim=1)

        x_iter = self.upscale(x_iter)
        x_iter = self.downscale(x_iter)
        x_iter = self.linear(x_iter)
        return x_iter
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = zero_shot()
    train_losses, val_losses = train(model, 'zero_shot', args.lr, args.batch_size, args.epochs)
    test(model, args.batch_size)