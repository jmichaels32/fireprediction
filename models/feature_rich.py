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

class feature_rich(nn.Module):
    def __init__(self):
        super(feature_rich, self).__init__()

        # Data processing (upscale then downscale)
        self.upscale = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.downscale = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.ReLU(),
            #Reshape(64, 64),
            #nn.Flatten(1, 2),
        )
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.segmentation_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 2 * 2, 64 * 64),
            nn.ReLU(),
            Reshape(64, 64),
        )
    
    def forward(self, x):
        x_iter = x
        x_iter = self.upscale(x_iter)
        x_iter = self.downscale(x_iter)
        x_iter = self.backbone(x_iter)
        x_iter = self.segmentation_head(x_iter)
        return x_iter
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = feature_rich()
    train_losses, val_losses = train(model, 'feature_rich', args.lr, args.batch_size, args.epochs)
    test(model, args.batch_size)