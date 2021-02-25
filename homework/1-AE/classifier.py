import torch
import torch.nn as nn
from blocks import RegularBlock

class Classifier(nn.Module):
    def __init__(self, n_classes=10, block=RegularBlock):
        super().__init__()
        self.encoder = nn.Sequential(
            block(1, 16, 3, stride=2),
            block(16, 32, 3, stride=2),
            block(32, 32, 3, stride=2),
            block(32, 64, 3, stride=2),
            block(64, 64, 3, stride=2).conv,
        )
        self.actv = nn.ELU()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        activations = self.get_activations(x)
        logits = self.fc2(activations)
        return logits
    
    def get_activations(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.actv(self.fc1(x))



class MnistClassifier(nn.Module):
    def __init__(self, n_classes=10, block=RegularBlock):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.actv = nn.ELU()

    def forward(self, x):
        activations = self.get_activations(x)
        logits = self.fc3(activations)
        return logits
    
    def get_activations(self, x):
        x = self.conv_layers(x)
        x = self.actv(self.fc1(x))
        return self.actv(self.fc2(x))