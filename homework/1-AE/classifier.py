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
            block(32, 32, 3, stride=2),
            block(32, 64, 3, stride=1).conv,
        )
        self.fc = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        activations = self.get_activations(x)
        probs = torch.softmax(x)
        return probs
    
    def get_activations(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = x.view(-1, 1024)
        return self.fc(x)
