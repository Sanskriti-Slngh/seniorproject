import torch
import torch.nn as nn

"""
Sanskriti Singh
Define my pytorch model architecture - simple FNN
"""
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(16, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)

        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 32)
        self.layer7 = nn.Linear(32, 16)
        self.layer8 = nn.Linear(16, 16)
        self.layer9 = nn.Linear(16, 8)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        if hasattr(self, 'layer2'):
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = torch.relu(self.layer7(x))
        x = torch.relu(self.layer8(x))
        x = torch.softmax(self.layer9(x), dim=1)
        return x

