import torch.nn as nn
from torchvision import models


class BYOL_network(nn.Module):
    def __init__(self, config):
        super(BYOL_network, self).__init__()
        self.resnet = models.resnet18()
        self.in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(self.in_features, config.mlp_hidden_size),
            nn.BatchNorm1d(config.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(config.mlp_hidden_size, config.projection_size)
        )
        self.z_std = None
    
    def forward(self, img):
        # print(img.shape)
        z = self.resnet(img)
        # print('z', z.shape)
        z = z.view(z.shape[0], z.shape[1])
        # print('z', z.shape)
        self.z_std = z.std()
        q_z = self.projection(z)
        # print(q_z.shape)
        return q_z
