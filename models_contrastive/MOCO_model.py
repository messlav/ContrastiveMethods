import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class MOCO_network(nn.Module):
    def __init__(self, config):
        super(MOCO_network, self).__init__()
        self.config = config

        self.model_q = models.resnet18(num_classes=config.dim)
        self.model_k = models.resnet18(num_classes=config.dim)

        in_features = self.model_q.fc.in_features
        self.model_q.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, config.dim)
        )
        self.model_k.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, config.dim)
        )

        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # self.queue = torch.randn(config.dim, config.K)
        self.register_buffer("queue", torch.randn(config.dim, config.K))
        self.queue = F.normalize(self.queue, dim=0)
        # self.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.N = config.batch_size
        self.C = config.dim
        self.T = config.temperature
        self.device = config.device
        self.K = config.K

    def forward(self, q_imgs, k_imgs):
        q = F.normalize(self.model_q(q_imgs), dim=1)

        with torch.no_grad():
            for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
                param_k.data = (param_k.data * self.config.moving_average +
                                param_q.data * (1.0 - self.config.moving_average))
        
            # wo batch shuffling
            k = F.normalize(self.model_k(k_imgs), dim=1)

        l_pos = torch.bmm(q.view(self.N, 1, self.C), k.view(self.N, self.C, 1)).squeeze(-1)
        l_neg = torch.mm(q.view(self.N, self.C), self.queue.clone().detach().view(self.C, self.K))
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        assert self.K % self.N == 0
        ptr = int(self.queue_ptr)
        self.queue[:, ptr : ptr + self.N] = k.T
        ptr = (ptr + self.N) % self.K
        self.queue_ptr[0] = ptr

        return logits, labels
