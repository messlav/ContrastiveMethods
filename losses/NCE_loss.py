import torch
import torch.nn as nn
import torch.nn.functional as F


class NCE_loss(nn.Module):
    def __init__(self, config, temperature=0.1):
        super(NCE_loss, self).__init__()
        self.config = config
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.n_views = 2
        self.shape = self.config.batch_size * self.n_views
        self.diagonal = torch.eye(self.shape, dtype=torch.bool, device=self.config.device)
        # print(self.diagonal.device, 'hey')
    
    def forward(self, outputs):
        labels_matrix = [torch.arange(self.config.batch_size) for i in range(self.n_views)]
        labels_matrix = torch.cat(labels_matrix, 0)
        labels_matrix = (labels_matrix.unsqueeze(0) == labels_matrix.unsqueeze(1)).to(self.config.device)
        # labels_matrix.to(self.config.device)
        # print(self.config.device, labels_matrix.device, self.diagonal.device)
        # print(labels_matrix.shape)
        labels_matrix = labels_matrix[~self.diagonal].view(self.shape, -1)

        outputs = F.normalize(outputs)

        similarity = outputs @ outputs.T
        # print(similarity.shape)
        similarity = similarity[~self.diagonal].view(self.shape, -1)

        negative = similarity[~labels_matrix.bool()].view(self.shape, -1) / self.temperature
        positive = similarity[labels_matrix.bool()].view(self.shape, -1) / self.temperature

        logits = torch.cat([positive, negative], dim=1)
        labels = torch.zeros(self.shape, dtype=torch.long).to(self.config.device)
        loss = self.criterion(logits, labels)
        
        return loss, logits, labels
