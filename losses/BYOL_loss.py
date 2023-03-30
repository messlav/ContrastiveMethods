import torch.nn as nn
import torch.nn.functional as F
import torch


class BYOL_loss(nn.Module):
    def __init__(self, online_model, offline_model, model_predict):
        super(BYOL_loss, self).__init__()
        self.online_model = online_model
        self.offline_model = offline_model
        self.model_predict = model_predict

    def regression_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, imgs_view1, imgs_view2):
        # print(0)
        online_network_out_1 = self.model_predict(self.online_model(imgs_view1))
        # print(1)
        z_std1 = self.online_model.z_std
        online_network_out_2 = self.model_predict(self.online_model(imgs_view2))
        z_std2 = self.online_model.z_std

        with torch.no_grad():
            offline_network_out_1 = self.offline_model(imgs_view1)
            offline_network_out_2 = self.offline_model(imgs_view2)

        loss1 = self.regression_loss(online_network_out_1, offline_network_out_2)
        loss2 = self.regression_loss(online_network_out_2, offline_network_out_1)
        # print(loss1.mean(), loss2.mean())

        return (loss1 + loss2).mean(), z_std1, z_std2
