import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.utils import set_random_seed
from utils.wandb_writer import WanDBWriter
from configs.BYOL import BYOL_config, ContrastiveImages, train_transforms
from losses.BYOL_loss import BYOL_loss
from models_contrastive.BYOL_model import BYOL_network


def main():
    config = BYOL_config()
    set_random_seed(config.seed)
    # loader
    train_dataset = datasets.STL10('data', 'unlabeled', download=True, transform=ContrastiveImages(train_transforms))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)

    test_dataset = datasets.STL10('data', 'train', download=True, transform=ContrastiveImages(train_transforms))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    # models
    model_online = BYOL_network(config)
    model_offline = BYOL_network(config)
    model_predict = nn.Sequential(
        nn.Linear(config.projection_size, config.mlp_hidden_size),
        nn.BatchNorm1d(config.mlp_hidden_size),
        nn.ReLU(),
        nn.Linear(config.mlp_hidden_size, config.projection_size)
    )
    model_online.to(config.device)
    model_offline.to(config.device)
    model_predict.to(config.device)

    for param_online, param_offline in zip(model_online.parameters(), model_offline.parameters()):
        param_offline.data.copy_(param_online.data)
        param_offline.requires_grad = False
    # loss, optimizer and hyperparameters
    optimizer = optim.Adam(list(model_online.parameters()) + list(model_predict.parameters()),
                           lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader))
    current_step = 0
    criterion = BYOL_loss(model_online, model_offline, model_predict)
    scaler = torch.cuda.amp.GradScaler()
    logger = WanDBWriter(config)
    # train
    tqdm_bar = tqdm(total=config.num_epochs * len(train_loader) - current_step)
    for epoch in range(config.num_epochs):
        for i, ((imgs_view1, imgs_view2), label) in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)
            imgs_view1, imgs_view2 = imgs_view1.to(config.device), imgs_view2.to(config.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, z_std1, z_std2 = criterion(imgs_view1, imgs_view2)
                # print(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logger.add_scalar('loss', loss.item())
            logger.add_scalar('std of z1', z_std1.item())
            logger.add_scalar('std of z2', z_std2.item())

            with torch.no_grad():
                for param_online, param_offline in zip(model_online.parameters(), model_offline.parameters()):
                    param_offline.data = (param_offline.data * config.moving_average
                                          + param_online.data * (1.0 - config.moving_average))

        if epoch >= 9:
            scheduler.step()

        logger.add_scalar('lr', optimizer.param_groups[0]["lr"])
        logger.add_image('img0', imgs_view1[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        logger.add_image('img1', imgs_view2[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)

        # evaluate
        if config.eval_epochs != 0 and epoch % config.eval_epochs == config.eval_epochs - 1:
            model_online.eval()
            model_predict.eval()
            current_step_test = 0
            for i, ((imgs_view1, imgs_view2), label) in enumerate(train_loader):
                current_step_test += 1
                logger.set_step(current_step, 'test')
                imgs_view1, imgs_view2 = imgs_view1.to(config.device), imgs_view2.to(config.device)

                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        loss, z_std1, z_std2 = criterion(imgs_view1, imgs_view2)

                logger.add_scalar('loss', loss.item())
                logger.add_scalar('std of z1', z_std1.item())
                logger.add_scalar('std of z2', z_std2.item())

            logger.add_image('img0', imgs_view1[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            logger.add_image('img1', imgs_view2[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            model_online.train()
            model_predict.train()

        if config.save_epochs != 0 and epoch % config.save_epochs == config.save_epochs - 1:
            torch.save(model_online.state_dict(), f'{config.model_save_name}_{epoch}.pth')

    logger.finish()


if __name__ == '__main__':
    main()
