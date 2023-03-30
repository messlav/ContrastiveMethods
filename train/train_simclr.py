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
from configs.simclr import SimCLR_config, ContrastiveImages, train_transforms
from losses.NCE_loss import NCE_loss


def main():
    config = SimCLR_config()
    set_random_seed(config.seed)

    train_dataset = datasets.STL10('data', 'unlabeled', download=True, transform=ContrastiveImages(train_transforms))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)

    test_dataset = datasets.STL10('data', 'train', download=True, transform=ContrastiveImages(train_transforms))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)

    model = models.resnet18(num_classes=config.num_features)
    in_features = model.fc.in_features
    projection_g = nn.Sequential(
        nn.Linear(in_features, in_features),
        nn.ReLU(),
        model.fc
    )
    model.fc = projection_g
    model.to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader))

    # loss, optimizer and hyperparameters
    current_step = 0
    criterion = NCE_loss(config)
    scaler = torch.cuda.amp.GradScaler()
    logger = WanDBWriter(config)

    tqdm_bar = tqdm(total=config.num_epochs * len(train_loader) - current_step)
    for epoch in range(config.num_epochs):
        for i, (imgs, _) in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)
            imgs = torch.cat(imgs, dim=0).to(config.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss, logits, labels, = criterion(outputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(logits.data, 1)
            accuracy = (predicted == labels).sum().item() / len(labels)

            logger.add_scalar('accuracy', accuracy)
            logger.add_scalar('loss', loss.item())

        if epoch >= 9:
            scheduler.step()

        logger.add_scalar('lr', optimizer.param_groups[0]["lr"])
        logger.add_image('img0', imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        logger.add_image('img1', imgs[1].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        logger.add_image('img2', imgs[2].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)

        if config.eval_epochs != 0 and epoch % config.eval_epochs == config.eval_epochs - 1:
            model.eval()
            current_step_test = 0
            for i, (imgs, labels) in enumerate(test_loader):
                current_step_test += 1
                logger.set_step(current_step, 'test')
                imgs = torch.cat(imgs, dim=0).to(config.device)

                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        outputs = model(imgs)
                        loss, logits, labels, = criterion(outputs)

                _, predicted = torch.max(logits.data, 1)
                accuracy = (predicted == labels).sum().item() / len(labels)

                logger.add_scalar('loss', loss)
                logger.add_scalar('accuracy', accuracy)

            logger.add_image('img0', imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            logger.add_image('img1', imgs[1].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            logger.add_image('img2', imgs[2].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            model.train()

        if config.save_epochs != 0 and epoch % config.save_epochs == config.save_epochs - 1:
            torch.save(model.state_dict(), f'model_SimCLR_2_{epoch}.pth')

    logger.finish()


if __name__ == '__main__':
    main()
