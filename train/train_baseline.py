import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from utils.utils import set_random_seed
from utils.wandb_writer import WanDBWriter
from configs.baseline import SupervisedBaselineConfig, BaselineDataConfig


def main():
    config = SupervisedBaselineConfig()
    data_config = BaselineDataConfig()
    set_random_seed(config.seed)
    train_dataset = datasets.STL10('data', 'train', download=True, transform=data_config.train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    test_dataset = datasets.STL10('data', 'test', download=True, transform=data_config.test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers, pin_memory=True)

    model = models.resnet18(num_classes=10)
    model = model.to(config.device)

    # loss, optimizer and hyperparameters
    current_step = 0
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
    #                       weight_decay=config.weight_decay, nesterov=True)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                          weight_decay=config.weight_decay, nesterov=config.nesterov)
    scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    scaler = torch.cuda.amp.GradScaler()
    logger = WanDBWriter(config)
    # train
    tqdm_bar = tqdm(total=config.num_epochs * len(train_loader) - current_step)
    for epoch in range(config.num_epochs):
        accuracy, loss = 0, 0
        for i, (imgs, labels) in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)

            imgs, labels = imgs.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()
            loss += loss.item()

        scheduler.step()
        logger.add_scalar('train/loss', loss / len(train_loader))
        logger.add_scalar('train/accuracy', accuracy / len(train_loader) / config.batch_size)
        # logger.add_scalar('lr', scheduler.get_last_lr())
        logger.add_image(f'train/img0', imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        logger.add_image(f'train/img1', imgs[1].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        logger.add_image(f'train/img2', imgs[2].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)

        # evaluate
        model.eval()
        accuracy, loss = 0, 0
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(config.device), labels.to(config.device)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()
            loss += loss.item()

        logger.add_scalar('test/loss', loss / len(test_loader))
        logger.add_scalar('test/accuracy', accuracy / len(test_loader) / config.batch_size)
        logger.add_image(f'test/img0', imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        logger.add_image(f'test/img1', imgs[1].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        logger.add_image(f'test/img2', imgs[2].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        model.train()

        if config.save_epochs != 0 and epoch % config.save_epochs == config.save_epochs - 1:
            torch.save(model.state_dict(), f'model_{epoch}.pth')

    logger.finish()


if __name__ == '__main__':
    main()
