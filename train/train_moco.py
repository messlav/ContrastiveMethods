import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.utils import set_random_seed
from utils.wandb_writer import WanDBWriter
from configs.moco import MOCO_config, ContrastiveImages, train_transforms
from models_contrastive.MOCO_model import MOCO_network


def main():
    config = MOCO_config()
    set_random_seed(config.seed)
    # loader
    train_dataset = datasets.STL10('data', 'unlabeled', download=True, transform=ContrastiveImages(train_transforms))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)

    test_dataset = datasets.STL10('data', 'train', download=True, transform=ContrastiveImages(train_transforms))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    # models
    model = MOCO_network(config)
    model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader))
    torch.backends.cudnn.benchmark = True
    current_step = 0
    scaler = torch.cuda.amp.GradScaler()
    logger = WanDBWriter(config)
    tqdm_bar = tqdm(total=config.num_epochs * len(train_loader) - current_step)
    model.train()
    for epoch in range(config.num_epochs):
        for i, ((q_imgs, k_imgs), label) in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)
            q_imgs, k_imgs = q_imgs.to(config.device), k_imgs.to(config.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output, target = model(q_imgs, k_imgs)
                loss = criterion(output, target)
                # print(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target).sum().item() / len(target)

            logger.add_scalar('loss', loss.item())
            logger.add_scalar('accuracy', accuracy)

        if epoch >= 9:
            scheduler.step()

        logger.add_scalar('lr', optimizer.param_groups[0]["lr"])
        logger.add_image('img0', q_imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        logger.add_image('img1', k_imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)

        # evaluate
        if config.eval_epochs != 0 and epoch % config.eval_epochs == config.eval_epochs - 1:
            model.eval()
            current_step_test = 0
            for i, ((q_imgs, k_imgs), label) in enumerate(train_loader):
                current_step_test += 1
                logger.set_step(current_step, 'test')
                q_imgs, k_imgs = q_imgs.to(config.device), k_imgs.to(config.device)

                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        output, target = model(q_imgs, k_imgs)
                        loss = criterion(output, target)

                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item() / len(target)

                logger.add_scalar('loss', loss.item())
                logger.add_scalar('accuracy', accuracy)

            logger.add_image('img0', q_imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            logger.add_image('img1', k_imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            model.train()

        if config.save_epochs != 0 and epoch % config.save_epochs == config.save_epochs - 1:
            torch.save(model.state_dict(), f'{config.model_save_name}_{epoch}.pth')

    logger.finish()


if __name__ == '__main__':
    main()
