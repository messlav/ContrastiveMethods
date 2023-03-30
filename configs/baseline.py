from dataclasses import dataclass
import torch
import torchvision.transforms as T


@dataclass
class SupervisedBaselineConfig:
    wandb_project: str = 'SLL_HW2'
    num_workers: int = 2
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed: int = 3407

    num_epochs: int = 90
    save_epochs: int = 10
    batch_size: int = 256
    optim: str = 'SGD'
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 1e-4

    scheduler: str = 'MultiStepLR'
    milestones = [30, 50, 70, 80]
    gamma = 0.1


@dataclass
class BaselineDataConfig:
    train_transforms = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # T.Resize((h,w))
    ])

    test_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
