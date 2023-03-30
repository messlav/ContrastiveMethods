from dataclasses import dataclass
import torch
import torchvision.transforms as T


@dataclass
class SimCLR_config:
    wandb_project: str = 'SLL_HW2'
    num_workers: int = 8
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed: int = 3407

    num_epochs: int = 200
    save_epochs: int = 10
    eval_epochs: int = 10
    batch_size: int = 256
    optim: str = 'Adam'
    lr: float = 3e-4
    weight_decay: float = 1e-4

    num_features: int = 512

    scheduler: str = 'CosineAnnealingLR'


class ContrastiveImages:
    def __init__(self, transform):
        self.transform = transform
        self.n_views = 2

    def __call__(self, img):
        return [self.transform(img) for i in range(self.n_views)]


s = 1.0
size = 96
color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
rnd_gray = T.RandomGrayscale(p=0.2)

train_transforms = T.Compose([
    T.RandomResizedCrop(size),
    T.RandomHorizontalFlip(p=0.5),
    rnd_color_jitter,
    rnd_gray,
    T.GaussianBlur(kernel_size=int(0.1 * 96)),
    T.ToTensor(),
])
