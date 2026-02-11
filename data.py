from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(cfg):
    transform = transforms.ToTensor()

    train = datasets.MNIST(cfg.dataset.data_dir, train=True, download=True, transform=transform)
    val = datasets.MNIST(cfg.dataset.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val,
        batch_size=cfg.dataset.batch_size,
        num_workers=4
    )

    return train_loader, val_loader