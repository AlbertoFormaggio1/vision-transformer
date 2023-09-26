import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_path: str,
                       test_path: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = NUM_WORKERS):

    # Importing the datasets with imageFolder
    train_ds = datasets.ImageFolder(train_path, transform=transform)
    test_ds = datasets.ImageFolder(test_path, transform=transform)

    # Creating the dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=True)

    classes = train_ds.classes

    return train_dataloader, test_dataloader, classes