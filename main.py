import torch, torchvision
from torchinfo import summary
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
import utils, engine, data_setup, model

import os
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = Path('../datasets/PizzaSushiSteak')
train_path = data_path / 'train'
test_path = data_path / 'test'

IMG_SIZE = 224

simple_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

BATCH_SIZE = 32
PATCH_SIZE = 16

train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(train_path, test_path, simple_transform, BATCH_SIZE)

model = model.ViTModel(BATCH_SIZE, IMG_SIZE, len(classes))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=0.3)
EPOCHS = 10

summary(model, (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))
results = engine.train(model, train_dataloader, test_dataloader, opt, loss_fn, EPOCHS, device)

utils.plot_loss_curves(results)
print(results)