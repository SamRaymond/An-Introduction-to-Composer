import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from composer.utils import dist
# from composer import Trainer
# from composer.models import ComposerModel
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from composer.utils import dist
import lightning as L

# Define the CNN model
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# class ComposerCNN(ComposerModel):
#     def __init__(self):
#         super().__init__()
#         self.model = CIFAR10CNN()

#     def forward(self, batch):
#         inputs, _ = batch
#         return self.model(inputs)

#     def loss(self, outputs, batch):
#         _, targets = batch
#         return F.cross_entropy(outputs, targets) #<-- we add the loss as a functional rather than a class
# define the LightningModule
class PTL_CNN(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

dist.initialize_dist('gpu')

# Set random seed for reproducibility
torch.manual_seed(42)

# Define transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
if dist.get_local_rank() == 0:
   trainset = torchvision.datasets.CIFAR10(root='/tmp/my_data', train=True, download=True, transform=transform)
   testset = torchvision.datasets.CIFAR10(root='/tmp/my_data', train=False, download=True, transform=transform)
dist.barrier()

if not dist.get_local_rank() == 0:
    trainset = torchvision.datasets.CIFAR10(root='/tmp/my_data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='/tmp/my_data', train=False, download=False, transform=transform)


# Create distributed samplers
train_sampler = DistributedSampler(trainset, shuffle=True)
test_sampler = DistributedSampler(testset, shuffle=True)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, sampler=test_sampler)

# Define the model, loss function, and optimizer
# model = ComposerCNN()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

model = PTL_CNN(CIFAR10CNN())
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = L.Trainer(
    max_epochs=num_epochs,
    devices=8, #how many GPUs to use per node
    num_nodes=2,
    accelerator="gpu",
    )
trainer.fit(model=model, train_dataloaders=trainloader)

# trainer = Trainer(
#     model=model,
#     train_dataloader=trainloader,
#     optimizers=optimizer,
#     max_duration=10,  # epochs
#     device='gpu'
# )
# trainer.fit()
