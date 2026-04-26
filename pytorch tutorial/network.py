import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
neural networks are layers that perform operations on data
The torch.nn namespace provides all the building blocks you need to build your own neural network.
Every module in PyTorch subclasses the nn.Module.
A neural network is a module itself that consists of other modules (layers). This nested structure allows for building and managing complex architectures easily.
"""

# In the following sections, we’ll build a neural network to classify images in the FashionMNIST dataset.
# ------------------------------------ 1. get device ------------------------------------
# We want to be able to train our model on an accelerator such as CUDA, MPS, MTIA, or XPU.
# If the current accelerator is available, we will use it. Otherwise, we use the CPU.
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
# my output: using cpu device

# ------------------------------------ 2. define class ------------------------------------
# by subclassing nn.Module and initialize the neural network layers in __init__.
# Every nn.Module subclass implements the operations on input data in the "forward" method.
"""
Flatten = Datenformatierung.
Linear = Mathematische Transformation (Matrix-Multiplikation).
ReLU = Aktivierung (Filtert irrelevante Signale).
Softmax = Wahrscheinlichkeits-Normalisierung.

dim=0: vertikale Achse der Matrix/Zeilen/Batch-Dimension
dim=1: horizontale Achse / spalten/Klassen-Dimension
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x) # flatten makes sth 2D -> 1D vector
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# to use the model, we pass it the input data. This exec. the forward-method, along with some bg-operations.
# calling the model on the input returns a 2d tensor with dim=0 for each output of 10 raw predicted values for each class
# dim=1 for corresponding to the individual values of each out





