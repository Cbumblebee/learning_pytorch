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
"""
Flatten = Datenformatierung.
Linear = Mathematische Transformation (Matrix-Multiplikation).
ReLU = Aktivierung (Filtert irrelevante Signale).
Softmax = Wahrscheinlichkeits-Normalisierung.

dim=0: vertikale Achse der Matrix/Zeilen/Batch-Dimension
dim=1: horizontale Achse / spalten/Klassen-Dimension
"""
X = torch.rand(1, 28, 28, device=device) #erzeuge zufälliges Rauschbild. In der Praxis wäre das ein echtes Bild.
logits = model(X) # logits sind die zahlen, wie Wahrscheinlich ein Bild in einer Klasse (Hose, Pulli, ...) passt. mit softmax passt man die an menschlich lesbare Werte an.
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1) #argmax schaut im Tensor nach, an welcher Position/welche Klasse der höchste Wert steht.
print(f"Predicted class: {y_pred}")

# ------------------------------------ 3. model layers ------------------------------------
# sample minibatch of 3 images of size 28x28
input_image = torch.rand(3,28,28)
print(input_image.size())
# nn.Flatten-layer. This converts each 2D image into a continuous 1D array.
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
# nn.linear. The Linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
## nn.ReLU. Non-linear activations after linear transformations. to learn wider variety
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
# nn.Sequential: ordered container of modules that data passes through in order.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
# nn.softmax: logits are scaled to [0, 1], representing probabilities. Dim says in which dimension the values must sum up to 1.
softmax = nn.Softmax(dim=1) # spalten, die 10 Klassen
pred_probab = softmax(logits)

# ------------------------------------ 3. model params ------------------------------------
"""
many layers are parameterized (have associated weights and biases that are optimized during training)
Subclassing nn.module automatically tracks all fields defined inside your model object, and makes all params 
accessible using parameters() or named_parameters()
"""
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

