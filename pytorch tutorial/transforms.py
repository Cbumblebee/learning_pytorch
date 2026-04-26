
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

"""
We use transforms to make data suitable for learning.
All TorchVision datasets have two parameters:
 -transform to modify the features
 - target_transform to modify the labels
 - these accept callables containing the transformation logic

The FashionMNIST features are in PIL Image format, and the labels are integers.
For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.
To make these transformations, we use ToTensor and Lambda.
"""
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # totensor makes a PIL image or numpy ndarray into a FloatTensor and scales image's pixel intensity values in the range [0., 1.]
    transform=ToTensor(),
    # lamda transforms here into a zero tensor of size 10 = num labels of the dataset and
    # calls scatter_ which assigns a value = 1 on the index given by label y
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)