import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('TkAgg') # or 'Qt5Agg' if PyQt5 is installed
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import DataLoader
""" 
 dataset code should be decoupled from model training code
 two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset
 they allow to use pre-loaded datasets as well as own data
 "Dataset" stores samples and labels
 "dataloader" wraps an iterable around the Dataset for easy access
"""

# ------------------------------------ 1. loading a dataset ------------------------------------
"""
 Fashion-MNIST dataset from TorchVision
 dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. 
 Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

 We load the FashionMNIST Dataset with the following parameters:
    root is the path where the train/test data is stored,
    train specifies training or test dataset,
    download=True downloads the data from the internet if it’s not available at root.
    transform and target_transform specify the feature and label transformations
    
    es lädt nicht alle 60k Bilder in den RAM, sondern merkt sich nur, wo sie liegen. 
    Mit training_data[i] kann man dann darauf zugreifen.
    Das Bild wird dann vom Memory/Cache geladen, die Transformation "ToTensor" angewendet, und dann als Tupel (Bild, Label) zurückgegeben.
"""
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor() # das richtige Format für PyTorch!
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# ------------------------------------ 2. Iterating and Visualizing the Dataset ------------------------------------
# We can index Datasets manually like a list: training_data[index].
# We use matplotlib to visualize some samples in our training data.
# Die Daten sind bereits mit Labeln verknüpft. Wir bilden sie mit labels_map wieder auf ihre Namen zurück.
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))  # width and height
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray") # Dimension (1, 28, 28) -> (28, 28), die 1 steht für den Channel; cmap=gray steht für Graustufenbild.
#plt.show()
plt.savefig("plot.png")

# ------------------------------------ 3. Creating a Custom Dataset for your files ------------------------------------
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__
class CustomImageDataset(Dataset):
    # We initialize the directory containing the images, the annotations file, and both transforms
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # returns the number of samples in our dataset
    def __len__(self):
        return len(self.img_labels)

    #  loads and returns a sample from the dataset at the given index idx
    # with the idx, it identifies the image's location on disk, converts that to a tensor using decode_image,
    # retrieves the corresponding label from the csv data in self.img_labels,
    # calls the transform functions on them and returns the tensor image and corresponding label in a tuple
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
# ------------------------------------ 4. Preparing your data for training with DataLoaders ------------------------------------
# we want pass samples in batches, reshuffle the data and use pythons multiprocessing to speed up data retrieval
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# ------------------------------------ 5.  Iterate through the DataLoader ------------------------------------
# each iteration returns a batch of train_features and train_labels
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
















