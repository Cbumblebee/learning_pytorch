# learning_pytorch
for my data mining seminar
I started with this and read until data loaders: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
But I decided to go with the official documentation: https://docs.pytorch.org/tutorials/beginner/basics/intro.html

# Note: This is for arch linux, but it should work on other OS (Mac I don't know)
To get the same requirements as me in a venv, use ``pip install -r requirements.txt``
# setting up my venv + installing pytorch
1. python -m venv .venv
2. source .venv/bin/activate
3. pip install torch
Installiere `python3-pip` falls nicht vorhanden, aber nutze es nur innerhalb von Umgebungen: pacman -S python-pip

# to get into your venv again in VS Code
just go directly in the right folder

# version
by executing the following script in pytorch, you will find its version:
```
import torch
torch.__version__
```

# If you have NVIDIA/CUDA-Support:
After installing PyTorch, you can check whether your installation recognizes your built-in NVIDIA GPU by running the following code in ```python
import torch torch.cuda.is_available() This returns:
True
```

