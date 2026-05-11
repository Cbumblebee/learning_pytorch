import torch
import torchvision.models as models

# ------------------------------------ 1. Saving and Loading Model Weights ------------------------------------
"""
they store params in an internal state dictionary called state_dict
persistent via torch.save

In einfachen Schritten:
Modell-Instanz erstellen: Du baust erst das leere "Skelett" (deine NeuralNetwork-Klasse).

Gewichte laden: Dann schraubst du die gespeicherten Parameter mit load_state_dict() in dieses Skelett.
weights_only=True sagt Python: "Lade nur reine Zahlenwerte (Gewichte) und führe absolut keinen anderen (schädlichen) Programmcode aus."
"""
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
# ------------------------------------ 2. Saving and Loading Models with Shapes ------------------------------------
"""
We might want to save the structure of this class together with the model, 
in which case we can pass model (and not model.state_dict()) to the saving function:

saving state_dict is considered the best practice
"""
torch.save(model, 'model.pth')
model = torch.load('model.pth', weights_only=False)


"""What is state_dict?
Link[https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict]
learnable parameters (weights and biases) of a Module-class are contained in the model.parameters(). 
A state_dict is a Python dictionary that maps each layer to its parameter tensor.

- A common PyTorch convention is to save models using either a .pt or .pth file extension.
- Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
    Failing to do this will yield inconsistent inference results.
- model_state = model.state_dict() returns a reference to the state and not its copy! 
- You must serialize best_model_state or use best_model_state = deepcopy(model.state_dict()) 
    otherwise your best best_model_state will keep getting updated by the subsequent training iterations.
Save:
torch.save(model.state_dict(), PATH)
Load:
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()
"""





