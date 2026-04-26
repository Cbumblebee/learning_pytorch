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






