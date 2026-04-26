"""
torch.autograd: computing gradients. supporting automatic computation of gradient for any computational graph
most frequently used algorithm is back propagation: params (model weights) are adjusted to
    the gradient of the loss function to the given parameter.
"""

import torch
# e.g.
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b # w and b are parameters
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# ------------------------------------ 1. Tensors, Functions and Computational graph ------------------------------------
# w and b are parameters which we need to optimize
# we need to compute the gradients of loss function with respect to those variables. => requires_grad
# to construct computational graph. Function class. it knows how to compute function in the
#   forward-direction and also how to compute its derivative in backward propagation
#   a reference to the backward propagation function is stored in grad_fn property of a tensor

print(f"Gradient function for z = {z.grad_fn}")
# Loss ist der Vergleich zum tatsächlichen Wert.
# sobald man das verglichen hat, macht man backward. PyTorch wandert rückwärts durch den computational graph,
# berechnet die Ableitungen (Gradienten) für jeden Parameter und speichert sie in .grad
print(f"Gradient function for loss = {loss.grad_fn}")

# ------------------------------------ 2. Computing Gradients ------------------------------------
# to optimize weights of params, we need to compute derivatives of our loss function to parameters
# we call loss.backward(), and then retrieve the values from w.grad and b.grad
loss.backward()
print(w.grad)
print(b.grad)

# We can stop tracking computations by surrounding our computation code with torch.no_grad() block:
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

"""
What are Computational Graphs:
autograd keeps a record of data (tensors) and all executed operations in a DAG consisting of Function-objects.
in the DAG, leaves are the input tensors, roots are the output tensors
By tracing this graph from roots to leaves, you can automatically compute the gradients with chain rule.

In a forward pass, autograd does two things simultaneously:
- run the requested operation to compute a resulting tensor
- maintain the operation’s gradient function in the DAG.
The backward pass kicks off when .backward() is called on the DAG root. autograd then:
- computes the gradients from each .grad_fn,
- accumulates them in the respective tensor’s .grad attribute
- using the chain rule, propagates all the way to the leaf tensors.
"""


# TODO: optional reading: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html 