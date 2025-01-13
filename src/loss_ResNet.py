# ResNet evaluation and loss visualisation

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import torch
import torch.nn as nn
import torch.optim as optim


import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms


########## Load Data

# Transform images to 224x224 and 3 channels and normalize to work with resnet
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create datasets for training & validation
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
# improves data retrieval
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

######### Load Model


learnt_model = 'finetuned_resnet18_fmnist.pth'

learnt_net = resnet18()
learnt_net.load_state_dict(torch.load(learnt_model))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learnt_net = learnt_net.to(device)


########### Evaluate Model

# Set the model to evaluation mode
learnt_net.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = learnt_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the fine-tuned model on the test images: {100 * correct / total:.2f}%')


############## Visualise Loss Landscape

def tau_2d(alpha, beta, theta_ast):
  a = alpha * theta_ast[:,None,None]
  b = beta * alpha * theta_ast[:,None,None]
  return a + b

from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

theta_ast = Params2Vec(learnt_net.parameters())

infer_net = resnet18()
theta = Params2Vec(infer_net.parameters())
infer_net = infer_net.to(device)

loss_fn = nn.CrossEntropyLoss()

x = torch.linspace(-20, 20, 20)
y = torch.linspace(-20, 20, 20)
alpha, beta = torch.meshgrid(x, y)
space = tau_2d(alpha, beta, theta_ast)

losses = torch.empty_like(space[0, :, :])

for a, _ in enumerate(x):
  print(f'a = {a}')
  for b, _ in enumerate(y):
    Vec2Params(space[:, a, b] + theta_ast, infer_net.parameters())
    for _, (data, label) in enumerate(validation_loader):
      with torch.no_grad():
        infer_net.eval()
        losses[a][b] = loss_fn(infer_net(data), label).item()


# Plot Loss
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig("./resnet_loss", format='pdf')