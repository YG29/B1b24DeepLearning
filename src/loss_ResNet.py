# ResNet evaluation and loss visualisation
# Adapted from: https://tvsujal.medium.com/visualising-the-loss-landscape-3a7bfa1c6fdf

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
in_features = learnt_net.fc.in_features
learnt_net.fc = nn.Linear(in_features, 10)
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

# Define transformation tau for 2D contour map
def tau_2d(alpha, beta, theta_ast):
  a = alpha * theta_ast[:,None,None]
  b = beta * alpha * theta_ast[:,None,None]
  return a + b

from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

# Convert optimal parameters to vector
theta_ast = Params2Vec(learnt_net.parameters())

# Define a secondary network for inference and save its initial parameters to vector
infer_net = resnet18()
in_features = infer_net.fc.in_features
infer_net.fc = nn.Linear(in_features, 10)
theta = Params2Vec(infer_net.parameters())
infer_net = infer_net.to(device)

loss_fn = nn.CrossEntropyLoss()

# Define grid over which to evaluate loss function along two random directions alpha and beta in parameter space
x = torch.linspace(-2, 2, 10)
y = torch.linspace(-2, 2, 10)
alpha, beta = torch.meshgrid(x, y)
alpha, beta = alpha.to(device), beta.to(device)
space = tau_2d(alpha, beta, theta_ast)

losses = torch.empty_like(space[0, :, :])

# Evaluate loss function over the given grid along alpha and beta
for a, _ in enumerate(x):
  print(f'a = {a}')
  for b, _ in enumerate(y):
    Vec2Params(space[:, a, b] + theta_ast, infer_net.parameters())
    for images, labels in validation_loader:
      images, labels = images.to(device), labels.to(device)
      with torch.no_grad():
        infer_net.eval()
        losses[a][b] = loss_fn(infer_net(images), labels).item()


# Plot Loss
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
mean = torch.mean(losses, (0,1))
std = torch.std(losses)
mean, std = mean.cpu(), std.cpu()
print(f"Mean = {mean}, std = {std}")
print("Pre-Normalization losses:")
print(losses)

# Normalize loss for sake of plotting
# z-score normalization
losses = (losses - mean) / std
## min-max normalization
# losses = (losses - torch.min(losses)) / (torch.max(losses) - torch.min(losses))
surf = ax.plot_surface(alpha.cpu(), beta.cpu(), losses.cpu(), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(2 * torch.min(losses).cpu(), 2 * torch.max(losses).cpu())
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_zticks([])

plt.suptitle("Loss Landscape", y=1, fontsize=18)
plt.title("Mislabel rate = 0%", fontsize=10)
ax.set_xlabel('$Alpha$', rotation=150)
ax.set_ylabel('$Beta$')
ax.set_zlabel('$Loss$', rotation=60)


plt.savefig("./resnet_loss.pdf", format='pdf')