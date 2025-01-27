# ResNet evaluation and loss visualisation

import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import torch
import torch.nn as nn
import torch.optim as optim


import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms

import loss_landscapes


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

learnt_model = 'models/finetuned_mislabeled90_resnet18_fmnist.pth'

learnt_net = resnet18()
in_features = learnt_net.fc.in_features
learnt_net.fc = nn.Linear(in_features, 10)
learnt_net.load_state_dict(torch.load(learnt_model, map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learnt_net = learnt_net.to(device)

############## Visualise Loss Landscape

# Parameters for loss visualization
STEPS = 30
distance = 2e6
loss_fn = nn.CrossEntropyLoss()

# Set loss visualization metric
x, y = iter(validation_loader).__next__()
metric = loss_landscapes.metrics.Loss(loss_fn, x, y)

model_final = copy.deepcopy(learnt_net)

# Evaluate loss landscape plane using above parameters
loss_data_fin = loss_landscapes.random_plane(model_final, metric, distance, STEPS, normalization='filter', deepcopy_model=True)

# Plot the above plane as 3D surface plot
fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

# Customize the z axis.
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_zticks([])

plt.suptitle("Loss Landscape", y=1, fontsize=18)
plt.title("Mislabel rate = 90% \n Model accuracy = 0.1100", fontsize=10)
ax.set_xlabel('$Alpha$', rotation=150)
ax.set_ylabel('$Beta$')
ax.set_zlabel('$Loss$', rotation=60)


plt.savefig("./images/resnettest_loss.pdf", format='pdf')