import numpy as np
import copy 

import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.ticker import LinearLocator

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
from torchvision import models
from torchvision import datasets
from torchvision.models.googlenet import GoogLeNet_Weights
import loss_landscapes

#Load data 
#Transform the data so that it works with GoogLeNet (documentation on pythorch)
transform = transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets for training & validation
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
# improves data retrieval
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size= 4, shuffle=False)

learnt_model = 'models/finetuned_mislabeled90_googlenet_fmnist.pth'
learnt_net = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

in_features = learnt_net.fc.in_features
learnt_net.fc = nn.Linear(in_features, 10)
learnt_net.load_state_dict(torch.load(learnt_model, map_location=torch.device('cpu'), weights_only=True))

device = torch.device("cpu")
learnt_net = learnt_net.to(device)

############## Visualise Loss Landscape

# data that the evaluator will use when evaluating loss

STEPS = 30
distance = 2e6
loss_fn = nn.CrossEntropyLoss()

x, y = iter(validation_loader).__next__()
metric = loss_landscapes.metrics.Loss(loss_fn, x, y)

model_final = copy.deepcopy(learnt_net)

loss_data_fin = loss_landscapes.random_plane(model_final, metric, distance, STEPS, normalization='filter', deepcopy_model=True)

fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.set_title('Surface Plot of Loss Landscape')

# Customize the z axis.
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_zticks([])

plt.suptitle("Loss Landscape", y=1, fontsize=18)
plt.title("Mislabel rate = 90% \n Model accuracy = 0.1224", fontsize=10)
ax.set_xlabel('$Alpha$', rotation=150)
ax.set_ylabel('$Beta$')
ax.set_zlabel('$Loss$', rotation=60)


plt.savefig("images/googlenet90_loss.pdf", format='pdf')
