# ResNet finetuning on mislabeled FMNIST

# imports
import numpy as np

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

# Create datasets for training
mislabel_rate = 0.25
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
subset, _ = torch.utils.data.random_split(training_set, [ mislabel_rate, 1-mislabel_rate])

for idx, target in enumerate(training_set.targets[subset.indices]):
    labels = list(range(10))
    labels.remove(target.item())
    training_set.targets[subset.indices[idx]] = np.random.choice(labels)

# Create data loaders for our datasets; shuffle for training
# improves data retrieval
training_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)

# Class labels
classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


########## Define model

# load pretrained resnet-18
model = resnet18(weights='DEFAULT')

# replace final layer
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)

# Set the model to training mode and use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler to adjust the learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


############# finetuning the model

num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Step the scheduler after each epoch
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(training_loader):.4f}")


print('Fine-tuning complete!')

# Save the fine-tuned model
torch.save(model.state_dict(), 'finetuned_mislabeled25_resnet18_fmnist.pth')
print('Model saved!')
