import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
from torchvision import models
from torchvision import datasets
from torchvision.models.googlenet import GoogLeNet_Weights
#Load data 
#Transform the data so that it works with GoogLeNet (documentation on pythorch)
transform = transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets for training and validation
mislabel_rate = 0.9
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
subset, _ = torch.utils.data.random_split(training_set, [ mislabel_rate, 1-mislabel_rate])

for idx, target in enumerate(training_set.targets[subset.indices]):
    labels = list(range(10))
    labels.remove(target.item())
    training_set.targets[subset.indices[idx]] = np.random.choice(labels)

validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
# improves data retrieval
training_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Load the (pre-trained) model
model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT) 

# Replace the final layer
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)

# Set the model to training mode and use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler to adjust the learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#Training and fine tuning the model
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

     
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Fine-tuning complete!')

torch.save(model.state_dict(), 'finetuned_mislabeled90_googlenet_fmnist.pth')
print('Model saved!')

# Evaluating the model 
# Set the model to evaluation mode
model.eval()

correct = 0
total = 0
loss = 0

with torch.no_grad():
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
        loss += criterion(outputs, labels).item() * labels.size(0)

print(f'Loss of the fine-tuned model on the test images: {loss / len(validation_loader.dataset):.4f}')
print(f'Accuracy of the fine-tuned model on the test images: {100 * correct / total:.2f}%')
