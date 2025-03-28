import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define a simple neural network for MNIST classification.
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # Input layer: 28x28 pixels flattened.
        self.relu = nn.ReLU()              # Activation function.
        self.fc2 = nn.Linear(512, 256)      # Hidden layer.
        self.fc3 = nn.Linear(256, 10)       # Output layer for 10 classes.

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Hyperparameters.
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Download and prepare the MNIST dataset.
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.ToTensor()
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTNet().to(device)

# Loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass.
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{len(train_loader)}], "
                f"Loss: {running_loss / 100:.4f}"
            )
            running_loss = 0.0

# Evaluate the model.
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device
