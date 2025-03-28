import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define a neural network for MNIST classification with configurable hidden layer sizes.
class MNISTNet(nn.Module):
    def __init__(self, hidden_size1=512, hidden_size2=256):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size1)  # First hidden layer.
        self.relu = nn.ReLU()                         # Activation function.
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second hidden layer.
        self.fc3 = nn.Linear(hidden_size2, 10)          # Output layer for 10 classes.

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def main(args):
    # Download and prepare the MNIST dataset.
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet(hidden_size1=args.hidden_size1, hidden_size2=args.hidden_size2).to(device)

    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop.
    for epoch in range(args.num_epochs):
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
                    f"Epoch [{epoch + 1}/{args.num_epochs}], "
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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total:.2f}%")
    print("ok")

if __name__ == '__main__':
    # Create the parser and add hyperparameter arguments.
    parser = argparse.ArgumentParser(description="MNIST Neural Network Training")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--hidden_size1', type=int, default=512,
                        help='number of neurons in the first hidden layer (default: 512)')
    parser.add_argument('--hidden_size2', type=int, default=256,
                        help='number of neurons in the second hidden layer (default: 256)')
    args = parser.parse_args()

    main(args)
