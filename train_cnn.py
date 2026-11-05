
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
# Make experiments reproducible
torch.manual_seed(42)

# Generate 200 RGB images of size 64x64
images = torch.rand(200, 3, 64, 64) * 0.1

# Create labels: 0 = vertical, 1 = horizontal
labels = torch.tensor([0] * 100 + [1] * 100)

# Add a vertical stripe to class 0 images
images[:100, :, :, 30:34] += 0.8

# Add a horizontal stripe to class 1 images
images[100:, :, 30:34, :] += 0.8

# Keep pixel values between 0 and 1
images = torch.clamp(images, 0, 1)

# Create dataset
dataset = TensorDataset(images, labels)

# Create DataLoader
train_dataset, val_dataset = random_split(
    dataset, [160, 40]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

print("Total images:", len(dataset))
print("Number of batches:", len(train_loader))
print("Image shape:", images.shape)
print("Label shape:", labels.shape)

import torch.nn as nn

class PatternCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = PatternCNN()

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training device:", device)
print("Loss function:", criterion)
print("Optimizer:", optimizer)

print(model)

# Number of training epochs
num_epochs = 20

# Training mode
model.train()

for epoch in range(num_epochs):

    running_loss = 0.0

    for batch_images, batch_labels in train_loader:

        # Move data to GPU
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(batch_images)

        # Calculate loss
        loss = criterion(outputs, batch_labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Average loss for this epoch
    epoch_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Loss: {epoch_loss:.4f}"
    )

    # Validation
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch_images, batch_labels in val_loader:

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)

            loss = criterion(outputs, batch_labels)

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    print(f"Validation Loss: {val_loss:.4f}")

    model.train()

# Switch to evaluation mode
model.eval()


