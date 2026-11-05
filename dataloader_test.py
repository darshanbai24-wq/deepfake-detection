
import torch
from torch.utils.data import TensorDataset, DataLoader

# Create 20 sample inputs
images = torch.randn(20, 3, 64, 64)

# Create 20 labels (0 = REAL, 1 = FAKE)
labels = torch.randint(0, 2, (20,))

# Combine images and labels into a dataset
dataset = TensorDataset(images, labels)

# Divide dataset into batches of 4
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Total images:", len(dataset))

for batch_number, (batch_images, batch_labels) in enumerate(loader):

    print("\nBatch:", batch_number + 1)
    print("Image shape:", batch_images.shape)
    print("Label shape:", batch_labels.shape)
