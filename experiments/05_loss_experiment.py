
import torch
import torch.nn as nn

# Two output scores for one image
prediction = torch.tensor([[2.0, 0.5]])

# Actual label: 0 = REAL
target = torch.tensor([0])

# Create loss function
criterion = nn.CrossEntropyLoss()

# Calculate loss
loss = criterion(prediction, target)

print("Prediction logits:", prediction)
print("Actual label:", target)
print("Loss:", loss.item())
