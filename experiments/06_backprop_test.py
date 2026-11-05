
import torch

# Step 1: Create a trainable weight
w = torch.tensor(5.0, requires_grad=True)

# Step 2: Calculate loss
loss = (w - 3) ** 2

print("Weight:", w.item())
print("Loss:", loss.item())

# Step 3: Perform backpropagation
loss.backward()

# Step 4: Inspect the gradient
print("Gradient:", w.grad.item())
