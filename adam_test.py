
import torch
import torch.nn as nn
torch.manual_seed(42)
# A simple neural network with one trainable parameter
model = nn.Linear(1, 1)

# Create Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Input and expected output
x = torch.tensor([[2.0]])
target = torch.tensor([[6.0]])

# Loss function
criterion = nn.MSELoss()

for epoch in range(1000):

    # Forward propagation
    prediction = model(x)

    # Calculate loss
    loss = criterion(prediction, target)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

print("Final prediction:", model(x).item())
