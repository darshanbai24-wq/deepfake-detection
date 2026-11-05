
import torch

w = torch.tensor(5.0, requires_grad=True)

learning_rate = 0.1
epoch = 0

while True:
    epoch += 1

    # Calculate loss
    loss = (w - 3) ** 2

    # Calculate gradient
    loss.backward()

    # Update weight
    with torch.no_grad():
        w -= learning_rate * w.grad

    # Clear old gradient
    w.grad.zero_()

    print(f"Epoch {epoch}: Weight = {w.item():.4f}, Loss = {loss.item():.4f}")

    if loss.item() < 0.0001:
        break
