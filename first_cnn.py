
import torch
import torch.nn as nn

# 1. Define our CNN
class DeepfakeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # First convolution
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Second convolution
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    # 2. Define forward propagation
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 3. Initialize the model
model = DeepfakeCNN()

# 4. Create a random batch of 8 RGB images
images = torch.randn(8, 3, 64, 64)

# 5. Pass images through the network
outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Predictions:")
print(outputs)
