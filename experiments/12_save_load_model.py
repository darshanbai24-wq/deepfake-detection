
import torch
import torch.nn as nn
from pathlib import Path

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define our CNN
class PatternCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
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
        return self.classifier(x)


# Create a model
model = PatternCNN().to(device)
model.eval()

# Generate sample input
sample = torch.rand(1, 3, 64, 64).to(device)

# Get prediction before saving
with torch.no_grad():
    original_output = model(sample)

# Save model weights
Path("results").mkdir(exist_ok=True)

model_path = "results/pattern_cnn.pth"
torch.save(model.state_dict(), model_path)

print("Model saved successfully!")

# Create a fresh model
loaded_model = PatternCNN().to(device)

# Load saved weights
weights = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

loaded_model.load_state_dict(weights)
loaded_model.eval()

# Get prediction after loading
with torch.no_grad():
    loaded_output = loaded_model(sample)

print("Original output:", original_output)
print("Loaded output:", loaded_output)

print(
    "Are outputs identical?",
    torch.allclose(original_output, loaded_output)
)
