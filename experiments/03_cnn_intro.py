import torch
import torch.nn.functional as F

# A simple 5x5 image
image = torch.tensor([
    [0., 0., 255., 255., 255.],
    [0., 0., 255., 255., 255.],
    [0., 0., 255., 255., 255.],
    [0., 0., 255., 255., 255.],
    [0., 0., 255., 255., 255.]
])

# A vertical edge detection filter
kernel = torch.tensor([
    [-1., 0., 1.],
    [-1., 0., 1.],
    [-1., 0., 1.]
])

# PyTorch expects [Batch, Channels, Height, Width]
image = image.reshape(1, 1, 5, 5)
kernel = kernel.reshape(1, 1, 3, 3)

# Perform convolution
output = F.conv2d(image, kernel, stride=1, padding=1)

print("Input shape:", image.shape)
print("Kernel shape:", kernel.shape)
print("Feature map:")
print(output)
print("Output shape:", output.shape)