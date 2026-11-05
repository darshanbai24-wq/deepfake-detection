import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

z = x @ y

print("Device:", z.device)
print("Result shape:", z.shape)