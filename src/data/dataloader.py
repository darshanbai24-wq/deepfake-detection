
from torch.utils.data import DataLoader

from src.data.dataset import FaceImageDataset, image_transform


def create_dataloader(
    root_dir,
    batch_size=16,
    shuffle=True,
    num_workers=2
):
    dataset = FaceImageDataset(
        root_dir=root_dir,
        transform=image_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
