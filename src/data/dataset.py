
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class FaceImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []

        # REAL = 0, FAKE = 1
        class_labels = {
            "real": 0,
            "fake": 1
        }

        # Read images from both class folders
        for class_name, label in class_labels.items():

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing folder: {class_dir}"
                )

            for image_path in class_dir.iterdir():

                if image_path.suffix.lower() in [
                    ".jpg", ".jpeg", ".png"
                ]:
                    self.samples.append(
                        (image_path, label)
                    )

        if not self.samples:
            raise ValueError("No images found!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        # Open image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing
        if self.transform:
            image = self.transform(image)

        return image, label


# Preprocessing for CNN input
image_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])
