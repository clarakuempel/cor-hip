import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import pytorch_lightning as pl


class PreprocessedVideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset for loading preprocessed PNG images.
        Args:
            data_dir (str): Path to the root directory containing preprocessed images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform if transform else T.ToTensor()

        # Collect all image paths
        self.data = self.get_image_files()

    def get_image_files(self):
        """Retrieve all preprocessed image files."""
        image_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in sorted(files):
                if file.endswith(".png") or file.endswith(".jpg"):
                    image_files.append(os.path.join(root, file))
        return image_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Load and preprocess the image at the given index."""
        image_path = self.data[idx]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image

class PreprocessedVideoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cfg, img_size):
        """
        DataModule for loading preprocessed video frames.
        Args:
            data_dir (str): Path to the root directory containing preprocessed images.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of workers for data loading.
            img_size (tuple): Target size of the images (height, width).
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

        # Define image transformations
        self.transform = T.Compose([
            T.Resize(img_size),  # Resize to target size
            T.ToTensor(),        # Convert to PyTorch tensor
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def setup(self, stage=None):
        """
        Prepare the dataset. This method is called by Lightning during training/validation/testing setup.
        """
        self.dataset = PreprocessedVideoDataset(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Return DataLoader for testing."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../conf", config_name="config", version_base="1.1")
    def main(cfg: DictConfig):
        # Path to preprocessed images
        data_dir = cfg.dataset.output_folder

        # Initialize DataModule
        video_data_module = PreprocessedVideoDataModule(
            data_dir=data_dir,
            cfg=cfg,
            img_size=(320,240)  # Target image size
        )

        # Setup and test DataLoader
        video_data_module.setup()
        for batch in video_data_module.train_dataloader():
            print(f"Batch size: {batch.size()}")  # Example: torch.Size([32, 3, 128, 128])
            breakpoint()  # For debugging

    main()