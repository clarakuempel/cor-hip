
import os

import numpy as np
import torch
from torch import device, nn

import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint




import hydra
from omegaconf import DictConfig, OmegaConf


from torch.utils.data import DataLoader, Dataset
from models import AudioGRUModel
from data import AudioDataset



def is_data_processed(output_dir):
    """Check if the data is already processed by verifying the existence of .npy files."""
    return any(file.endswith(".npy") for file in os.listdir(output_dir))





@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    input_size = cfg.model.input_size    
    hidden_size = cfg.model.hidden_size    
    num_layers = cfg.model.num_layers    
    learning_rate = cfg.model.lr
    batch_size = cfg.batch_size     
    max_epochs = cfg.max_epochs


    root_dir = cfg.dataset.input_folders_small if cfg.data_subset else cfg.dataset.input_folders


    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print(f"Number of devices: {trainer.num_devices}")
    pl.seed_everything(cfg.seed)
    # torch.backends.cudnn.determinstic = True
    # torch.backends.cudnn.benchmark = False

    train_dataset = AudioDataset(cfg.dataset, root_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # TODO: set up connectome model here
    model = AudioGRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate)
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints",
        filename="audio_gru-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # wandb_logger = WandbLogger(
    #     project=cfg.wandb.project,
    #     config= {**cfg.model, **cfg.dataset, **cfg.wandb.config}
    # )
    csv_logger = CSVLogger("logs", name="cor-hip")


    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        callbacks=[checkpoint_callback], 
        logger=csv_logger,
        devices=1,
        accelerator="gpu"
        )

    trainer.fit(model, train_loader)





# %%
if __name__ == "__main__":
    main()
 