
import os
import numpy as np

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from models import AudioGRUModel
from models.connectome import Connectome
from models.graph import Graph, Architecture

from data import AudioDataset
from data.video_datamodule import VideoDataModule




def is_data_processed(output_dir):
    """Check if the data is already processed by verifying the existence of .npy files."""
    return any(file.endswith(".npy") for file in os.listdir(output_dir))



@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Set up logging
    wandb.init(
        project=cfg.wandb.project,
        mode="disabled",
        config=OmegaConf.to_container(cfg, resolve=True) 
    )

    # Set up device and seed
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    pl.seed_everything(cfg.seed)


    # root_dir = cfg.dataset.input_folders_small if cfg.data_subset else cfg.dataset.input_folders
    # train_dataset = AudioDataset(cfg.dataset, root_dir)
    # train_dataset = VideoAudioModule(cfg.dataset, root_dir)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)


    data_module = VideoDataModule(
        data_dir=cfg.dataset.output_folder,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        img_size=(320, 240)
    )
    data_module.setup()

    breakpoint()
    
    # Set up model
    if cfg.model.name == "gru_audio":
        print("Selected model: GRU Audio")
        model = AudioGRUModel(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            projection_size=cfg.model.projection_size,
            learning_rate=cfg.model.lr,
            temperature=cfg.model.temperature,
        )
        checkpoint_filename = "audio_gru-{epoch:02d}-{train_loss:.2f}"


    elif cfg.model.name == "connectome":
        print("Selected model: Connectome-based Architecture")
        graph = Graph(
            '/home/ckuempel/cor-hip/utils/sample_graph_ucf_test.csv' ,
            input_nodes = [2],
            # input_nodes = [0],
            output_nodes = [2]
        )
        graph_model = Architecture(
            graph=graph,
            input_sizes=graph.find_input_sizes(),
            input_dims=graph.find_input_dims(),
            topdown=True
        ).to(device)
        model = Connectome(cfg.model, graph_model, temperature=cfg.model.temperature)
        checkpoint_filename = "connectome_model-{epoch:02d}-{train_loss:.2f}"
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")
    
    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(os.getcwd(), "checkpoints"),
        filename=checkpoint_filename,
        save_top_k=3,
        mode="min",
    )

    # Set up logger
    csv_logger = CSVLogger(save_dir=os.path.join(os.getcwd(), "logs"), name="cor-hip")

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs= cfg.max_epochs, 
        callbacks=[checkpoint_callback], 
        logger=[csv_logger],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    
    # trainer.fit(model, train_loader)
    trainer.fit(model, data_module)

    wandb.save(os.path.join(os.getcwd(), "wandb/offline-*"))



# %%
if __name__ == "__main__":
    main()
 