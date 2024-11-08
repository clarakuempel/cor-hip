
import os

import numpy as np
import torch
from torch import device, nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint




import hydra
from omegaconf import DictConfig, OmegaConf


from torch.utils.data import DataLoader, Dataset
from models import AudioGRUModel
from data import AudioDataset



@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    input_size = cfg.model.input_size    
    hidden_size = cfg.model.hidden_size    
    num_layers = cfg.model.num_layers    
    learning_rate = cfg.model.lr
    batch_size = cfg.batch_size     
    max_epochs = cfg.max_epochs

    data_dir = cfg.dataset.output_folder_audio
    train_dataset = AudioDataset(data_dir, target_length=64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AudioGRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints",
        filename="audio_gru-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        config= {**cfg.model, **cfg.dataset, **cfg.wandb.config}
    )

    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback], logger=wandb_logger)

    



    trainer.fit(model, train_loader)


    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print("Device:", device)
    # print(f"Number of devices: {trainer.num_devices}")
    # pl.seed_everything(cfg.seed)
    # torch.backends.cudnn.determinstic = True
    # torch.backends.cudnn.benchmark = False

    


    # model = Connectome(**cfg.model)
    #     datasets = {
    #         "train": ,
    #         "val": 
    #     }




    ## if checkpoint, load from checkpoint, otherwise train


    ## TODO: load args / cfg


    # TODO: set seed for reproducibilty

    ## load data / DataLoader Python
    # train, test, ....

    ## build connectome graph
    ## build NN from graph

    ## set up training loop: Readout, optimizer, criterion, losses, 
    ## load model evtl
    ## for epoch: .... (save losses)

    # 




# %%
if __name__ == "__main__":
    main()
 