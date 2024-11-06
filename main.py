
import os

import numpy as np
import torch
from torch import device, nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Connectome


import wandb
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min'
    )

    unique_id = uuid.uuid4()
    run_name = cfg.model.name + "_epoch:" + str(cfg.epochs) + "_" + str(unique_id)
    print(run_name)

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    trainer = Trainer(
        max_epochs=cfg.epochs,
        callbacks=[checkpoint_callback],
        limit_val_batches=200,
        limit_test_batches=40,        
        accumulate_grad_batches=4,
    )

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print(f"Number of devices: {trainer.num_devices}")
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    


    model = Connectome(**cfg.model)
        datasets = {
            "train": ,
            "val": 
        }




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
 