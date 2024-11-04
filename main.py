
import os

import numpy as np
import torch
from torch import device, nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


import wandb
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


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
 