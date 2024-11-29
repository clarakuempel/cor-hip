import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

# from connectome_to_model.model.graph import Architecture

class Connectome(pl.LightningModule):
    def __init__(self, cfg, model, temperature=0.5):
        super(Connectome, self).__init__()
        
        self.model = model
        self.learning_rate = cfg.lr
        self.temperature = temperature
        
    def forward(self, x):

        
        return self.model(x, batch=False)


    def training_step(self, batch, batch_idx):
        
        anchor, positive = batch#

        while anchor.ndim < 5:
            anchor = torch.unsqueeze(anchor, 1) 
        while positive.ndim < 5:
            positive = torch.unsqueeze(positive, 1) 

        anchor_rep = self(anchor)
        positive_rep = self(positive)

       

        loss = self.contrastive_loss(anchor_rep[0], positive_rep[0])

        return loss
    
    def contrastive_loss(self, anchor_rep, positive_rep):

        batch_size = anchor_rep.size(0)
        representations = torch.cat([anchor_rep, positive_rep], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(self.device)        
        loss = F.cross_entropy(similarity_matrix / self.temperature, labels)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)