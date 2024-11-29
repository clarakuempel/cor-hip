import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
# import wandb
# wandb.init(mode="offline")


class AudioGRUModel(pl.LightningModule):
    """
    AudioGRUModel is a PyTorch Lightning Module that defines a GRU model for audio data."""
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, projection_size=64, learning_rate=1e-3, temperature=0.5):
        # TOOD: put temp
        super(AudioGRUModel, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)    # GRU layer
        self.fc = nn.Linear(hidden_size, hidden_size)                               # for the last hidden state of the GRU
        self.projection_head = nn.Sequential(                                       # projection head  (to lower-dimensional space)            
            nn.Linear(hidden_size, projection_size),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size)
        )
        self.learning_rate = learning_rate
        self.temperature = temperature

    def forward(self, x):
        """Forward pass of the model."""
        output, _ = self.gru(x)                                                 # pass through the GRU
        representation = self.fc(output[:, -1, :])                              # get the last hidden state
        projection = self.projection_head(F.normalize(representation, dim=1))   # normalize and project
        return F.normalize(projection, dim=1)                                   # normalize the projection

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        anchor, positive = batch  
        anchor_rep = self(anchor)
        positive_rep = self(positive)

        loss = self.contrastive_loss(anchor_rep, positive_rep)
        self.log("train_loss", loss)
        


        # Log representations to WandB 
        if batch_idx % 10 == 0:  
            # self.logger.experiment.log({
            #     "anchor_representation": wandb.Histogram(anchor_rep.detach().cpu().numpy()),
            #     "positive_representation": wandb.Histogram(positive_rep.detach().cpu().numpy())
            # })

            anchor_mean = anchor_rep.detach().cpu().mean().item()
            anchor_std = anchor_rep.detach().cpu().std().item()
            positive_mean = positive_rep.detach().cpu().mean().item()
            positive_std = positive_rep.detach().cpu().std().item()

            # Log to CSV
            self.log("anchor_representation_mean", anchor_mean)
            self.log("anchor_representation_std", anchor_std)
            self.log("positive_representation_mean", positive_mean)
            self.log("positive_representation_std", positive_std)

        return loss



    def contrastive_loss(self, anchor_rep, positive_rep):
        """Calculate contrastive loss."""

        batch_size = anchor_rep.size(0)
        representations = torch.cat([anchor_rep, positive_rep], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        # labels = torch.arange(batch_size).to(self.device)
        # labels = torch.cat([labels, labels], dim=0)

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(self.device)        
        loss = F.cross_entropy(similarity_matrix / self.temperature, labels)
        return loss


    def configure_optimizers(self):
        """Configure the optimizer for the model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer