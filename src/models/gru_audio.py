import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb


class AudioGRUModel(pl.LightningModule):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, projection_size=64, learning_rate=1e-3, temperature=0.5):
        super(AudioGRUModel, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size) # for the last hidden state of the GRU
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size)
        )
        self.learning_rate = learning_rate
        self.temperature = temperature

    def forward(self, x):
        output, _ = self.gru(x)
        representation = self.fc(output[:, -1, :])
        projection = self.projection_head(F.normalize(representation, dim=1))
        return F.normalize(projection, dim=1)  
        # return F.normalize(representation, dim=1)  

    def training_step(self, batch, batch_idx):
        anchor, positive = batch  
        anchor_rep = self(anchor)
        positive_rep = self(positive)

        # Contrastive loss
        loss = self.contrastive_loss(anchor_rep, positive_rep)
        self.log("train_loss", loss)
        


        # Log representations to WandB 
        if batch_idx % 10 == 0:  
            self.logger.experiment.log({
                "anchor_representation": wandb.Histogram(anchor_rep.detach().cpu().numpy()),
                "positive_representation": wandb.Histogram(positive_rep.detach().cpu().numpy())
            })

        return loss



    def contrastive_loss(self, anchor_rep, positive_rep):

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer