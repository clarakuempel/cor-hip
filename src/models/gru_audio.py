import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb


class AudioGRUModel(pl.LightningModule):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, learning_rate=1e-3, temperature=0.5):
        super(AudioGRUModel, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.learning_rate = learning_rate
        self.temperature = temperature

    def forward(self, x):

        output, _ = self.gru(x)
        representation = self.fc(output[:, -1, :])
        return F.normalize(representation, dim=1)  

    def training_step(self, batch, batch_idx):
        anchor, positive = batch  
        anchor_rep = self(anchor)
        positive_rep = self(positive)

        # Contrastive loss
        loss = self.contrastive_loss(anchor_rep, positive_rep)
        self.log("train_loss", loss)
        


        # Log representations to WandB (optional)
        if batch_idx % 10 == 0:  # Log every 10 steps
            self.logger.experiment.log({
                "anchor_representation": wandb.Histogram(anchor_rep.detach().cpu().numpy()),
                "positive_representation": wandb.Histogram(positive_rep.detach().cpu().numpy())
            })

        return loss



    def contrastive_loss(self, anchor_rep, positive_rep):

        batch_size = anchor_rep.size(0)
        representations = torch.cat([anchor_rep, positive_rep], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)


        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)


        loss = F.cross_entropy(similarity_matrix / self.temperature, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer