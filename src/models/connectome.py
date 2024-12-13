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

        return self.model(x, batch=True)
    
    def contrastive_loss(self, anchor, positive, temperature):
        """
        Compute contrastive loss using the InfoNCE loss function.
        Args:
            anchor: Tensor of embeddings (shape: [batch_size, embed_dim]).
            positive: Tensor of embeddings for positive samples (shape: [batch_size, embed_dim]).
            temperature: Scaling factor for logits.
        Returns:
            Scalar loss value.
        """
        
        anchor = anchor[0]
        positive = positive[0]

        batch_size = anchor.shape[0]


        anchor_flat = anchor.view(anchor.shape[0], -1)
        positive_aligned_flat = positive.view(positive.shape[0], -1)

        anchor_norm = F.normalize(anchor_flat, dim=1)
        positive_aligned_norm = F.normalize(positive_aligned_flat, dim=1) 

        cosine_similarity = torch.sum(anchor_norm * positive_aligned_norm, dim=1) 
        labels = torch.arange(batch_size).float().to(anchor.device)
       
        loss = F.cross_entropy(cosine_similarity, labels)
        return loss


    def training_step(self, batch, batch_idx):

        """
        Perform a single training step.
        Args:
            batch: A tuple containing input frames and the target frame.
            batch_idx: The index of the current batch.
        Returns:
            Loss value for the batch.
        """
        input_frames, target_frame = batch

        # Compute embeddings for input sequence and target frame
        input_embeddings = self(input_frames)  # Shape: [batch_size, embed_dim]
        target_embeddings = self(target_frame.unsqueeze(1))  # Shape: [batch_size, embed_dim]
        # Compute contrastive loss
        loss = self.contrastive_loss(input_embeddings, target_embeddings, self.temperature)

        # Log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)