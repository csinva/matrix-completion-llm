from lightning.pytorch.utilities.types import OptimizerLRScheduler
import transformers
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import lightning.pytorch as pl


class TabEmbeddings(torch.nn.Module):
    def __init__(self, n_embed, use_pos_embeddings=True):
        '''
        Original values will be linearly projected to multiple dimensions
        1 emb dim will represent nan_mask
        If use_pos_embeddings is True,
            2 emb dims will represent positional embeddings (row/col indexes)
        '''
        super().__init__()
        self.use_pos_embeddings = use_pos_embeddings
        if use_pos_embeddings:
            self.val_embeddings = torch.nn.Linear(1, n_embed - 3)
        else:
            self.val_embeddings = torch.nn.Linear(1, n_embed - 1)
        self.layer_norm = torch.nn.LayerNorm(
            n_embed, eps=1e-12, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

    def forward(self, x: torch.Tensor, nan_mask: torch.Tensor, n_rows, n_columns):
        '''
        Params
        ------
        x: torch.Tensor
            input tensor of shape (batch_size, seq_len)
            seq_len is the flattened matrix
        '''
        # val_embeddings are (batch_size, seq_len, n_embed)
        embeddings = self.val_embeddings(x.unsqueeze(-1))

        # append nan_mask (batch_size, seq_len) to embeddings, resulting in (batch_size, seq_len, n_embed + 1)
        embeddings = torch.cat([embeddings, nan_mask.unsqueeze(-1)], dim=-1)

        # append positional embeddings (batch_size, seq_len)
        if self.use_pos_embeddings:
            # one embedding dimension represents the column number, another dimension represents the row
            col_tensor = torch.tile(torch.Tensor(
                np.arange(n_columns)), (n_rows, 1))
            col_tensor = ((col_tensor - col_tensor.mean()) /
                          col_tensor.std())
            self.col_tensor = col_tensor.flatten()  # (seq_len)
            row_tensor = torch.tile(torch.Tensor(
                np.arange(n_rows)), (n_columns, 1)).T
            row_tensor = ((row_tensor - row_tensor.mean()) /
                          row_tensor.std())
            self.row_tensor = row_tensor.flatten()  # (seq_len)
            col_tensor = self.col_tensor.repeat(x.shape[0], 1).to(x.device)
            row_tensor = self.row_tensor.repeat(x.shape[0], 1).to(x.device)
            embeddings = torch.cat(
                [embeddings, col_tensor.unsqueeze(-1)], dim=-1)
            embeddings = torch.cat(
                [embeddings, row_tensor.unsqueeze(-1)], dim=-1)

        # normalize
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# class TabSelfAttention(torch.nn.Module):
#     def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
#         super().__init__()
#         self.mha = torch.nn.MultiheadAttention(
#             embed_dim=n_embed, num_heads=n_heads, dropout=dropout, batch_first=True)
#         self.proj = torch.nn.Linear(n_embed, n_embed)
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, x, att_mask):
#         context, _ = self.mha(x, x, x, key_padding_mask=att_mask)
#         proj = self.proj(context)
#         out = self.dropout(proj)
#         return out

class TabAttentionHead(torch.nn.Module):
    def __init__(self, head_size, dropout=0.1, n_embed=3):
        super().__init__()

        self.query = torch.nn.Linear(
            in_features=n_embed, out_features=head_size)
        self.key = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.values = torch.nn.Linear(
            in_features=n_embed, out_features=head_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, att_mask):
        '''
        Params
        ------
        x: torch.Tensor
            input tensor of shape (batch_size, seq_len, n_embed)
        '''
        batch_size, seq_len, n_embed = x.shape

        # these are all (batch_size, seq_len, head_size)
        q = self.query(x)
        k = self.key(x)
        v = self.values(x)

        # this becomes (batch_size, seq_len, seq_len)
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(n_embed)
        # mask out not attended tokens
        weights = weights.masked_fill(att_mask == 0, -1e9)

        # this is (batch_size, seq_len, seq_len)
        scores = F.softmax(weights, dim=-1)
        scores = self.dropout(scores)

        # final output is (batch_size, seq_len, head_size)
        out = scores @ v
        return out


class TabSelfAttention(torch.nn.Module):
    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()
        head_size = n_embed // n_heads
        n_heads = n_heads
        if not head_size * n_heads == n_embed:
            raise ValueError('n_embed should be divisible by n_heads')

        self.heads = torch.nn.ModuleList(
            [TabAttentionHead(head_size, dropout, n_embed) for _ in range(n_heads)])

        # project from multiple heads to the embedding space
        self.proj = torch.nn.Linear(head_size * n_heads, n_embed)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, att_mask):
        v = torch.cat([head(x, att_mask) for head in self.heads], dim=-1)
        proj = self.proj(v)
        out = self.dropout(proj)
        return out


class MLP(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=3):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.GELU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class TabLayer(torch.nn.Module):
    """Single layer of tabular self-attention
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()
        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.self_attention = TabSelfAttention(n_heads, dropout, n_embed)

        self.layer_norm2 = torch.nn.LayerNorm(n_embed)
        self.mlp = MLP(dropout, n_embed)

    def forward(self, x, att_mask):
        x = self.layer_norm1(x)
        x = x + self.self_attention(x, att_mask)

        x = self.layer_norm2(x)
        out = x + self.mlp(x)

        return out


class TabLLM(L.LightningModule):
    """BERT-style encoder
    """

    def __init__(self,
                 n_layers=2, n_heads=3, dropout=0.1, n_embed=10, use_pos_embeddings=True,

                 # training args
                 learning_rate=1e-3,
                 n_rows_list=[],
                 n_columns_list=[]

                 ):
        """
        Params
        ------
        n_layers: int
            number of BERT layer in the model (default=2)
        n_heads: int    
            number of heads in the MultiHeaded Self Attention Mechanism (default=1)
        dropout: float
            hidden dropout of the BERT model (default=0.1)
        n_embed: int
            hidden embeddings dimensionality (default=3)
        use_pos_embeddings: bool
            whether to use positional embeddings (default=True)
        """
        super().__init__()

        self.embedding = TabEmbeddings(n_embed, use_pos_embeddings)

        self.tab_layers = torch.nn.ModuleList(
            [TabLayer(n_heads, dropout, n_embed) for _ in range(n_layers)])

        self.unembedding = torch.nn.Linear(in_features=n_embed, out_features=1)

        # training args
        self.learning_rate = learning_rate
        self.n_rows_list = n_rows_list
        self.n_columns_list = n_columns_list

    def forward(self, x: torch.Tensor, nan_mask: torch.Tensor, att_mask: torch.Tensor,
                n_rows: int, n_columns: int):
        '''
        Params
        ------
        x: torch.Tensor
            input tensor of shape (batch_size, seq_len)
            seq_len is the flattened matrix
        nan_mask: torch.Tensor
            nan_mask is the same shape as x,
            but with 1s where x is nan and 0s elsewhere
        att_mask: torch.Tensor
            attention mask is the same shape as x,
            but with 1s where x should be attended and 0 elsewhere
        n_rows: int
            number of rows in the input matrix
        n_columns: int
            number of columns in the input matrix
        '''

        # embeddings become (batch_size, seq_len, n_embed)
        x = self.embedding(x, nan_mask, n_rows, n_columns)

        # apply each encoding layer
        for layer in self.tab_layers:
            x = layer(x, att_mask)

        # project embedding size to scalar matrix
        mat = self.unembedding(x).squeeze(-1)
        return mat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss'}}

    def training_step(self, batch, batch_idx):
        (x_nan_t, x_clean_t, nan_mask_t, att_mask_t) = batch
        pred = self.forward(x_nan_t, nan_mask_t, att_mask_t,
                            n_rows=max(self.n_rows_list), n_columns=max(self.n_columns_list))
        train_loss = F.mse_loss(
            x_clean_t[nan_mask_t.bool()], pred[nan_mask_t.bool()], reduction='mean')
        self.log('loss', train_loss, prog_bar=True,
                 on_epoch=True, on_step=False, sync_dist=True)
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        (x_nan_t, x_clean_t, nan_mask_t, att_mask_t) = batch
        pred = self.forward(x_nan_t, nan_mask_t, att_mask_t,
                            n_rows=max(self.n_rows_list), n_columns=max(self.n_columns_list))
        val_loss = (
            torch.abs(x_clean_t[nan_mask_t.bool()] -
                      pred[nan_mask_t.bool()])
        ).sum().item() / nan_mask_t.sum().item()
        self.log('val_loss', val_loss, prog_bar=True,
                 on_epoch=True, on_step=False, sync_dist=True)
        return {'val_loss': val_loss}

    def on_validation_epoch_end(self):
        print('')
