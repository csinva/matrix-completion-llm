import transformers
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TabEmbeddings(torch.nn.Module):
    def __init__(self, n_embed):  # , max_seq_len=16):
        '''
        Original values will be linearly projected to multiple dimensions
        One embedding dimension will represent nan_mask,
        Two embedding dimensions will represent positional embeddings (row/col indexes)
        '''
        super().__init__()
        self.val_embeddings = torch.nn.Linear(1, n_embed - 3)

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
        # one embedding dimension will represent the column number
        # one embedding dimension will represent the row number
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
        embeddings = torch.cat([embeddings, col_tensor.unsqueeze(-1)], dim=-1)
        embeddings = torch.cat([embeddings, row_tensor.unsqueeze(-1)], dim=-1)

        # normalize
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TabAttentionHead(torch.nn.Module):
    """
    A single attention head in MultiHeaded Self Attention layer.
    The idea is identical to the original paper ("Attention is all you need"),
    however instead of implementing multiple heads to be evaluated in parallel we matrix multiplication,
    separated in a distinct class for easier and clearer interpretability
    """

    def __init__(self, head_size, dropout=0.1, n_embed=3):
        super().__init__()

        self.query = torch.nn.Linear(
            in_features=n_embed, out_features=head_size)
        self.key = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.values = torch.nn.Linear(
            in_features=n_embed, out_features=head_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, att_mask):
        # B, Seq_len, N_embed
        B, seq_len, n_embed = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.values(x)

        weights = (q @ k.transpose(-2, -1)) / \
            math.sqrt(n_embed)  # (B, Seq_len, Seq_len)
        # mask out not attended tokens
        weights = weights.masked_fill(att_mask == 0, -1e9)

        scores = F.softmax(weights, dim=-1)
        scores = self.dropout(scores)

        context = scores @ v

        return context


class TabSelfAttention(torch.nn.Module):
    """
    MultiHeaded Self-Attention mechanism as described in "Attention is all you need"
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        head_size = n_embed // n_heads

        n_heads = n_heads

        self.heads = torch.nn.ModuleList(
            [TabAttentionHead(head_size, dropout, n_embed) for _ in range(n_heads)])

        # project from multiple heads to the single space
        self.proj = torch.nn.Linear(head_size * n_heads, n_embed)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, att_mask):
        context = torch.cat([head(x, att_mask) for head in self.heads], dim=-1)

        proj = self.proj(context)

        out = self.dropout(proj)

        return out


class FeedForward(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=3):
        super().__init__()

        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.GELU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.ffwd(x)

        return out


class TabLayer(torch.nn.Module):
    """
    Single layer of BERT transformer model
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()
        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.self_attention = TabSelfAttention(n_heads, dropout, n_embed)

        self.layer_norm2 = torch.nn.LayerNorm(n_embed)
        self.feed_forward = FeedForward(dropout, n_embed)

    def forward(self, x, att_mask):
        x = self.layer_norm1(x)
        x = x + self.self_attention(x, att_mask)

        x = self.layer_norm2(x)
        out = x + self.feed_forward(x)

        return out


class TabEncoder(torch.nn.Module):
    def __init__(self, n_layers=2, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [TabLayer(n_heads, dropout, n_embed) for _ in range(n_layers)])

    def forward(self, x, att_mask):
        for layer in self.layers:
            x = layer(x, att_mask)

        return x


class TabLLM(torch.nn.Module):
    """
    BERT-style encoder
    """

    def __init__(self, n_layers=2, n_heads=1, dropout=0.1, n_embed=3):
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
        """
        super().__init__()

        self.embedding = TabEmbeddings(n_embed)

        self.encoder = TabEncoder(n_layers, n_heads, dropout, n_embed)

        self.predictor = torch.nn.Linear(in_features=n_embed, out_features=1)

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
            attention mask for padded tokens
            (batch_size, seq_len, seq_len)
        n_rows: int
            number of rows in the input matrix
        n_columns: int
            number of columns in the input matrix
        '''

        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        att_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)

        # embeddings become (batch_size, seq_len, n_embed)
        embeddings = self.embedding(x, nan_mask, n_rows, n_columns)

        encoded = self.encoder(embeddings, att_mask)

        predictions = self.predictor(encoded).squeeze(-1)
        return predictions
