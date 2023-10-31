import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MatrixFactorization(torch.nn.Module):

    def __init__(self,n_embed,n,m):
        """
        Params 
        -------
        n_embed: int 
        embedding dimension of latent factors 
        n: int 
        number of rows 
        m : int
        number of cols
        """
        super().__init__()
        self.n_embed = n_embed
        self.n = n
        self.m = m
        self.P = nn.Embedding(n,n_embed)
        self.Q = nn.Embedding(m,n_embed)
        self.P_bias = nn.Embedding(n,1)
        self.Q_bias = nn.Embedding(m,1)

    def forward(self,entry):
        """
        entry: torch.Tensor 
        imput tensor of shape (batch_size,2)
        """
        row_idx = entry[:,0] 
        col_idx = entry[:,1] 
        p_row = self.P(row_idx) #get embeddings corresponding to rows in batch
        q_col = self.Q(col_idx) #get embeddings corresponding to batch
        row_bias =  self.P_bias(row_idx)
        col_bias = self.Q_bias(col_idx)
        outputs = (p_row * q_col).sum(axis=1) + np.squeeze(row_bias) + np.squeeze(row_bias)
        return outputs.flatten()


class MatrixCompletionData(Dataset):
    def __init__(self,X):
        """
        X: np.array of size (n,m)
        
        """
        super().__init__()
        self.X = X
        self.non_empty_idxs = np.argwhere(~np.isnan(X))
    
    def __len__(self):
        return len(self.non_empty_idxs)
    
    def __getitem__(self,idx):
        row_idx,col_idx = self.non_empty_idxs[idx]
        label = self.X[row_idx,col_idx]
        return torch.tensor(self.non_empty_idxs[idx]),torch.tensor(label)
        

class MatrixFactorizationImputer(torch.nn.Module):
    def __init__(self,n_embed,batch_size = 1,learning_rate = 1e-3,weight_decay = 1e-3):

        self.n_embed = n_embed
        
        #training args 
        self.learning_rate = learning_rate 
        self.weight_decay = weight_decay
        self.batch_size = batch_size 

    def fit_one_epoch(self,mf,data_iterator,optimizer,loss_criterion):
        mf.train()
        epoch_loss = 0
        for (x,y) in data_iterator:
            optimizer.zero_grad()
            y_pred = mf.forward(x)
            loss = loss_criterion(y_pred.double(),y.double())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(data_iterator)

    def fit_transform(self,X,num_epochs = 5):
        n,m = X.shape
        mf = MatrixFactorization(self.n_embed,n,m)
        optimizer = optim.SGD(mf.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        mf_dataset = MatrixCompletionData(X)
        data_iterator = DataLoader(mf_dataset,shuffle = True,batch_size = self.batch_size)
        loss_criterion = nn.MSELoss()
        for epoch_idx in range(num_epochs):
            train_loss = self.fit_one_epoch(mf, data_iterator, optimizer, loss_criterion)
        #impute matrix 
        mf.eval()
        empty_idxs = np.argwhere(np.isnan(X))
        imputation_preds = mf.forward(torch.tensor(empty_idxs))
        for i,idx in enumerate(empty_idxs):
            X[idx[0]][idx[1]] = imputation_preds[i]
        return X
        
        

        


if __name__=="__main__": 
    m = MatrixFactorization(3,10,10)
    X = np.array([[np.nan, 2], [6, np.nan], [np.nan, 6]])

    imp = MatrixFactorizationImputer(n_embed = 3)
    imp.fit_transform(X)

    #print(m.forward(non_empty_idxs))
    #print(torch.tensor([1,2]))

