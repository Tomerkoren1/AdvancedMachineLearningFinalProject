
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torchmetrics import F1Score
import copy
import wandb

class tBertTrainer():
  def __init__(self, model, epochs = 5, lr=3e-3, use_wandb=True):
    
    self.model = model
    self.best_val_model = None
    self.epochs = epochs
    
    self.loss = nn.BCELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)

    # Metrics
    self.f1 = F1Score(num_classes=1)
    self.loss_train = []
    self.loss_val = []
    self.f1s = []

    self.use_wandb = use_wandb
    if use_wandb:
      self.epoch_samples_number = 0
      

  def fit(self, train_dataloader, val_dataloader):

    # put model in train mode
    min_valid_loss = np.inf
    
    # run on all epochs
    for e in range(self.epochs):
      print(f"Epoch: {e}")

      # train
      self.train_epoch(train_dataloader)

      # eval      
      self.validate_epoch(val_dataloader)

      # save the optimal model    
      if min_valid_loss > self.loss_val[-1]:
          self.best_val_model = copy.deepcopy(self.model)
          min_valid_loss = self.loss_val[-1]

      if self.use_wandb:
          wandb.log({"epoch_train_loss": self.loss_train[-1], "epoch_val_loss": self.loss_val[-1], "epoch_val_F1": self.f1s[-1], "epoch":e})

      print(f"Train Loss: {self.loss_train[-1]}, Val Loss: {self.loss_val[-1]}, Val F1: {self.f1s[-1]}")

  def train_epoch(self,train_dataloader):
    
    self.model.train(True)
    train_loss = 0
    num_batches = len(train_dataloader)
    for data in tqdm(train_dataloader):
      inputs1, inputs2, labels = data

      # Clear the gradients
      self.optimizer.zero_grad()

      # Forword
      target = self.model([inputs1, inputs2])
        
      # Calculate Loss
      labels = labels.type(torch.FloatTensor).unsqueeze(1).cuda()
      loss = self.loss(target, labels)

      # Calculate gradients 
      loss.backward()

      # Update Weights
      self.optimizer.step()

      # Calculate Losss
      train_loss += loss.item()

      if self.use_wandb:
          wandb.log({"batch_train_loss": loss.item()},step=self.epoch_samples_number)
          self.epoch_samples_number += 1

      # Delete resources
      del loss
      del labels

    self.loss_train.append(train_loss / num_batches)
    torch.cuda.empty_cache()

  def validate_epoch(self, val_dataloader):
    self.model.train(False)
    val_loss = 0
    num_batches = len(val_dataloader)
    labels_lst, preds_lst = [], []
    with torch.no_grad():
      for data in val_dataloader:
        inputs1, inputs2, labels = data
        
        # Forword
        target = self.model([inputs1, inputs2])
        
        # Calculate loss
        labels = labels.type(torch.FloatTensor).unsqueeze(1).cuda()
        loss = self.loss(target,labels)
        val_loss += loss.item()
        
        # Store results
        preds_lst.extend(target.cpu().detach()[:,0].tolist())
        labels_lst.extend(labels.tolist())

        del loss
        del labels

    self.loss_val.append(val_loss / num_batches)
    f1_score =  self.f1(torch.tensor(np.round(preds_lst)), torch.tensor(labels_lst).type(torch.IntTensor)).item()
    self.f1s.append(f1_score)

  def test(self, test_dataloader):
    self.model.train(False)
    labels, preds = [], []

    with torch.no_grad():
      for data in test_dataloader:
        inputs1, inputs2, label = data

        # Forword
        target = self.best_val_model([inputs1, inputs2])

        # Store results
        preds.extend(target.cpu().detach()[:,0].tolist())
        labels.extend(label.tolist())
    
    f1_score =  self.f1(torch.tensor(np.round(preds)), torch.tensor(labels)).item()
    print(f"Test F1: {f1_score}")
    return f1_score