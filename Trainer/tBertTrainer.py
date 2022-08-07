
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torchmetrics import F1Score
import copy

class tBertTrainer():
  def __init__(self, model, epochs = 5, lr=3e-3):
    
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

  def fit(self, train_dataloader, val_dataloader):

    # put model in train mode
    self.model.train()
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

      print(f"Train Loss: {self.loss_train[-1]}, Val Loss: {self.loss_val[-1]}, Val F1: {self.f1s[-1]}")
    self.model = self.best_val_model

  def train_epoch(self,train_dataloader):
    
    self.model.train()
    train_loss = 0
    num_batches = len(train_dataloader)
    for data in tqdm(train_dataloader):
      inputs1, inputs2, labels = data
      inputs = [inputs1, inputs2]

      # Clear the gradients
      self.optimizer.zero_grad()

      # Forword
      target = self.model(inputs)
        
      # Calculate Loss
      labels = labels.type(torch.FloatTensor).unsqueeze(1).cuda()
      loss = self.loss(target,labels)

      # Calculate gradients 
      loss.backward()

      # Update Weights
      self.optimizer.step()

      # Calculate Loss
      train_loss += loss.item()

    self.loss_train.append(train_loss / num_batches)

  def validate_epoch(self, val_dataloader):
    self.model.eval()
    val_loss, f1s = 0, 0
    num_batches = len(val_dataloader)
    for data in val_dataloader:
      inputs1, inputs2, labels = data
      
      # Forword
      target = self.model([inputs1, inputs2])
      
      # Calculate loss
      labels = labels.type(torch.FloatTensor).unsqueeze(1).cuda()
      loss = self.loss(target,labels)
      val_loss += loss.item()

      # F1 score
      f1_score = self.f1(target.round().cpu().detach()[:,0], labels.type(torch.IntTensor).cpu().detach()[:,0]).item()
      f1s += f1_score

    self.loss_val.append(val_loss / num_batches)
    self.f1s.append(f1s / num_batches)

  def test(self, test_dataloader):
    self.model.eval() 
    labels = []
    preds = []

    with torch.no_grad():
      for data in test_dataloader:
        inputs1, inputs2, label = data

        # Forword
        target = self.best_val_model([inputs1, inputs2])
        target = target.round()

        # Stroe results
        preds.extend(target.cpu().detach()[:,0].tolist())
        labels.extend(label.tolist())
    
    f1_score =  self.f1(torch.tensor(preds), torch.tensor(labels)).item()
    return f1_score