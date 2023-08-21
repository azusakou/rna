import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import math
import os
from os.path import exists
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
class Model0():
    def __init__(self,
    model,
    epochs,
    optimizer,
    criterion,
    model_filename,
    batch_size,
    lr
    ):
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.epochs = epochs
        self.optimizer = self.get_optimizer(optimizer, lr)
        self.criterion = self.get_loss(criterion)
        self.regression = True
        self.bestloss = math.inf
        self.model_filename = model_filename    
        self.load_model()    

    def get_loss(self,criterion):
        if criterion == "mse":return nn.MSELoss()
        return nn.CrossEntropyLoss()

    def get_optimizer(self, optimizer, lr):
        if optimizer == "adam":return Adam(self.model.parameters(), lr = lr)
        return SGD(self.model.parameters(), lr = lr, momentum = 0.9)
    
    def load_model(self):
        if exists(self.model_filename):
            checkpoint = torch.load(self.model_filename)
            self.model.load_state_dict(checkpoint['model'])
            print(f"model loaded successfylly")

    def trainer(self, train_loader, valid_loader):    
        self.train_loader = train_loader
        self.valid_loader = valid_loader 
        for epx in range(self.epochs):
            self.train(epx)    
            self.test(epx)
            
    
    def train(self, epoch):
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.float()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()    
        
    def test(self,epoch):
        self.model.eval()
        for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
            inputs = inputs.float()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = loss.detach().item()
            self.bestloss = loss
            state = {
                'model': self.model.state_dict(),
                'loss': self.bestloss,
            }
            torch.save(state, self.model_filename)
        if epoch%20==0:print(f"End of Epoch {epoch} loss {loss} best {self.bestloss}")
    
    def predict(self,x):
        self.model.eval()
        x = x.to(self.device)
        return self.model(x).item()
    
def train(train_loader, model, optimizer, scheduler, criterion, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #inputs = inputs.float()
        inputs, targets = inputs.float().to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.detach().item()
        loss.backward()

        if (batch_idx) % 1 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 500)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return total_loss/(batch_idx+1)


class Model():
    def __init__(self,
    model,
    epochs,
    optimizer,
    criterion,
    model_filename,
    batch_size,
    lr
    ):
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.epochs = epochs
        self.optimizer = self.get_optimizer(optimizer, lr)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.criterion = self.get_loss(criterion)
        self.regression = True
        self.bestloss = math.inf
        self.model_filename = model_filename    
        self.load_model()    
    
    def get_loss(self,criterion):
        if criterion == "mse":return nn.MSELoss()
        return nn.CrossEntropyLoss()

    def get_optimizer(self, optimizer, lr):
        if optimizer == "adam":return Adam(self.model.parameters(), lr = lr)
        return SGD(self.model.parameters(), lr = lr, momentum = 0.9)
    
    def load_model(self):
        if exists(self.model_filename):
            checkpoint = torch.load(self.model_filename)
            self.model.load_state_dict(checkpoint['model'])
            print(f"model loaded successfylly")

    def trainer(self, train_loader, valid_loader):
        train_dic = {'vanilla': train,
                    }        
        self.train_loader = train_loader
        self.valid_loader = valid_loader 
        tbar = tqdm(range(self.epochs), leave=False)
        for epx in tbar:
            tloss = train_dic['vanilla'](self.train_loader, self.model, self.optimizer, self.scheduler, self.criterion, epx, self.device)
            vloss = self.val(epx)

            tbar.set_description(f"Epoch_[{epx + 1}]_[{len(train_loader)}]_[VAL best loss {self.bestloss}]")
            tbar.set_postfix({'train_Loss': '{0:1.4f}'.format(tloss),
                              'val_Loss': '{0:1.4f}'.format(vloss),
                              'LR': '{0:1.8f}'.format(self.scheduler.get_last_lr()[0]),
                              })

    def val(self,epoch):
        self.model.eval()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
            inputs = inputs.float()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = loss.detach().item()
            total_loss += loss
            self.bestloss = loss if loss < self.bestloss else self.bestloss
            if epoch+1 == self.epochs:
                state = {
                    'model': self.model.state_dict(),
                    'loss': self.bestloss,
                }
                torch.save(state, self.model_filename)
   
        return total_loss/(batch_idx+1)
    
    def predict(self,x):
        self.model.eval()
        x = x.to(self.device)
        return self.model(x).item()
