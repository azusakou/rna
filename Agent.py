import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

import math
import os
from os import path
from tqdm import tqdm

from utils import *
from RNAGenerate import value
import pdb

def test(cfg, model, iteration, Env, test_data,track_test):
    if not path.exists(cfg.eval_dir):os.mkdir(cfg.eval_dir)
    data = list(test_data.keys())
    dirs = [f"{cfg.eval_dir}{x}/" for x in data]
    for p in dirs:
        if not path.exists(p):os.mkdir(p)
    for d, loc in zip(data, dirs):
        tbar = tqdm(test_data[d], leave=False)
        for seq_id, seq in enumerate(tbar):
            if len(list(seq)) > cfg.maxseq_len:continue
            Env.reset(seq)
            sites = Env.availableSites()
            sites = Env.shuffleSites()
            for s in sites:
                Env.current_site = s
                if Env.currentPaired():best_action = value(model, Env, cfg.paired_sites, True)
                else:best_action = value(model, Env, cfg.single_sites, True)
            Env.applyMove(best_action)
            if Env.hammingLoss() > 0 : Env.localSearch()
            if Env.reward() == 1.0 : track_test[d].add(seq_id)
            writeSummary(seq_id, iteration, Env.hammingLoss(), Env.getDesigned(), loc+"summary_"+str(iteration)+".csv")
            tbar.set_description(f"Inference on {d} iteration {iteration} seq {len(list(seq))} ")
        if cfg.debug_test: break
    return test_data,track_test
    
def train(train_loader, model, optimizer, scheduler, criterion, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #inputs = inputs.float()
        #pdb.set_trace()
        inputs, targets = inputs.float().to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.detach().item()
        loss.backward()

        if (batch_idx) % 1 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 500)
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 20 == 0: scheduler.step()
    return total_loss/(batch_idx+1)

class Agent():
    def __init__(self, cfg, model):
        self.batch_size = cfg.batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.epochs = cfg.mx_epochs
        self.optimizer = self.get_optimizer(cfg.optimizer, cfg.lr)
        self.scheduler = StepLR(self.optimizer, step_size=300, gamma=0.9)
        self.criterion = self.get_loss(cfg.criterion)
        self.regression = True
        self.bestloss = math.inf
        self.model_filename = cfg.model_file
        self.load_model()    
    
    def get_loss(self,criterion):
        if criterion == "mse":return nn.MSELoss()
        return nn.CrossEntropyLoss()

    def get_optimizer(self, optimizer, lr):
        if optimizer == "adam":return Adam(self.model.parameters(), lr = lr)
        return SGD(self.model.parameters(), lr = lr, momentum = 0.9)
    
    def load_model(self):
        if path.exists(self.model_filename):
            checkpoint = torch.load(self.model_filename)
            self.model.load_state_dict(checkpoint['model'])
            print(f"model loaded successfylly")
    
    def predict(self,x):
        self.model.eval()
        x = x.to(self.device)
        return self.model(x).item()

    def update(self, train_loader, valid_loader):
        self.bestloss = math.inf
        train_dic = {'vanilla': train,
                    }        
        self.train_loader = train_loader
        self.valid_loader = valid_loader 
        tbar = tqdm(range(self.epochs), leave=False)
        for epx in tbar:
            tloss = train_dic['vanilla'](self.train_loader, self.model, self.optimizer, self.scheduler, self.criterion, epx, self.device)
            vloss = self.val(epx)

            tbar.set_description(f"Train [Epoch_{epx + 1}][train set_{len(train_loader)}][val set_{len(valid_loader)}]")
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