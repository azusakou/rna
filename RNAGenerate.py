import numpy as np
#from RNADataLoader import CustomDataset
from torch.utils.data import Dataset, DataLoader
from utils import *
from os import path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pdb

def value(model, Env, actions, eval_mode = False):
    '''
    this function uses in generate training data and evaluation, and save to Env.coded_state_value
    return the best action
    '''
    global steps_done
    status = True if len(list(actions[0])) == 2 else False
    action_value, best_action_data  = [], []
    #pdb.set_trace()
    current_state_value = Env.designedSequence()#;print(len(current_state_value),current_state_value)
    for a in actions:
        steps_done+=1
        input_sample = Env.code(a)
        input_tensor = torch.tensor(input_sample)
        input_tensor = input_tensor.view(1, 1, input_tensor.size(0),input_tensor.size(1))
        pred_hat = model.predict(input_tensor)
        action_value.append(pred_hat)
        best_action_data.append(input_sample)
        Env.undoMove(current_state_value)
    assert(current_state_value==Env.designedSequence())
    action_value = torch.tensor(action_value).view(-1, 1)
    action = evalArgmax(action_value) if eval_mode else decayingEgreedy(action_value)
    best_action = paired_sites_ix[action] if status else single_sites_ix[action] 
    best_action_data = torch.tensor(best_action_data[action])
    if not eval_mode: Env.coded_state_value.append(best_action_data)
    return best_action

def trainOrtest(cfg):
    if random.random() < float(cfg.test_size) : return True
    return False

def labelStates0(cfg, Env, reward, positive_train, negative_train, positive_test, negative_test):
    '''
    generate features and labels
    '''
    if reward==1:
        for input_tensor in Env.coded_state_value:
            if not trainOrtest(cfg):
                if len(positive_train) >= cfg.replay_size:positive_train.popleft()
                positive_train.append([input_tensor, torch.tensor([1.0])])
            else:
                if len(positive_test) >= cfg.replay_size:positive_test.popleft()
                positive_test.append([input_tensor, torch.tensor([1.0])])
    else:
        for input_tensor in Env.coded_state_value:
                if not trainOrtest(cfg):
                    if len(negative_train) >= cfg.replay_size:negative_train.popleft()
                    negative_train.append([input_tensor, torch.tensor([-1.0])])
                else:
                    if len(negative_test) >= cfg.replay_size:negative_test.popleft()
                    negative_test.append([input_tensor, torch.tensor([-1.0])])  
    return positive_train, negative_train, positive_test, negative_test

def labelStates(cfg, Env, reward, positive_train, negative_train, positive_test, negative_test):
    '''
    generate features and labels
    '''
    if reward==1:
        if not trainOrtest(cfg):
            for input_tensor in Env.coded_state_value:
                if len(positive_train) >= cfg.replay_size:positive_train.popleft()
                positive_train.append([input_tensor, torch.tensor([1.0])])
        else:
            for input_tensor in Env.coded_state_value:
                if len(positive_test) >= cfg.replay_size:positive_test.popleft()
                positive_test.append([input_tensor, torch.tensor([1.0])])
    else:
        if not trainOrtest(cfg):
            for input_tensor in Env.coded_state_value:
                if len(negative_train) >= cfg.replay_size:negative_train.popleft()
                negative_train.append([input_tensor, torch.tensor([-1.0])])
        else:
            for input_tensor in Env.coded_state_value:
                if len(negative_test) >= cfg.replay_size:negative_test.popleft()
                negative_test.append([input_tensor, torch.tensor([-1.0])])  
    return positive_train, negative_train, positive_test, negative_test

def sampleReplay(positive, negative, sample_sz):
    if len(positive) >= sample_sz:
        neg_sample_train = random.sample(negative, sample_sz)
        pos_sample_train = random.sample(positive, sample_sz)
    elif len(positive) == 0: #if there is no positive yet
        sample_sz = 5
        neg_sample_train = random.sample(negative, sample_sz)
        pos_sample_train = []
    else:
        sample_sz = len(positive)
        neg_sample_train = random.sample(negative, sample_sz)
        pos_sample_train = random.sample(positive, sample_sz)
    
    X , y = [], []
    for d in pos_sample_train+neg_sample_train:
        X.append(d[0])
        y.append(d[1])        

    X = pad_sequence(X, padding_value=-1) # batching the workload
    X = X.view(X.size(1),1,X.size(0),X.size(2))
    y = torch.cat(y).view(-1, 1)
    assert(X.size(0)==y.size(0))
    return X, y

class CustomDataset(Dataset):
    def __init__(self, X , y):
       self.X = X
       self.y = y
    def __len__(self):
        y = self.y
        return y.size(0)
    def __getitem__(self, idx):                 
        return self.X[idx],self.y[idx]
    
def load_data(cfg, positive_train, negative_train, positive_test, negative_test):
    xtrain, ytrain = sampleReplay(positive_train, negative_train, cfg.sample_train)
    xtest, ytest =  sampleReplay(positive_test, negative_test, cfg.sample_test)
    train_loader = DataLoader(CustomDataset(xtrain, ytrain), batch_size = cfg.batch_size, num_workers = cfg.num_workers)
    valid_loader = DataLoader(CustomDataset(xtest, ytest), batch_size = cfg.batch_size, num_workers = cfg.num_workers)
    return train_loader, valid_loader

def BatchPlayout(cfg, B, positive_train, negative_train, positive_test, negative_test, Env, model):#batch playout function
    if not path.exists(cfg.train_dir):os.mkdir(cfg.train_dir)
    tbar = tqdm(B, leave=False)
    for seq_id, seq in enumerate(tbar):
        if len(list(seq)) > cfg.maxseq_len:continue
        Env.reset(seq)
        sites = Env.availableSites()
        sites = Env.shuffleSites()
        for s in sites:
            Env.current_site = s
            if Env.currentPaired():best_action = value(model, Env, cfg.paired_sites)
            else:best_action = value(model, Env, cfg.single_sites)
            Env.applyMove(best_action)
        if Env.hammingLoss() > 0 : Env.localSearch()
        positive_train, negative_train, positive_test, negative_test = labelStates(cfg, Env, Env.reward(), 
                                                                                   positive_train, negative_train, positive_test, negative_test)

        tbar.set_description(f"BatchPlayout {seq_id}/{len(B)} with pos {len(positive_train)}/ neg {len(negative_train)}")
    train_loader, valid_loader = load_data(cfg, positive_train, negative_train, positive_test, negative_test)
    return train_loader, valid_loader, positive_train, negative_train, positive_test, negative_test

def process_sequence(seq, cfg, positive_train, negative_train, positive_test, negative_test, Env, model):
    if len(list(seq)) > cfg.maxseq_len:
        return positive_train, negative_train, positive_test, negative_test
    Env.reset(seq)
    sites = Env.availableSites()
    sites = Env.shuffleSites()
    for s in sites:
        Env.current_site = s
        if Env.currentPaired():
            best_action = value(model, Env, cfg.paired_sites)
        else:
            best_action = value(model, Env, cfg.single_sites)
        Env.applyMove(best_action)
    if Env.hammingLoss() > 0:
        Env.localSearch()
    return labelStates(cfg, Env, Env.reward(), positive_train, negative_train, positive_test, negative_test)

def Playout(cfg, B, positive_train, negative_train, positive_test, negative_test, Env, model):
    if not os.path.exists(cfg.train_dir):
        os.mkdir(cfg.train_dir)
    tbar = tqdm(range(len(B)), leave=False)
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_sequence, seq, cfg, positive_train, negative_train, positive_test, negative_test, Env, model) for seq in B]
        seq_id = 0
        for future in as_completed(futures):
            seq_id += 1
            tmp_positive_train, tmp_negative_train, tmp_positive_test, tmp_negative_test = future.result()
            positive_train.extend(tmp_positive_train)
            negative_train.extend(tmp_negative_train)
            positive_test.extend(tmp_positive_test)
            negative_test.extend(tmp_negative_test)
            tbar.set_description(f"BatchPlayout {seq_id}/{len(B)} with pos {len(positive_train)}/ neg {len(negative_train)}")
            tbar.update(1)

    train_loader, valid_loader = load_data(cfg, positive_train, negative_train, positive_test, negative_test)
    return train_loader, valid_loader, positive_train, negative_train, positive_test, negative_test


import torch.multiprocessing as mp

def worker(seq, cfg, positive_train, negative_train, positive_test, negative_test, Env, model, queue):
    result = process_sequence(seq, cfg, positive_train, negative_train, positive_test, negative_test, Env, model)
    queue.put(result)

def Playout_torch(cfg, B, positive_train, negative_train, positive_test, negative_test, Env, model):
    if not os.path.exists(cfg.train_dir):
        os.mkdir(cfg.train_dir)

    processes = []
    queue = mp.Queue()
    for seq in B:
        p = mp.Process(target=worker, args=(seq, cfg, positive_train, negative_train, positive_test, negative_test, Env, model, queue))
        p.start()
        processes.append(p)

    seq_id = 0
    with tqdm(total=len(B), leave=False) as tbar:
        while seq_id < len(B):
            tmp_positive_train, tmp_negative_train, tmp_positive_test, tmp_negative_test = queue.get()
            positive_train.extend(tmp_positive_train)
            negative_train.extend(tmp_negative_train)
            positive_test.extend(tmp_positive_test)
            negative_test.extend(tmp_negative_test)
            seq_id += 1
            tbar.set_description(f"BatchPlayout {seq_id}/{len(B)} with pos {len(positive_train)}/ neg {len(negative_train)}")

    for p in processes:
        p.join()

    train_loader, valid_loader = load_data(cfg, positive_train, negative_train, positive_test, negative_test)
    return train_loader, valid_loader, positive_train, negative_train, positive_test, negative_test

