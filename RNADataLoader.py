import numpy as np
#from RNADataLoader import CustomDataset
from torch.utils.data import Dataset, DataLoader
from utils import *
from os import path
import os
from tqdm import tqdm

def value(model, state, actions, eval = False):
    '''
    this function uses in generate training data and evaluation, and save to state.coded_state_value
    return the best action
    '''
    global steps_done
    status = True if len(list(actions[0])) == 2 else False
    action_value, best_action_data  = [], []
    current_state_value = state.designedSequence()
    for a in actions:
        steps_done+=1
        input_sample = state.code(a)
        input_tensor = torch.tensor(input_sample)
        input_tensor = input_tensor.view(1, 1, input_tensor.size(0),input_tensor.size(1))
        action_value.append(model.predict(input_tensor))
        best_action_data.append(input_sample)
        state.undoMove(current_state_value)
    assert(current_state_value==state.designedSequence())
    action_value = torch.tensor(action_value).view(-1, 1)  
    action = evalArgmax(action_value) if eval else decayingEgreedy(action_value)
    best_action = paired_sites_ix[action] if status else single_sites_ix[action] 
    best_action_data = torch.tensor(best_action_data[action])
    if not eval: state.coded_state_value.append(best_action_data)
    return best_action

def trainOrtest(args):
    if random.random() < float(args.test_size) : return True
    return False

def labelStates(args, state, reward, positive_train, negative_train, positive_test, negative_test):
    '''
    generate features and labels
    '''
    if reward==1:
        for input_tensor in state.coded_state_value:
            if not trainOrtest(args):
                if len(positive_train) >= args.replay_size:positive_train.popleft()
                positive_train.append([input_tensor, torch.tensor([1.0])])
            else:
                if len(positive_test) >= args.replay_size:positive_test.popleft()
                positive_test.append([input_tensor, torch.tensor([1.0])])
    else:
        for input_tensor in state.coded_state_value:
                if not trainOrtest(args):
                    if len(negative_train) >= args.replay_size:negative_train.popleft()
                    negative_train.append([input_tensor, torch.tensor([-1.0])])
                else:
                    if len(negative_test) >= args.replay_size:negative_test.popleft()
                    negative_test.append([input_tensor, torch.tensor([-1.0])])  
    return positive_train, negative_train, positive_test, negative_test

def sampleReplay(positive, negative, sample_sz):
    if len(positive) >= sample_sz:
        neg_sample_train = random.sample(negative, sample_sz)
        pos_sample_train = random.sample(positive, sample_sz)
    elif len(positive) == 0: #if there is no positive yet
        sample_sz = 10
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
    
def load_data(args, positive_train, negative_train, positive_test, negative_test):
    xtrain, ytrain = sampleReplay(positive_train, negative_train, args.sample_train)
    xtest, ytest =  sampleReplay(positive_test, negative_test, args.sample_test)
    train_loader = DataLoader(CustomDataset(xtrain, ytrain), batch_size = args.batch_size, num_workers = args.num_workers)
    valid_loader = DataLoader(CustomDataset(xtest, ytest), batch_size = args.batch_size, num_workers = args.num_workers)
    return train_loader, valid_loader

def SPRNABatchPlayout(args, B, P, U, positive_train, negative_train, positive_test, negative_test, RNAEnv, model):#batch playout function
    print(f"Batch Sequence Folding ... ")
    tbar = tqdm(B)
    for seq_id, seq in enumerate(tbar):
        if len(list(seq)) > args.maxseq_len:continue
        state = RNAEnv(seq, args.W, P, U)
        sites = state.availableSites()
        sites = state.shuffleSites()
        for s in sites:
            state.current_site = s
            if state.currentPaired():best_action = value(model, state, P)
            else:best_action = value(model, state, U)
            state.applyMove(best_action)
        if state.hammingLoss() > 0 : state.localSearch()
        positive_train, negative_train, positive_test, negative_test = labelStates(args, state, state.reward(), 
                                                                                   positive_train, negative_train, positive_test, negative_test)
        if not path.exists(args.train_dir):os.mkdir(args.train_dir)
        train_loader, valid_loader = load_data(args, positive_train, negative_train, positive_test, negative_test)
        tbar.set_description(f"BatchPlayout {seq_id}/{len(B)} with pos {len(positive_train)}/ neg {len(negative_train)}")
    return train_loader, valid_loader, positive_train, negative_train, positive_test, negative_test
