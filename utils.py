from itertools import islice
from glob import glob
from csv import writer
import random
import math 
import torch
import torch.nn as nn
from collections import deque
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
#from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import os
import numpy as np
from config import Config

CFG = Config()

single_sites = CFG.single_sites
paired_sites = CFG.paired_sites
single_sites_ix = {ix:x for ix, x in enumerate(single_sites)}
paired_sites_ix = {ix:x for ix, x in enumerate(paired_sites)}
all_sites_ix = {ix:x for ix, x in enumerate(single_sites+paired_sites)}

EPS_START = 0.97
EPS_END = 0.1
EPS_DECAY = 10000
steps_done = 0

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ps(fname): return str(fname).replace(' ', '_')

def evalArgmax(preds):
    with torch.no_grad():
        out = nn.Softmax(dim=0)(preds)
        return torch.argmax(out).item()

def explore():
    global steps_done
    steps_done+=2
    eps_threshold = EPS_END+(EPS_START-EPS_END)*math.exp(-1.*steps_done/EPS_DECAY)
    if random.random() > eps_threshold:return False
    return True

def decayingEgreedy(preds):
    if explore(): return random.choice(list(range(len(preds))))
    return evalArgmax(preds)

def trackEvalResults():
    track_results = {
                "datasetC": set(),
                "datasetA": set(),
                "datasetB": set(),
                "datasetD": set()
                }
    return track_results

def loadTrain():
    training_data = loadCandD("train")
    training_data = sorted(training_data, key=len)
    return training_data

def loadValidation():
    eval_data = {"datasetC": loadCandD("modena"),
                 "datasetA": loadAandB("antarnav"),
                 "datasetB": loadAandB("antarnat"),
                 "datasetD": loadCandD("test")
              }
    assert(len(eval_data['datasetD'])==100 and len(eval_data['datasetB'])==83 and len(eval_data['datasetC'])==29)
    return eval_data

def loadCandD(data):
    path = f"data/{data}/*.rna"
    all_files =  glob(path)
    all_files.sort()
    targets = []
    for _ , r in enumerate(all_files):
        with open(r, "r") as myfile:
            lines = myfile.readlines()
            if not lines:continue
            assert(len(lines)==1)
            lines = lines[0].strip()
            targets.append(lines)
    return targets

def loadAandB(data):
    if data == "antarnav" : path = "./data/antarnav.txt"
    elif data == "antarnat" :path = "./data/antarnat.txt"
    targets = []
    with open(path, "r") as myfile:
        for line in myfile.readlines():
            line = line.strip()
            if line.startswith('.') or line.startswith('('):
                targets.append(line)
    return targets

def pick_samples(cfg, dataset, iteration, n=200, window_size = 3):
    dataset_size = len(dataset)
    start_point = iteration if iteration < 25 else iteration ** 2
    if start_point + window_size*n > dataset_size:
        start_point = dataset_size - window_size*n
    end_point = start_point + window_size*n
    window = dataset[start_point:end_point] 
    chosen_samples = random.sample(window, n) if iteration < int(cfg.episodes * 0.8) else random.sample(dataset, n)
    return chosen_samples

def pick_samples_c(cfg, dataset, iteration, n=200, window_size=3):
    dataset_size = len(dataset)
    start_point = iteration if iteration < 25 else iteration ** 2
    if start_point + window_size * n > dataset_size:
        start_point = dataset_size - window_size * n
    end_point = start_point + window_size * n
    window = dataset[start_point:end_point]
    
    # Calculate 80% and 20% of n
    n_90_percent = int(n * 0.9)
    n_10_percent = n - n_90_percent
    
    # If iteration is less than 80% of episodes, pick 80% from window and 20% from dataset
    if iteration < int(cfg.episodes * 0.8):
        chosen_samples_window = random.sample(window, n_90_percent)
        chosen_samples_dataset = random.sample(dataset, n_10_percent)
        chosen_samples = chosen_samples_window + chosen_samples_dataset
    else:
        chosen_samples = random.sample(dataset, n)

    return chosen_samples

def binaryCodings(seq):
    def to_bin(x):return '{0:04b}'.format(x)
    binary_codes = dict()
    for ix, x in enumerate(seq) : binary_codes[seq[ix]] = to_bin(ix)
    binary_codes['unknown_pair'] = to_bin(10)
    binary_codes['unknown_single'] = to_bin(11)
    return binary_codes

def generateW(seq, n):
    it = iter(seq) 
    result = tuple(islice(it, n))
    if len(result) == n:yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def getAllPairings(target):
    stack = []
    paired_bases= list()
    unpaired_bases = list()
    for i in range(len(target)):
        if target[i] == '(':
            stack.append(i)
        if target[i] == ')':
            paired_bases.append((stack.pop(), i))
        elif target[i]=='.':
            unpaired_bases.append(i)
    del stack
    return paired_bases, unpaired_bases

"""optimizer & scheduler"""
def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
         'lr': decoder_lr, 'weight_decay': weight_decay}
    ]

    return optimizer_parameters

def bert_base_AdamW_LLRD(model,encoder_lr, decoder_lr,):
    '''
    不同学习率
    '''
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = encoder_lr
    head_lr = decoder_lr
    lr = init_lr

    # === Pooler and regressor ======================================================

    params_0 = [p for n, p in named_parameters if ("fc" in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if ("fc" in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}
    opt_parameters.append(head_params)

    # === 12 Hidden layers ==========================================================

    for layer in range(11, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)

        lr *= 0.99  # 1

    # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
    opt_parameters.append(embed_params)

    return AdamW(opt_parameters, lr=init_lr)

def writeSummary(seq_id, iteration, score, pred, fname):
    ls = [seq_id, iteration, score, pred]
    with open(fname, 'a') as myfile:
        writer_object = writer(myfile)
        writer_object.writerow(ls)

def logSummaries(iteration, track_evaluation, evaluation_data):
    print(f"Iteration {iteration} results ... ")
    for k in list(track_evaluation.keys()):print(f"{k} - {len(track_evaluation[k])}/{len(evaluation_data[k])}")