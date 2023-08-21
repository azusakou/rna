import argparse
import random
import os
from os import path
from utils import *
from copy import deepcopy
from trainer import *
from NNet import RnaNet
from TrainModel import Model
from RNAGymEnv import RNAEnv
from RNADataLoader import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--root_path', help = "root dir", type = str, default = "./sprna/")
parser.add_argument('-f', '--model_file', type = str, help = "path to value network", default = "./sprna/model.ckpt")
parser.add_argument('-e', '--episodes', type = int, help = "number of episodes to play", default = 500)
parser.add_argument('-b', '--batch_size', type = int, help = "batch size", default = 64)
parser.add_argument('-x', '--mx_epochs', type = int, help = "Max epochs to play", default = 30)
parser.add_argument('-w', '--num_workers', type = int, help = "number of workers", default = 1)
parser.add_argument('-a', '--replay_size', type = int, help = "replay memeory size", default = 100000)
parser.add_argument('-s', '--test_size', type = float, help = "test size per cent", default = 0.25)
parser.add_argument('-r', '--W', type = int, help = "the feature parameter W", default = 8)
parser.add_argument('-k', '--sample_train', type = int, help = "replay sample size to train on", default = 20000)
parser.add_argument('-z', '--sample_test', type = int, help = "replay sample size to test on", default = 20000)
parser.add_argument('-l', '--batch_playout_size', type = int, help = "batch size for the playout", default = 2)
parser.add_argument('-d', '--train_dir', type = str, help = "results directory", default = './sprna/train/')
parser.add_argument('-i', '--eval_dir', type = str, help = "eval results directory", default = './sprna/eval/')
parser.add_argument('-t', '--log_train', type = bool, help = "log train results", default = True)
parser.add_argument('-o', '--optimizer', type = str, help = "optimizer", default = "adam")
parser.add_argument('-c', '--loss', type = str, help = "loss", default = "mse")
parser.add_argument('-g', '--lr', type = float, help = "learnig rate", default = 0.001)
parser.add_argument('-n', '--maxseq_len', type = int, help = "max seq len to be  designed", default = 400)

args = parser.parse_args()


if __name__ == "__main__":
    if not path.exists(args.root_path):os.mkdir(args.root_path)
    seed_everything(seed=42)

    model = Model(RnaNet(), 
              args.mx_epochs, 
              args.optimizer,
              args.loss,
              args.model_file,
              args.batch_size,
              args.lr
              )
    track_evaluation = trackEvalResults()
    evaluation_data = loadValidation()
    print(track_evaluation.keys(), evaluation_data.keys())
    training_data = loadCandD("train")

    P = ["GC","CG","AU","UA","UG","GU"] # paired sites
    U = ["G","A","U","C"] #unpaired sites

    pos_train, neg_train, pos_test, neg_test = deque([], maxlen = args.replay_size), deque([], maxlen = args.replay_size), deque([], maxlen = args.replay_size), deque([], maxlen = args.replay_size)

    for iteration in range(args.episodes):

        Batch_data = random.sample(training_data, args.batch_playout_size)
        train_loader, valid_loader, pos_train, neg_train, pos_test, neg_test = SPRNABatchPlayout(args, Batch_data, P, U, 
                                                                                                pos_train, neg_train, pos_test, neg_test,
                                                                                                RNAEnv, model) #batch playout
        model.trainer(train_loader, valid_loader)  
        #test(args, model, iteration, P, U, RNAEnv, evaluation_data, track_evaluation) 
        print(f"pos_test {len(pos_test)} pos_train {len(pos_train)} neg_test {len(neg_test)} neg_train {len(neg_train)}")
        logSummaries(iteration, track_evaluation, evaluation_data)
        
    print(f"Done! ...")
