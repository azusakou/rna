import random
import os
from utils import *
from NNet import RnaNet
from Agent import Agent, test
from RNAGymEnv import RNAEnv
from RNAGenerate import Playout as BatchPlay
#from RNAGenerate import BatchPlayout as BatchPlay # if CUDA
from config import Config

CFG = Config()

if __name__ == "__main__":
    if not os.path.exists(CFG.root_path):os.mkdir(CFG.root_path)
    seed_everything(seed=42)

    agent = Agent(CFG, RnaNet())
    environment = RNAEnv(CFG)

    training_data, test_data, track_test = loadTrain(), loadValidation(), trackEvalResults()
    if CFG.trackprint: print(track_test.keys(), test_data.keys())

    pos_train, neg_train, pos_test, neg_test = [deque([], maxlen=CFG.replay_size) for _ in range(4)]
    pos_train_len = len(pos_train)
    for iteration in range(CFG.episodes):
        if CFG.debug: test(CFG, agent, iteration, environment, test_data, track_test) 
        
        if not CFG.debug:
            batch_data = pick_samples(CFG, training_data, iteration, CFG.batch_playout_size)
            train_loader, valid_loader, pos_train, neg_train, pos_test, neg_test = \
                BatchPlay(CFG, batch_data, pos_train, neg_train, pos_test, neg_test, environment, agent)

            #if len(pos_train) > -1 and (len(pos_train) > pos_train_len): 
            agent.update(train_loader, valid_loader);pos_train_len = len(pos_train)
            if (iteration+1)%30 == 0 or (iteration+1) == CFG.episodes:
                test_data,track_test = test(CFG, agent, iteration, environment, test_data, track_test)
                logSummaries(iteration, track_test, test_data)
        
    print(f"Done!")
