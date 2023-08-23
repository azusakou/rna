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

    training_data, test_data, track_test = loadCandD("train"), loadValidation(), trackEvalResults()
    print(track_test.keys(), test_data.keys())

    pos_train, neg_train, pos_test, neg_test = [deque([], maxlen=CFG.replay_size) for _ in range(4)]

    for iteration in range(CFG.episodes):
        if CFG.debug: test(CFG, agent, iteration, environment, test_data, track_test) 
        
        if not CFG.debug:
            batch_data = random.sample(training_data, CFG.batch_playout_size)
            train_loader, valid_loader, pos_train, neg_train, pos_test, neg_test = \
                BatchPlay(CFG, batch_data, pos_train, neg_train, pos_test, neg_test, environment, agent)

            agent.update(train_loader, valid_loader)  
            test_data,track_test = test(CFG, agent, iteration, environment, test_data, track_test)
            logSummaries(iteration, track_test, test_data)
        
    print(f"Done!")
