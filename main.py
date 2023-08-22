import random
import os
from utils import *
from NNet import RnaNet
from Agent import Agent, test
from RNAGymEnv import RNAEnv
from RNAGenerate import BatchPlay
from config import Config

CFG = Config()

if __name__ == "__main__":
    if not os.path.exists(CFG.root_path):os.mkdir(CFG.root_path)
    seed_everything(seed=42)

    agent = Agent(CFG, RnaNet())
    environment = RNAEnv(CFG)

    track_evaluation = trackEvalResults()
    training_data, evaluation_data = loadCandD("train"), loadValidation()
    print(track_evaluation.keys(), evaluation_data.keys())

    pos_train, neg_train, pos_test, neg_test = [deque([], maxlen=CFG.replay_size) for _ in range(4)]

    for iteration in range(CFG.episodes):
        #test(CFG, agent, iteration, environment, evaluation_data, track_evaluation) 
        batch_data = random.sample(training_data, CFG.batch_playout_size)
        train_loader, valid_loader, pos_train, neg_train, pos_test, neg_test = \
            BatchPlay(CFG, batch_data, pos_train, neg_train, pos_test, neg_test, environment, agent)

        agent.update(train_loader, valid_loader)  
        test(CFG, agent, iteration, environment, evaluation_data, track_evaluation)

        logSummaries(iteration, track_evaluation, evaluation_data)
        
    print(f"Done!")
