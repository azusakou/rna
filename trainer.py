from utils import *
from os import path
import os
from RNADataLoader import value
from tqdm import tqdm

def test(args, model, iteration, P, U, RNAEnv, evaluation_data,track_evaluation):
    if not path.exists(args.eval_dir):os.mkdir(args.eval_dir)
    data = list(evaluation_data.keys())
    dirs = [f"{args.eval_dir}{x}/" for x in data]
    for p in dirs:
        if not path.exists(p):os.mkdir(p)
    for d, loc in zip(data, dirs):
        tbar = tqdm(evaluation_data[d])
        for seq_id, seq in enumerate(tbar):
            if len(list(seq)) > args.maxseq_len:continue
            state = RNAEnv(seq, args.W, P, U)
            sites = state.availableSites()
            sites = state.shuffleSites()
            for s in sites:
                state.current_site = s
                if state.currentPaired():best_action = value(model, state, P, True)
                else:best_action = value(model, state, U, True)
            state.applyMove(best_action)
            if state.hammingLoss() > 0 : state.localSearch()
            if state.reward() == 1.0 : track_evaluation[d].add(seq_id)
            writeSummary(seq_id, iteration, state.hammingLoss(), state.getDesigned(), loc+"summary_"+str(iteration)+".csv")
            tbar.set_description(f"Inference on {d} iteration {iteration} seq {len(list(seq))} ")

        print(f"a check! {d}, {loc} {track_evaluation[d]}")