# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import yaml
import os
import argparse
import datetime
import numpy as np
import time
from config import options
from utilities import log
import argparse
from utilities import Logger
import gymnasium as gym
from PPO import PPO
from Synthesizor import Synthesizor

def RLTrainer(args,logFile):
    synthesis_env = Synthesizor(args,logFile)
    rl_model = PPO("AIGPolicy",synthesis_env,verbose=1)
    rl_model.learn(total_timesteps=args.num_timesteps)
    rl_model.save("scarl_ppo_model")
    
def RLEvaluator(args,logFile):
    rl_model = PPO.load("scarl_ppo_model")
    synthesis_env = Synthesizor(args,logFile)
    state, _ = synthesis_env.reset()
    terminated=False
    while not terminated:
        action, _states = rl_model.predict(state)
        state, reward, terminated, _, _ = synthesis_env.step(action)
    print(f"Final Reward: {reward}")    

if __name__ == '__main__':
    args = options().parse_args()
    yaml.load(args.params,Loader=yaml.FullLoader)
    logFile = Logger(args)
    RLTrainer(args,logFile)
    RLEvaluator(args,logFile)