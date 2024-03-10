# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import numpy as np
import gym
import os,re
import torch
from utils import cprint
from static_env import StaticEnv
import abc_py as abcPy

class Synthesizor:
    def __init__(self, params):
        self._abc = abcPy.AbcInterface()
        self._abc.start()
        self.orig_aig = origAIG
        self.ep_length = NUM_LENGTH_EPISODES
        self.step_idx = 0
        self.lib = libFile
        self.logFile = logFile
        self.baselineReturn = self.getResynReturn()
        
    def getResynReturn(self):
        return False