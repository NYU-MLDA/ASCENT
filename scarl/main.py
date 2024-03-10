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


if __name__ == '__main__':

    args = options().parse_args()
    yaml.load(args.params,Loader=yaml.FullLoader)
