# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import logging
import os.path as osp

class Logger:
    def __init__(self,args):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.loggingFileName = osp.join(args.dump,'run.log')
        self.logger.basicConfig(filename=self.loggingFileName)
        
    def printInfo(self,msg):
        self.logger.info(msg)
    
    def errorInfo(self,msg):
        self.logger.error(msg)
        
    def warningInfo(self,msg):
        self.logger.warn(msg) 