# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import logging
import os.path as osp
import torch_geometric

class Logger:
    def __init__(self,args,fileName=None):
        if fileName:
            self.loggingFileName = osp.join(args.dump,fileName)
        else:
            self.loggingFileName = osp.join(args.dump,'run.log')
        logging.basicConfig(filename=self.loggingFileName)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def printInfo(self,msg):
        self.logger.info(msg)
    
    def errorInfo(self,msg):
        self.logger.error(msg)
        
    def warningInfo(self,msg):
        self.logger.warn(msg)
        
        
def create_pyg_object(self,seqSentence,graphData=None):
    data={}
    if not graphData == None:
        for k,v in graphData.items():
            data[k] = v
        numNodes = data['nodes']
        data = torch_geometric.data.Data.from_dict(data)
        data.num_nodes = numNodes
    else:
        data = torch_geometric.data.Data.from_dict(data)
    return data