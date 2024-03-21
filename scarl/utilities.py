# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import logging
import os.path as osp
import torch_geometric

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


def extract_state_from_aig(self,aigState):
    fileBaseName = os.path.splitext(os.path.basename(aigState))[0]
    _,seq,stepNum = fileBaseName.split('+')
    stepNum = int(stepNum.split('step')[-1])
    recipe_encoding = [int(x) for x in seq]
    
    self._abc.read(aigState)
    data = {}
    aigGraph = nx.DiGraph()
    numNodes = self._abc.numNodes()
    data['node_type'] = np.zeros(numNodes,dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes,dtype=int)
    edge_src_index = []
    edge_target_index = []
    edge_type = []
    for nodeIdx in range(numNodes):
        aigNode = self._abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == self.aig_node_type['AIG_NODE_CONST1'] or nodeType == self.aig_node_type['AIG_NODE_PI']:
            data['node_type'][nodeIdx] = 0       # Primary Inputs     
        elif nodeType == self.aig_node_type['AIG_NODE_PO']:
            data['node_type'][nodeIdx] = 1       # Primary Outputs
        else:
            data['node_type'][nodeIdx] = 2       # Internal Nodes
            if nodeType == self.aig_node_type['AIG_NODE_INVNO'] or nodeType == self.aig_node_type['AIG_NODE_NOINV']:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == self.aig_node_type['AIG_NODE_INVINV']:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
            if nodeType == self.aig_node_type['AIG_NODE_INVNO'] or nodeType == self.aig_node_type['AIG_NODE_INVINV']:
                edge_type.append(0)
            else:
                edge_type.append(1)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
            if nodeType == self.aig_node_type['AIG_NODE_NOINV'] or nodeType == self.aig_node_type['AIG_NODE_INVINV']:
                edge_type.append(0)
            else:
                edge_type.append(1)
        aigGraph.add_node(nodeIdx,node_type=data['node_type'][nodeIdx],num_inverted_predecessors=data['num_inverted_predecessors'][nodeIdx])
                    
    data['edge_index'] = torch.tensor([edge_src_index,edge_target_index],dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['edge_attr'] = torch.tensor(np.array(edge_type).reshape(-1,1))
    #data = torch_geometric.data.Data.from_dict(data)
    #data.num_nodes = numNodes
    data['nodes'] = numNodes
    for idx,src,target in enumerate(zip(edge_src_index,edge_target_index)):
        aigGraph.add_edge(src,target,edge_logic=edge_type[idx])
    aigGraph.graph['recipe_len'] = stepNum
    aigGraph.graph['recipe_encoding'] = recipe_encoding
    return aigGraph