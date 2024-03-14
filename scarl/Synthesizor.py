# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import numpy as np
import torch
import gym
import os,re
import math
from utilities import 
from static_env import StaticEnv
import abc_py as abcPy
import os.path as osp
import os,shutil
from utilities import Logger
from collections import OrderedDict

class Synthesizor:
    def __init__(self, args,logFile):
        self._abc = abcPy.AbcInterface()
        self._abc.start()
        self.verilog_file = args.file
        self.lib = args.lib
        self.ep_length = args.params['NUM_LENGTH_EPISODES']
        self.root_dump_dir = args.dump
        self.logFile = logFile
        self.args = args
        self.aig_dump_dir = osp.join(self.root_dump_dir, 'aig_states')
        self.set_aig_node_type_mapping()
        self.generate_synthesis_id_2_cmd_mapping()

    def set_aig_node_type_mapping(self):
        self.aig_node_type = OrderedDict(\
            {
                'AIG_NODE_CONST1': 0,
                'AIG_NODE_PO': 1,
                'AIG_NODE_PI': 2,
                'AIG_NODE_NONO':3,
                'AIG_NODE_INVNO':4,
                'AIG_NODE_INVINV':5,
                'AIG_NODE_NUMBER':6,
                'AIG_NODE_NOINV':7
            })
        
    def generate_synthesis_id_2_cmd_mapping(self):
        self.synthesisId2CmdMapping = OrderedDict(\
        {
            0: "refactor",
            1: "refactor -z",
            2: "rewrite" ,
            3: "rewrite -z" ,
            4: "resub -K 6" ,
            5: "resub -K 6 -N 2",
            6: "resub -K 8",
            7: "resub -K 8 -N 2",
            8: "resub -K 10",
            9: "resub -K 10 -N 2",
            10: "resub -K 12",
            11: "resub -K 12 -N 2",
            12: "balance"
        })
    
    def generate_synthesis_cmd(self,synthesis_vec):
        endl = "\n"
        for i in synthesis_vec:
            synthesisCmd += (self.synthesisId2CmdMapping[int(math.floor(i))]+endl)
        return synthesisCmd
        
    def create_yosys_script_file(self,yosys_script,abc_script,synthesized_file_path):
        #Define the end of line character
        endl = "\n"
        #Open the yosys_script file in write mode
        with open(yosys_script,'w') as f:
            #Write the read_verilog command to the file
            f.write('read_verilog -sv '+self.verilog_file+endl)
            #Write the hierarchy command to the file
            f.write('hierarchy -check -top '+self.args.params['TOP_MODULE_NAME']+endl)
            #Write the flatten command to the file
            f.write('flatten '+endl+'synth -top '+self.args.params['TOP_MODULE_NAME']+endl+'flatten'+endl)
            #Write the proc, fsm, opt, memory, opt, techmap and opt commands to the file
            f.write('proc; fsm ; opt; memory; opt; techmap; opt'+endl)
            #Write the dfflibmap command to the file
            f.write('dfflibmap -liberty '+self.lib+endl)
            #Write the abc command to the file
            f.write('abc -liberty '+self.lib+' -script'+abc_script+endl)
            #Write the write_verilog command to the file
            f.write('write_verilog -noattr '+synthesized_file_path+endl)
            #Write the stat command to the file
            f.write('stat -liberty '+self.lib+endl)
            
    def create_abc_script_file(self,abc_script,synthesis_vec,aigPath=None):
        endl='\n'
        with open(abc_script,'w') as f:
            f.write('strash'+endl)
            if aigPath:
                f.write(f"write {aigPath}"+endl)
            synthesisCmd = self.generate_synthesis_cmd(synthesis_vec)
            f.write(synthesisCmd)
            f.write('amap'+endl+'topo'+endl+'stime -c'+endl+'buffer -c'+endl+'upsize -c'+endl+'dnsize -c')
        
    def checkFilePathsAndCreateAig(self):
        if not osp.exists(self.verilog_file):
            self.logFile.errorInfo(f"{self.verilog_file} not found")
            exit(1)
            
        if not osp.exists(self.lib):
            self.logFile.errorInfo(f"{self.lib} not found")
            exit(1)
            
        if not osp.exists(self.root_dump_dir):
            self.logFile.errorInfo(f"{self.root_dump_dir} not found")
            exit(1)

        # Create appropriate file and scripts paths to extract original AIG
        origSynthesisDir = osp.join(self.root_dump_dir, "orig_synthesis")
        abcScript = osp.join(origSynthesisDir, "abc.script")
        yosysScript = osp.join(origSynthesisDir, "yosys_script.tcl")
        synthesizedVerilogNetlist = osp.join(origSynthesisDir, "synthesized_netlist.v")
        origAIGPath = osp.join(origSynthesisDir, "orig_aig.aig")
        
        
        # compress2rs synthesis vector and create the scripts file
        compress2rs_vec = [12,4,2,5,0,6,12,7,2,8,3,9,12,10,1,11,3,12]
        self.create_yosys_script_file(yosysScript, abcScript,synthesizedVerilogNetlist)
        self.create_abc_script_file(abcScript,compress2rs_vec,origAIGPath)
        
        # Generate the original AIG file
        yosysRunLogFile = osp.join(origSynthesisDir, "yosys_syn.log")
        yosys_synthesis_cmd = f'yosys -s {yosysScript} -l {yosysRunLogFile}'
        os.system(yosys_synthesis_cmd)
        
        # Create the copy of original AIG file in aig dump folder
        self.origAIG = osp.join(self.aig_dump_dir, "orig_aig+0+step0.aig")
        shutil.copy(origAIGPath, self.origAIG)
        
        
    def init_state(self):
        state = self.origAIG
        return state
        
    def define_action_space(self):
        self.action_set = self.synthesisId2CmdMapping.keys()
        self.n_actions = len(self.action_set)
        
    def next_state(self,current_state,action,depth):
        fileDirName = osp.dirname(current_state)
        fileBaseName,prefix,_ = osp.splitext(osp.basename(current_state))[0].split("+")
        nextState = os.path.join(fileDirName,fileBaseName+"+"+prefix+str(action)+"+step"+str(depth+1)+".aig")
        self._abc.read(current_state)
        if action == 0:
            self._abc.refactor(l=False, z=False) #rf
        elif action == 1:
            self._abc.refactor(l=False, z=True) #rf -z
        elif action == 2:
            self._abc.rewrite(l=False, z=False) #rw -z
        elif action == 3:
            self._abc.rewrite(l=False, z=True) #rw -z
        elif action == 4:
            self._abc.resub(k=6,n=1,l=False, z=False) #rs -K 6
        elif action == 5:
            self._abc.resub(k=6,n=2,l=False, z=False) #rs -K 6 -N 2
        elif action == 6:
            self._abc.resub(k=8,n=1,l=False, z=False) #rs -K 8
        elif action == 7:
            self._abc.resub(k=8,n=2,l=False, z=False) #rs -K 8 -N 2
        elif action == 8:
            self._abc.resub(k=10,n=1,l=False, z=False) #rs -K 10
        elif action == 9:
            self._abc.resub(k=10,n=2,l=False, z=False) #rs -K 10 -N 2
        elif action == 10:
            self._abc.resub(k=12,n=1,l=False, z=False) #rs -K 12
        elif action == 11:
            self._abc.resub(k=12,n=2,l=False, z=False) #rs -K 12 -N 2
        elif action == 12:
            self._abc.balance(l=False) #balance
        else:
            assert(False)
        self._abc.write(nextState)
        return nextState
    
    def extract_graph_state(self,state):
        self._abc.read(state)
        data = {}
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
        data['edge_index'] = torch.tensor([edge_src_index,edge_target_index],dtype=torch.long)
        data['node_type'] = torch.tensor(data['node_type'])
        data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
        data['edge_attr'] = torch.tensor(np.array(edge_type).reshape(-1,1))
        #data = torch_geometric.data.Data.from_dict(data)
        #data.num_nodes = numNodes
        data['nodes'] = numNodes
        return data