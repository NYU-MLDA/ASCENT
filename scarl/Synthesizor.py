# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import os,re
import math
import abc_py as abcPy
import os.path as osp
import os,shutil
from utilities import Logger
from collections import OrderedDict
import torch_geometric
import torch_geometric.data
from torch_geometric.utils.convert import to_networkx,from_networkx
import networkx as nx
from zipfile import ZipFile
import zipfile
from gymnasium.spaces import Dict

class Synthesizor(gym.Env):
    def __init__(self, args):
        super().__init__()
        self._abc = abcPy.AbcInterface()
        self._abc.start()
        self.verilog_file = args.file
        self.lib = args.lib
        self.ep_length = args.params['NUM_LENGTH_RECIPE']
        self.root_dump_dir = args.dump
        self.logFile = Logger(args,"synthesizor.log")
        self.args = args
        self.aig_dump_dir = osp.join(self.root_dump_dir, 'aig_states')
        self.set_aig_node_type_mapping()
        self.generate_synthesis_id_2_cmd_mapping()
        self.define_action_space()
        self.num_envs=1 # Tackle VecEnv num_envs dummy
        

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
        synthesisCmd=""
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
            f.write('abc -liberty '+self.lib+' -script '+abc_script+endl)
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
    
    @staticmethod        
    def get_area_from_synthesis_logfile(synthesis_logfile):
        area_parsing_cmd = "grep 'Chip area for module' "+synthesis_logfile+" | awk -F':' '{print $2}' | awk '{$1=$1};1'"
        print(os.popen(area_parsing_cmd).read().strip())
        area = float(os.popen(area_parsing_cmd).read().strip())
        return area
        
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
        if osp.exists(origSynthesisDir):
            shutil.rmtree(origSynthesisDir)
        os.mkdir(origSynthesisDir)
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
        yosys_synthesis_cmd = f'yosys -s {yosysScript} -l {yosysRunLogFile} > /dev/null 2>&1'
        os.system(yosys_synthesis_cmd)
        self.c2rs_area = self.get_area_from_synthesis_logfile(yosysRunLogFile)
        
        # Create the copy of original AIG file in aig dump folder
        if osp.exists(self.aig_dump_dir):
            shutil.rmtree(self.aig_dump_dir)
        os.mkdir(self.aig_dump_dir)
        self.origAIG = osp.join(self.aig_dump_dir, "orig_aig+0+step0.aig")
        shutil.copy(origAIGPath, self.origAIG)
        
        
    def get_state_from_aig(self,aigState):
        nxState = os.path.splitext(aigState)[0]+'.pt.zip'
        if os.path.exists(nxState):
            filePathName = osp.basename(osp.splitext(nxState)[0])
            with ZipFile(nxState) as myzip:
                with myzip.open(filePathName) as myfile:
                    state = torch.load(myfile)
        else:
            state = self.extract_state_from_aig(aigState)
            ptFilePath = nxState.split('.zip')[0]
            torch.save(state,ptFilePath)
            with ZipFile(nxState,'w',zipfile.ZIP_DEFLATED) as fzip:
                fzip.write(ptFilePath,arcname=osp.basename(ptFilePath))
            os.remove(ptFilePath)
        return state
        
        
    def init_state(self):
        aigState = self.origAIG
        self.state = self.get_state_from_aig(aigState)
        self.depth = 0
        return self.state
    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        return self.init_state(),{}
    
    def render(self):
        return super().render()

    def close(self):
        return super().close()
        
    def define_action_space(self):
        self.action_set = self.synthesisId2CmdMapping.keys()
        self.n_actions = len(self.action_set)
        self.action_space = spaces.Discrete(self.n_actions)
        
    def step(self,action):
        assert action in self.action_set
        next_state = self.next_state(action)
        self.state = next_state
        self.depth += 1
        terminated = self.depth == self.ep_length
        if terminated:
            reward = self.get_reward(next_state)
        else:
            reward = 0.0
        return next_state, reward, terminated, False, {}
        
    def next_state(self,action):
        """_summary_
        Args:
            state (nx.DiGraph): AIG encoded as NetworkX DiGraph
            action (_type_): ABC synthesis action
            depth (_type_): Current depth of synthesis recipe

        Returns:
            state: Synthesized AIG after applying the synthesis transformation
        """
        # Hack to get the aig state from networkx graph object.
        # Turn it to pytorch data dictionary if using pytorch object as state
        if isinstance(self.state,nx.DiGraph):
            current_aig_state = self.state.graph['aig_state']
        elif isinstance(self.state,dict):
            current_aig_state = self.state['aig_state']
        else:
            print("Instance type unknown. Exiting")
            exit(1)
        fileDirName = osp.dirname(current_aig_state)
        fileBaseName,prefix,stepNum = osp.splitext(osp.basename(current_aig_state))[0].split("+")
        depth = int(stepNum.split('step')[-1])+1
        next_aig_state = os.path.join(fileDirName,fileBaseName+"+"+prefix+"_"+str(action)+"+step"+str(depth)+".aig")
        self._abc.read(current_aig_state)
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
        self._abc.write(next_aig_state)
        nextState = self.get_state_from_aig(next_aig_state)
        return nextState
    
    def get_reward(self,state):
        """_summary_

        Args:
            state (nxDigraph): The function expects the state information in the form of AIG DiGraph
            or PyG Geometric Data dictionary. Use "aig_state" attribute to retrieve the path of terminal
            state AIG, extract the recipe to write down the abc.script.

        Returns:
            float : The reward should be designed on post synthesized verilog file. e.g. area, PT score etc.
        """
        if isinstance(state,nx.DiGraph):
            state = state.graph['aig_state']
        elif isinstance(state,dict):
            state = state['aig_state']
        else:
            print("Instance type unknown. Exiting")
            exit(1)
        _,prefix,_ = osp.splitext(osp.basename(state))[0].split("+")
        rl_synthesisVec = [int(synthid) for synthid in prefix.split("_")[1:]] # The first one is always 0
        
        # Create appropriate file and scripts paths to extract original AIG
        rewardSynthesisDir = osp.join(self.root_dump_dir, "reward_synthesis")
        if not osp.exists(rewardSynthesisDir):
            os.mkdir(rewardSynthesisDir)
        abcScript = osp.join(rewardSynthesisDir, "abc.script")
        yosysScript = osp.join(rewardSynthesisDir, "yosys_script.tcl")
        synthesizedVerilogNetlist = osp.join(rewardSynthesisDir, "synthesized_netlist.v")
        
        
        # create the scripts file
        self.create_yosys_script_file(yosysScript, abcScript,synthesizedVerilogNetlist)
        self.create_abc_script_file(abcScript,rl_synthesisVec)
        
        # Run the synthesis and gather the number
        yosysRunLogFile = osp.join(rewardSynthesisDir, "yosys_syn.log")
        yosys_synthesis_cmd = f'yosys -s {yosysScript} -l {yosysRunLogFile} > /dev/null 2>&1'
        os.system(yosys_synthesis_cmd)
        area = self.get_area_from_synthesis_logfile(yosysRunLogFile)
        reward = max(-1,(1-(area/self.c2rs_area)))
        return reward
    
    def extract_state_from_aig(self,aigState):
        fileBaseName = os.path.splitext(os.path.basename(aigState))[0]
        _,seq,stepNum = fileBaseName.split('+')
        stepNum = int(stepNum.split('step')[-1])
        recipe_encoding = [int(x) for x in seq.split("_")[1:]]
        
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
        data['x'] = torch.stack((data['node_type'],data['num_inverted_predecessors']),dim=0).T
        data['edge_attr'] = torch.tensor(np.array(edge_type).reshape(-1,1))
        #data = torch_geometric.data.Data.from_dict(data)
        #data.num_nodes = numNodes
        data['nodes'] = numNodes
        data['recipe_len'] = stepNum
        data['recipe_encoding'] = recipe_encoding
        data['aig_state'] = aigState
        for idx,(src,target) in enumerate(zip(edge_src_index,edge_target_index)):
            aigGraph.add_edge(src,target,edge_logic=edge_type[idx])
        aigGraph.graph['recipe_len'] = stepNum
        aigGraph.graph['recipe_encoding'] = recipe_encoding
        aigGraph.graph['aig_state'] = aigState
        return data

class dictStruct:
    def __init__(self, dictObj):
        self.__dict__.update(dictObj)

#   Test logic synthesis environment
if __name__ == '__main__':
    run_params = {
        "NUM_LENGTH_RECIPE":18,
        "TOP_MODULE_NAME": 'aes128_table_ecb'
    }
    args = {
        'file': "/home/jb7410/scarl_home/data/aes128_table_ecb_mod.v",
        'lib': "/home/jb7410/scarl_home/data/merge.lib",
        'dump':"/home/jb7410/scarl_home/dump/trial1",
        'params':run_params
    }

    argObj = dictStruct(args)
    #logfile = "/home/jb7410/scarl_home/dump/trial1"
    synthObj = Synthesizor(argObj)
    synthObj.checkFilePathsAndCreateAig()
    obs, _ = synthObj.reset()
    done = False
    n_steps = 0
    while not done:
        # Take random actions
        random_action = synthObj.action_space.sample()
        obs, reward, terminated, truncated, info = synthObj.step(random_action)
        done = terminated or truncated
        print(reward, terminated, truncated, info,done)
        n_steps += 1

    print(n_steps, info)
