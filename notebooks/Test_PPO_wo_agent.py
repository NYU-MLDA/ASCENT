#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, sys

path = os.path.abspath("/home/jb7410/scarl_home/scarl/scarl")
sys.path.append(path)


# In[3]:


from Synthesizor import Synthesizor


# In[4]:


class Employee(object):
    def __init__(self, _dict):
        self.__dict__.update(_dict)


# In[5]:


run_params = {
        "NUM_LENGTH_RECIPE":18,
        "TOP_MODULE_NAME": 'aes128_table_ecb'
}
argsVals = {
        'file': "/home/jb7410/scarl_home/data/aes128_table_ecb_mod.v",
        'lib': "/home/jb7410/scarl_home/data/merge.lib",
        'dump':"/home/jb7410/scarl_home/dump/trial35_wo_agent",
        'params':run_params
}
logfile = "/home/jb7410/scarl_home/dump/trial35_wo_agent"


# In[6]:


args = Employee(argsVals)
synthEnv = Synthesizor(args)
synthEnv.checkFilePathsAndCreateAig()
state = synthEnv.init_state()


# In[7]:


argsVals['dump']


# In[8]:


def evaluate_model(rl_mod):
    state, _ = synthEnv.reset()
    #print(state)
    terminated=False
    while not terminated:
        action, _states = rl_mod.predict(state)
        state, reward, terminated, _, _ = synthEnv.step(action[0])
    print(f"Final Reward: {reward}")


# In[9]:


from PPO import PPO


# In[10]:


rl_model = PPO("AIGPolicy",synthEnv,verbose=1,tensorboard_log=argsVals['dump']+"/ppo_scarl_tensorboard/",seed=566,agent_recommendation_enabled=False)


# In[11]:


from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=18, save_path=argsVals['dump']+'/logs/',name_prefix='rl_model')


# In[12]:


#rl_model.learn(total_timesteps=1,tb_log_name="first_run",progress_bar=True,callback=checkpoint_callback)
# rl_model.learn(total_timesteps=7500,tb_log_name="first_run",progress_bar=True,callback=checkpoint_callback)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="second_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="third_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="fourth_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="fifth_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="sixth_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="seventh_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="eighth_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="ninth_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)
# rl_model.learn(total_timesteps=7500,tb_log_name="tenth_run",progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
# print("\nFINAL REWARD\n")
# print("--------------\n")
# evaluate_model(rl_model)

num_iters = 75

for idx in range(num_iters):
    log_name="run_"+str(idx)
    rl_model.learn(total_timesteps=1000,tb_log_name=log_name,progress_bar=True,callback=checkpoint_callback,reset_num_timesteps=False)
    print("--------------")
    print("Evaluate RL model")
    evaluate_model(rl_model)
    print("--------------")
# In[ ]:


evaluate_model(rl_model)


# In[ ]:


evaluate_model(rl_model)


# In[ ]:


state, _ = synthEnv.reset()
action, _states = rl_model.predict(state)
pyg_data = rl_model.policy.dict_to_pyg(state)
print(rl_model.policy.get_distribution(pyg_data).distribution.probs)


# In[ ]:


state, _ = synthEnv.reset()
action, _states = rl_model.predict(state)
pyg_data = rl_model.policy.dict_to_pyg(state)
print(rl_model.policy.get_distribution(pyg_data).distribution.probs)
print(action)


# In[ ]:


state, _ = synthEnv.reset()
action, _states = rl_model.predict(state)
pyg_data = rl_model.policy.dict_to_pyg(state)
print(rl_model.policy.get_distribution(pyg_data).distribution.probs)
print(action)


# In[ ]:


state, _ = synthEnv.reset()
action, _states = rl_model.predict(state)
# pyg_data = rl_model.policy.dict_to_pyg(state)
# print(rl_model.policy.get_distribution(pyg_data).distribution.probs)
print(action)


# In[ ]:


state, _ = synthEnv.reset()
action, _states = rl_model.predict(state)
pyg_data = rl_model.policy.dict_to_pyg(state)
print(rl_model.policy.get_distribution(pyg_data).distribution.probs)
print(action)


# In[ ]:


import torch


# In[ ]:


state, _ = synthEnv.reset()
action, _states = rl_model.predict(state)
pyg_data = rl_model.policy.dict_to_pyg(state)
print(rl_model.policy.get_distribution(pyg_data).distribution.probs)
print(action)


# In[ ]:


state, _ = synthEnv.reset()
action, _states = rl_model.predict(state)
pyg_data = rl_model.policy.dict_to_pyg(state)
print(rl_model.policy.get_distribution(pyg_data).distribution.probs)
print(torch.argmax(rl_model.policy.get_distribution(pyg_data).distribution.probs, dim=1))
print(action)


# In[ ]:


state, _ = synthEnv.reset()
action, _states = rl_model.predict(state)
pyg_data = rl_model.policy.dict_to_pyg(state)
print(rl_model.policy.get_distribution(pyg_data).get_actions(deterministic=True))
print(rl_model.policy.get_distribution(pyg_data).get_actions(deterministic=False))
print(rl_model.policy.get_distribution(pyg_data).distribution.probs)
print(torch.argmax(rl_model.policy.get_distribution(pyg_data).distribution.probs, dim=1))
print(action)


# In[ ]:





# In[ ]:




