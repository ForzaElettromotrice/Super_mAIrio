import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import datetime
from pathlib import Path

# from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers2 import apply_wrappers

from mario2 import Mario

import os

# from utils import *

ENV = 'SuperMarioBros-1-1-v0'
TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
EPISODES = 10000

def main():
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents = True)

    if torch.cuda.is_available():
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available")

    
    env = gym_super_mario_bros.make(ENV, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    
    env = apply_wrappers(env)
    
    mario = Mario(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

    if not TRAIN:
        folder_name = ""
        ckpt_name = ""
        mario.load_model(os.path.join("models", folder_name, ckpt_name))
        mario.epsilon = 0.2
        mario.eps_min = 0.0
        mario.eps_decay = 0.0
    
    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    
    for e in range(EPISODES):
        total_reward = 0
        max_x = 0
        done = False
        
        state, _ = env.reset()
        
        while not done:
            if DISPLAY:
                env.render()
            
            action = mario.do_action(next_state)
            next_state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            
            max_x = max(info["x_pos"], max_x)
            
            if TRAIN:
                mario.store_memory(state, action, reward, next_state, done)
                mario.train()
                
            state = next_state
            
            if done:
                break
        
        if TRAIN and (e + 1) % CKPT_SAVE_INTERVAL == 0:
            mario.save_model(os.path.join(save_dir, "model_" + str(e + 1) + "_iter.pt"))
        
        
        print(f"Episode {e + 1} Total reward: {total_reward}, Max x: {max_x}")
        
    env.close()

if __name__ == "__main__":
    main()