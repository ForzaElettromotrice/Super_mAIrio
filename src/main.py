import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
import torch

from nes_py.wrappers import JoypadSpace

from mario import Mario
from wrapper import apply_wrappers

ENV = 'SuperMarioBros-v0'
TRAIN = False
DISPLAY = True
EPISODES = 50000

def main():
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents = True)

    if torch.cuda.is_available():
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available")
    
    env = gym_super_mario_bros.make(ENV, apply_api_compatibility = True, render_mode='human' if DISPLAY else 'rgb')

    # Apply Wrappers to environment
    env = apply_wrappers(env)

    # Limit the actions to
    #  - walk right
    #  - jump right
    env = JoypadSpace(env, [['right'], ['right', 'A']])

    checkpoint = Path("./checkpoints/checkpoint-solo-destra/best_one.chkpt")
    mario = Mario(state_dim = env.observation_space.shape, action_dim = env.action_space.n, save_dir = save_dir, checkpoint = checkpoint)

    # mario.burnin = 32
    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        max_x = 0

        while True:
            action = mario.act(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # if mario.curr_step % 5000 == 0:
            #     print(mario.exploration_rate)

            if TRAIN: 
                mario.cache(state, next_state, action, reward, done)
                mario.learn()

            # Update state
            state = next_state
            total_reward += reward
            max_x = max(info["x_pos"], max_x)

            # Check if end of game
            if done or info["flag_get"]:
                break

        print(f"Episode {e + 1} Total reward: {total_reward}, Max x: {max_x}")

if __name__ == '__main__':
    main()
