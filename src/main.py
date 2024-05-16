from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
import cv2
import pygame
import gym
#   print(gym.__version__) # 0.26.2

def main():
    # JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')

    # env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [['right'], ['right', 'A']])


    done = True
    clock = pygame.time.Clock()
    for step in range(5000):
        clock.tick(24)
        if done:
            state = env.reset()
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        
        # state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # state = cv2.resize(state, (84, 84))
        
        cv2.imshow("State", state)
        
        env.render()
        
    env.close() 

if __name__ == '__main__':
    main()