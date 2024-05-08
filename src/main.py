from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
import cv2
import pygame

# installare pygame, gym-super-mario-bros, nes-py

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
clock = pygame.time.Clock()
for step in range(5000):
    clock.tick(24)
    if done:
        state = env.reset()
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    
    state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Image", state)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    env.render()

env.close() 