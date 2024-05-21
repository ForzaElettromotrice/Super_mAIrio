import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
import torch
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from mario import Mario
from wrapper import SkipFrame, GrayScaleObservation, ResizeObservation


def custom_reward(old, new):
    if old is None:
        return 0

    v = new["x_pos"] - old["x_pos"]
    c = new["time"] - old["time"]
    d = old["life"] - new["life"]
    p = new["score"] - old["score"]

    reward = 0

    reward += p / 2

    if new["status"] == "tall" and old["status"] == "small":
        reward += 500

    if d == 1:
        reward -= 30

    if v == 0:
        reward -= 10

    reward += v * 3 + c

    return reward


#   print(gym.__version__) # 0.26.2

def main():
    # JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility = True, render_mode = 'human')

    # Apply Wrappers to environment
    env = SkipFrame(env, skip = 4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape = 84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack = 4, new_step_api = True)
    else:
        env = FrameStack(env, num_stack = 4)

    # env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [['right'], ['right', 'A'], ['left'], ['A'], ['left', 'A']])
    # env = JoypadSpace(env, [['right'], ['right', 'A'], ['A']])
    # done = True
    # clock = pygame.time.Clock()
    # for step in range(5000):
    #     clock.tick(24)
    #     if done:
    #         state = env.reset()
    #     state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    #     done = terminated or truncated

    #     # state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    #     # state = cv2.resize(state, (84, 84))

    #     cv2.imshow("State", state)

    #     env.render()

    # env.close() 
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}\n")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents = True)

    checkpoint = Path("./checkpoints/2024-05-21T12-44-08/mario_net_7.chkpt")
    # checkpoint = None

    mario = Mario(state_dim = (4, 84, 84), action_dim = env.action_space.n, save_dir = save_dir, checkpoint = checkpoint)

    # logger = MetricLogger(save_dir)
    # mario.burnin = 10000
    episodes = 1000
    # mario.exploration_rate = 1
    for e in range(episodes):
        print(f"Episode {e + 1}")
        state = env.reset()

        # Play the game!
        old_info = None

        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, terminated, truncated, info = env.step(action)
            # print(reward)
            reward = custom_reward(old_info, info)
            # print(reward)
            old_info = info
            # print(mario.curr_step)
            # print(reward, info)
            done = terminated or truncated
            # next_state, reward, done, trunc, info = env.step(action)
            if mario.curr_step % 5000 == 0:
                print(mario.exploration_rate)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            # logger.log_step(reward, loss, q)
            # print(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        # logger.log_episode()

        # if (e % 20 == 0) or (e == episodes - 1):
        #     logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)


if __name__ == '__main__':
    main()
