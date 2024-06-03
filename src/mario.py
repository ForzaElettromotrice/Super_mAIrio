import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from marionet import MarioNet


class Mario:
    def __init__(self, state_dim, 
                 action_dim, save_dir, 
                 checkpoint = None,
                 learning_rate=0.0005, 
                 gamma=0.9, 
                 exploration_rate=1.0, 
                 exploration_decay=0.99999975, 
                 exploration_min=0.1, 
                 replay_buffer_capacity=10_000, 
                 batch_size=32, 
                 sync_network_rate=10000, 
                 burnin = 1000,
                 learn_every = 3):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device = self.device)

        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_decay
        self.exploration_rate_min = exploration_min
        self.sync_every = sync_network_rate
        self.batch_size = batch_size
        self.burnin = burnin 
        self.learn_every = learn_every
        self.curr_step = 0
        self.save_every = 5e3 

        self.memory = TensorDictReplayBuffer(storage = LazyMemmapStorage(replay_buffer_capacity, device = torch.device("cpu")))

        self.use_cuda = torch.cuda.is_available()
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def act(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device = self.device).unsqueeze(0)
            action_values = self.net(state, model = "online")
            action_idx = torch.argmax(action_values, axis = 1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        
        self.memory.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def sync_Q_target(self):
        if self.curr_step % self.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        if self.curr_step % self.save_every == 0:
            save_path = (
                    self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
            )
            torch.save(
                dict(model = self.net.state_dict(), exploration_rate = self.exploration_rate),
                save_path,
            )
            print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        self.sync_Q_target()
        self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        q_est = self.net(state, model = "online")[np.arange(0, self.batch_size), action] 
        q_tar = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.loss_fn(q_est, q_tar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model = "online")
        best_action = torch.argmax(next_state_Q, axis = 1)
        next_Q = self.net(next_state, model = "target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location = ('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
