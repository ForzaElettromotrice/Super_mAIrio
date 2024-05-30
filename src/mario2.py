import torch
import numpy as np
from marionn2 import MarioNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class Mario:
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 learning_rate=0.00025, 
                 gamma=0.9, 
                 exploration_rate=1.0, 
                 exploration_decay=0.99999975, 
                 exploration_min=0.1, 
                 replay_buffer_capacity=10_000, 
                 batch_size=32, 
                 sync_network_rate=10000):
        
        self.num_actions = num_actions
        self.train_step_counter = 0

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate
        self.burnin = 32
        
        self.use_cuda = torch.cuda.is_available()

        # Networks for Qonline and Qtarget
        self.online_network = MarioNN(input_dims, num_actions)
        self.target_network = MarioNN(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.SmoothL1Loss()

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def do_action(self, observation):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.num_actions)

        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        # Grabbing the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()
    
    def decay_exploration_rate(self):
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay, self.exploration_min)

    def store_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))
        
    def sync_networks(self):
        if self.train_step_counter % self.sync_network_rate == 0 and self.train_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        # torch.save(self.online_network.state_dict(), path)
        torch.save(
            dict(model = self.online_network.state_dict(), exploration_rate = self.exploration_rate),
            path
        )
        print(f"Checkpoint {path} saved successfully")

    def load_model(self, path):
        if not path:
            return
        if not path.exists():
            raise ValueError(f"{path} does not exist")
        
        ckp = torch.load(path, map_location = ('cuda' if self.use_cuda else 'cpu'))
        self.exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        
        print(f"Loading model {path} with exploration rate {self.exploration_rate}")
        self.online_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if self.train_step_counter < self.burnin:
            self.train_step_counter += 1
            return
        
        self.sync_networks()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(states)[np.arange(0, self.batch_size), actions]  # Shape is (batch_size, n_actions)
        # predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions]

        # Max returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        next_q_values = self.online_network(next_states)
        best_action = torch.argmax(next_q_values, axis = 1)
        target_q_values = self.target_network(next_states)[np.arange(0, self.batch_size), best_action]
        # The rewards of any future states don't matter if the current state is a terminal state
        # If done is true, then 1 - done is 0, so the part after the plus sign (representing the future rewards) is 0
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_counter += 1
        self.decay_exploration_rate()


        