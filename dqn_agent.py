import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from dqn_cnn import DQNCNN


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.astype(np.uint8),
            action,
            reward,
            next_state.astype(np.uint8),
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.uint8),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.uint8),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQNCNN(n_actions).to(self.device)
        self.target_net = DQNCNN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=5e-5)
        self.criterion = nn.SmoothL1Loss()

        # 🔥 AJUSTE PARA LOCAL (MENOS MEMORIA)
        self.memory = ReplayBuffer(20000)

        self.batch_size = 32
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995

        self.update_target_every = 2000
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_net.fc[-1].out_features)

        state = np.transpose(state, (2, 0, 1))
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0

        with torch.no_grad():
            q_values = self.q_net(state)

        return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < 1000:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device) / 255.0
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device) / 255.0

        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)

        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        q_values = self.q_net(states).gather(1, actions).squeeze(-1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())