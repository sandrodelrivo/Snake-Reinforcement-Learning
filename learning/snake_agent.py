import torch
from learning.snake_net import ActorNet, CriticNet
import numpy as np
import random
from collections import deque

def arr_to_tensor(x): return torch.from_numpy(x).float()

class Memory():

    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)


class SnakeAgent:

    # basic framework from the pytorch implementation

    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.total_reward = 0
        self.curr_step = 0

        self.gamma = 0.98

        self.actor_net = ActorNet(self.state_dim, self.action_dim)
        self.critic_net = CriticNet(self.state_dim)

        a_checkpoint = torch.load("./actor_net.chkpt")
        self.actor_net.load_state_dict(a_checkpoint, strict=False)

        c_checkpoint = torch.load("./critic_net.chkpt")
        self.critic_net.load_state_dict(c_checkpoint, strict=False)

        self.act_opt = torch.optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.critic_opt = torch.optim.Adam(self.critic_net.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.critic_loss_fn = torch.nn.MSELoss()

        self.learn_every = 10
        self.memory = Memory()
        self.save_every = 100000

    def act(self, state):
        probs = self.actor_net.forward(arr_to_tensor(state))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        # increment step
        self.curr_step += 1
        return dist, action

    def update_critic(self, advantage):

        critic_loss = advantage.pow(2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return critic_loss

    def update_actor(self, advantage):

        actor_loss = (-torch.stack(self.memory.log_probs) * advantage.detach()).mean()
        self.act_opt.zero_grad()
        actor_loss.backward()
        self.act_opt.step()

    def train(self, q_val):

        values = torch.stack(self.memory.values)
        q_vals = np.zeros((len(self.memory), 1))

        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + self.gamma * q_val * (1.0 - done)
            q_vals[len(self.memory) - 1 - i] = q_val  # store values from the end to the beginning

        advantage = torch.Tensor(q_vals) - values

        self.update_critic(advantage)
        self.update_actor(advantage)

    def learn(self, obs):

        if self.curr_step % self.save_every == 0:
            self.save()

        state, next_state, action, reward, done, dist = obs

        target = self.critic_net.forward(arr_to_tensor(state))

        self.total_reward += reward

        self.memory.add(dist.log_prob(action), target, reward, done)

        if self.curr_step % self.learn_every == 0:
            last_q_val = self.critic_net.forward(arr_to_tensor(next_state)).detach().data.numpy()
            self.train(last_q_val)
            self.memory.clear()

        return 0, 0

    def save(self):
        act_path = (
                self.save_dir / f"actor_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        crit_path = (
                self.save_dir / f"critic_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(self.actor_net.state_dict(), act_path)
        torch.save(self.critic_net.state_dict(), crit_path)

        print(f"SnakeNet saved to {act_path} at step {self.curr_step}")
