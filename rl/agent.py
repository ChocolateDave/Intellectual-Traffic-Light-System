# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# -----------------------
# Project: RL agents
# Author: Juanwu Lu
# Date: 5 Sep. 2020
# -----------------------

import os
import sys
import time
from copy import deepcopy
from inspect import getmembers

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .memory import ReplayBuffer
from .nnet import OWM_DQN, RLT, Dueling_DQN, Naive_DQN
from .summary import summary

sys.path.append("./project/")

def class_vars(classobj):
    r"""从配置类中循环调用相关参数
    Recursively retrieve class members and their values.
    :Param
        classobj (class): the target class object.
    :Return
        class_vars (dict): class members and their values.
    """
    return {k: v for k, v in getmembers(classobj)
            if not k.startswith('__') and not callable(k)}

class BaseAgent(object):
    r"""深度强化学习智能体基础类
    Arbitrary class of RL agent.
    """

    def __init__(self, config, env):
        # Environment parameter
        self.env = env
        self.action_space = env.action_size
        self.state_dim = env.observation_space
        # Agent parameter
        self.attrs = class_vars(config)
        self.batch_size = config.batch_size
        self.ckpt = config.checkpoint
        self.double = config.double
        self.eps = config.epsilon_start
        self.eps_min = config.epsilon_end
        self.eps_decay = config.epsilon_decay
        self.gamma = config.gamma
        self.learn_init = config.learn_initial
        self.lr = config.lr
        self.loss_func = nn.MSELoss()
        if config.prioritized:
            pass
        else:
            self.mem = ReplayBuffer(
                self.state_dim, self.action_space, config.memory_size)
        self.tau = config.sync_tau
        self.timestep = 0
        self.total_rewards = []
        self.writer = SummaryWriter()

    def choose_action(self, obs):
        r"""智能体动作决策函数，采用贪心算法
        Main function to determine action by ε-greedy algorithm.
        Params:
            obs (tensor): current state.
        Return:
            action (int): action to take.
        """
        raise NotImplementedError

    def epsilon_decrementer(self):
        r"""贪婪算法下动作探索概率的迭代
        Update probability of random action by step.
        """
        self.writer.add_scalar("Interaction/epsilon", self.eps, self.timestep)
        self.eps = self.eps - self.eps_decay \
            if self.eps > self.eps_min else self.eps_min

    def interact(self):
        r"""智能体与环境交互主函数，存储记忆并决定是否自我训练
        Main function to determine interaction policies, store transition and train.
        """
        #assert isinstance(self.currentState,(T.tensor, np.array)), "Didn't initialize env"
        # interact with the env
        action = self.choose_action(self.currentState)
        state, reward, done, _ = self.env.step(action)
        self.epsilon_decrementer()
        self.mem.add(self.currentState, action, reward, state, done)
        if self.timestep > self.learn_init:
            self.train()
            self.total_rewards.append(reward)
            self.writer.add_scalar("Interaction/step_reward",
                        reward, self.timestep)
            self.writer.add_scalar("Interaction/smoothed_reward",
                        np.mean(self.total_rewards[-100:]), self.timestep)
            if self.timestep % self.ckpt == 0:
                self._save()
        if done:
            state = self.env.reset()
        self.currentState = state
        self.timestep += 1

    def sample_exp(self):
        r"""经验池采样器
        Sampling minibatches from memory.
        Return:
            states (tensor): previous states tensor.
            actions (tensor): actions tensor.
            rewards (tensor): retrived rewards tensor.
            states_ (tensor): next states tensor.
            dones (tensor): indicators of terminal.
        """
        state, action, reward, state_, done = self.mem.sample(self.batch_size)
        states = T.tensor(state).to(self.device)
        actions = T.tensor(action).to(self.device)
        rewards = T.tensor(reward).to(self.device)
        states_ = T.tensor(state_).to(self.device)
        dones = T.tensor(done).to(self.device)
        return states, actions, rewards, states_, dones

    def sync(self):
        r"""训练中定期更新目标Q网络
        Sync the target model by specific frequency.
        """
        self.tgt_q.load_state_dict(self.q.state_dict())

    def train(self):
        r"""智能体训练主函数
        Main function for training the agent:
        a) check if memory satisfies minimum batch size.
        b) synchronize target network with preset frequency. 
        c) calculate q values and corresponding loss.
        d) back propagate loss by chain rule.
        """
        raise NotImplementedError

    def _save(self):
        r"""训练状态保存函数
        Save network state for latent resume.
        """
        status = {'timestep': self.timestep, 'eval_net': self.q.state_dict(),
            'tgt_net': self.tgt_q.state_dict(), 'optim': self.optimizer.state_dict()}
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        T.save(status, os.path.join(self.ckpt_dir, 'checkpoint.ckpt'))

    def _load(self):
        r"""训练恢复加载器
        Load network state and resume training.
        """
        # Search for the latest motified network checkpoint
        dir_list = os.listdir("./checkpoints")
        if not dir_list:
            return
        else:
            dir_list = sorted(dir_list,\
                key=lambda x: os.path.getmtime(os.path.join("./checkpoints", x)))
        ckpt_dir = os.path.join("./checkpoints", dir_list[0])
        # Load network state
        ckpt = T.load(os.path.join(ckpt_dir, 'checkpoint.ckpt'))
        self.q.load_state_dict(ckpt['eval_net'])
        self.tgt_q.load_state_dict(ckpt['tgt_net'])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.timestep = ckpt['timestep'] + 1

    @property
    def ckpt_dir(self):
        return os.path.join('./checkpoints', self.model_dir)

class DQNAgent(BaseAgent):
    def __init__(self, config, env):
        super(DQNAgent, self).__init__(config, env)
        # Build nns
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.q = Naive_DQN(self.state_dim, self.action_space).to(self.device)
        self.tgt_q = deepcopy(self.q)
        print("Network initialized! Status reporting ...")
        summary(self.q, tuple(self.state_dim), device=self.device)
        self.model_dir = "%s-%s-%s-%s/" % (os.environ['USER'], 'tl_v1',\
            'pytorch', self.q.name)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        self.writer.add_graph(self.q, T.zeros(1, *self.state_dim).to(self.device))
        if os.path.exists("./checkpoints"):
            self._load()

    def choose_action(self, obs):
        if np.random.random() > self.eps:
            state = T.tensor([obs], dtype=T.float).to(self.device)
            actions = self.q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.randint(0, self.action_space)
        return action

    def train(self):
        # check if satisfy minimum batches
        if self.mem.mem_cntr < self.batch_size:
            return
        # synchronize target network
        if self.timestep % self.tau == 0:
            self.sync()
        s, a, r, s_, dones = self.sample_exp()
        # calculate Q values and corresponding loss
        q_pred = self.q.forward(s).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        q_next = self.tgt_q.forward(s_).max(1)[0]
        q_next[dones] = 0.0
        q_target = r + self.gamma*q_next.detach()
        self.optimizer.zero_grad()
        loss = self.loss_func(q_pred, q_target).to(self.device)
        loss.backward()
        self.writer.add_scalar("Train/loss", loss.item(), self.timestep)
        self.optimizer.step()

class DuelingAgent(BaseAgent):
    def __init__(self, config, env):
        super(DuelingAgent, self).__init__(config, env)
        # Build nn
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.q = Dueling_DQN(self.state_dim, self.action_space).to(self.device)
        self.tgt_q = deepcopy(self.q)
        print("Network initialized! Status reporting ...")
        summary(self.q, tuple(self.state_dim), device=self.device)
        self.model_dir = "%s-%s-%s-%s/" % (os.environ['USER'], 'tl_v1',\
            'pytorch', self.q.name)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        self.writer.add_graph(self.q, T.zeros(1, *self.state_dim).to(self.device))
        if os.path.exists("./checkpoints"):
            self._load()
    
    def choose_action(self, obs):
        if np.random.random() > self.eps:
            state = T.tensor([obs], dtype=T.float).to(self.device)
            actions = self.q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def train(self):
        # check if satisfy minimum batches
        if self.mem.mem_cntr < self.batch_size:
            return
        # synchronize target network
        if self.timestep % self.tau == 0:
            self.sync()
        s, a, r, s_, dones = self.sample_exp()
        # calculate Q values and corresponding loss
        q_pred = self.q.forward(s).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        q_next = self.tgt_q.forward(s_).max(1)[0]
        q_next[dones] = 0.0
        q_target = r + self.gamma*q_next.detach()
        self.optimizer.zero_grad()
        loss = self.loss_func(q_pred, q_target).to(self.device)
        loss.backward()
        self.writer.add_scalar("Train/loss", loss.item(), self.timestep)
        self.optimizer.step()

class OWMAgent(BaseAgent):
    pass

class RLTAgent(BaseAgent):
    def __init__(self, config, env):
        super(RLTAgent, self).__init__(config, env)
        # Build nn
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.q = RLT(self.state_dim, config.depth, self.action_space).to(self.device)
        self.tgt_q = deepcopy(self.q)
        print("Network initialized! Status reporting ...")
        summary(self.q, tuple(self.state_dim), device=self.device)
        self.model_dir = "%s-%s-%s-%s/" % (os.environ['USER'], 'tl_v1',\
            'pytorch', self.q.name)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        self.writer.add_graph(self.q, T.zeros(1, *self.state_dim).to(self.device))
        if os.path.exists("./checkpoints"):
            self._load()

    def choose_action(self, obs):
        if np.random.random() > self.eps:
            state = T.tensor([obs], dtype=T.float).to(self.device)
            actions = self.q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def train(self):
        # check if satisfy minimum batches
        if self.mem.mem_cntr < self.batch_size:
            return
        # synchronize target network
        if self.timestep % self.tau == 0:
            self.sync()
        s, a, r, s_, dones = self.sample_exp()
        # calculate Q values and corresponding loss
        q_pred = self.q.forward(s).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        q_next = self.tgt_q.forward(s_).max(1)[0]
        q_next[dones] = 0.0
        q_target = r + self.gamma*q_next.detach()
        self.optimizer.zero_grad()
        loss = self.loss_func(q_pred, q_target).to(self.device)
        loss.backward()
        self.writer.add_scalar("Train/loss", loss.item(), self.timestep)
        self.optimizer.step()
