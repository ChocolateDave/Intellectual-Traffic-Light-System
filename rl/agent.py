# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# -----------------------
# Project: RL agents
# Author: Juanwu Lu
# Date: 5 Sep. 2020
# -----------------------

import bz2
import os
import pickle
import sys
from copy import deepcopy
from inspect import getmembers

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .memory import PrioritizedReplayBuffer, ReplayBuffer
from .nnet import RLT, Dueling_DQN, Naive_DQN, ReITA
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
            if done:
                self.writer.add_scalar("Interaction/cumulative_reward",
                        np.sum(self.total_rewards), self.timestep)
                self.total_rewards.clear()
            if self.timestep % self.ckpt == 0:
                self._save()
                self._save_mem()
        if done:
            state, done = self.env.reset(), False
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

    def _save_mem(self, disable_bzip=False):
        mem_pth = os.path.join(self.ckpt_dir, 'mem.p')
        if disable_bzip:
            with open(mem_pth, 'wb') as _pickle:
                pickle.dump(self.mem, _pickle)
        else:
            with bz2.open(mem_pth, 'rb') as zipped_pickle:
                pickle.dump(self.mem, zipped_pickle)

    def _load_mem(self, disable_bzip=False):
        mem_pth = os.path.join(self.ckpt_dir, 'mem.p')
        if disable_bzip:
            with open(mem_pth, 'rb') as _pickle:
                self.mem = pickle.load(_pickle)
        else:
            with bz2.open(mem_pth, 'rb') as zipped_pickle:
                self.mem = pickle.load(zipped_pickle)
    
    @property
    def ckpt_dir(self):
        return os.path.join('./checkpoints', self.model_dir)

# ----------------------------------------------------------

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

# ----------------------------------------------------------

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

# ----------------------------------------------------------

class RLTAgent(BaseAgent):
    def __init__(self, config, env):
        super(RLTAgent, self).__init__(config, env)
        # Build nn
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.model_dir = "%s-%s-%s-%s/" % (os.environ['USER'], 'tl_v1',\
            'pytorch', self.q.name)
        
        self.q = RLT(self.state_dim, config.depth, config.h_features, self.action_space).to(self.device)
        if os.path.exists("./checkpoints"):
            self._load()
            self._load_mem()
        self.q.train()
        self.tgt_q = deepcopy(self.q)
        for param in self.tgt_q.parameters():
            param.requires_grad = False
        print("Network initialized! Status reporting ...")
        summary(self.q, tuple(self.state_dim), device=self.device)
        
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr, eps=config.optm_eps)
        self.writer.add_graph(self.q, T.zeros(1, *self.state_dim).to(self.device))

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

# ----------------------------------------------------------

class ReitAgent(BaseAgent):
    def __init__(self, config, env):
        super(RLTAgent, self).__init__(config, env)

        # Training params
        self.beta_increase = (1-config.prio_beta) / config.prio_frame
        self.discount = config.gamma
        self.vmin, self.vmax = config.vmin, config.vmax
        self.mem = PrioritizedReplayBuffer(config.prio_alpha, config.prio_beta, 
                self.device, self.discount, self.state_dim, config.memory_size, config.multi_step)
        self.support = T.linspace(self.vmin, self.vmax, config.atoms).to(self.device)
        self.delta_o = (config.vmax - config.vmin) / (config.atoms - 1)
        self.n = config.multi_step
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr, eps=config.optm_eps)
        self.writer.add_graph(self.q, T.zeros(1, *self.state_dim).to(self.device))

        # Build nn
        self.atoms = config.atoms
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.model_dir = "%s-%s-%s-%s/" % (os.environ['USER'], 'tl_v1',\
            'pytorch', self.q.name)
        
        self.q = ReITA(
            self.state_dim, config.depth, self.atoms, config.h_features, config.noise_std, self.action_space
            ).to(self.device)
        if os.path.exists("./checkpoints"):
            self._load()
            self._load_mem()
        self.q.train()
        self.tgt_q = deepcopy(self.q)
        for param in self.tgt_q.parameters():
            param.requires_grad = False
        print("Network initialized! Status reporting ...")
        summary(self.q, tuple(self.state_dim), device=self.device)
                
    def choose_action(self, obs):
        with T.no_grad():
            action = (self.q(obs.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
        return action
    
    def reset_noise(self):
        self.q.reset_noise()

    def train(self):
        # check if satisfy minimum batches
        if self.mem.mem_cntr < self.batch_size:
            return
        # synchronize target network
        if self.timestep % self.tau == 0:
            self.sync()
        # annal importance sampling weight
        self.mem.beta = min(self.mem.beta+self.beta_increase, 1)
        # draw new set of noisy weights
        self.q.reset_noise()
        # sample transistions
        idxs, s, a, r, s_, notdone, w = self.mem.sample(self.batch_size)
        # calculate current state probabilities
        log_ps = self.q.forward(s, log=True)
        log_ps_a = log_ps[range(self.batch_size), a]
        
        with T.no_grad():
            # calculate nth next state probabilities
            pns = self.q.forward(s_)
            dns = self.support.expand_as(pns) * pns
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.tgt_q.reset_noise()
            pns = self.tgt_q.forward(s_)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # calculate Tz (Bellman operator T applied to z)
            Tz = r.unsqueeze(1) + notdone * (self.discount ** self.n) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.vmin, max=self.vmax)
            # calculate l2 projection of Tz on to fixed support z
            b = (Tz - self.vmin) / self.delta_o
            l, u = b.floor().to(T.int64), b.ceil.to(T.int64)
            # fix disappearing probability mass when l = b = u
            l[(u>0) & (l==u)] -= 1
            u[(l<(self.atoms-1)) & (l==u)] += 1

            # distribute probability of Tz
            m = s.new_zeros(self.batch_size, self.atoms)
            offset = T.linspace(0, ((self.batch_size-1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(a)
            m.view(-1).index_add_(0, (l+offset).view(-1), (pns_a*(u.float()-b)).view(-1))
            m.view(-1).index_add_(0, (u+offset).view(-1), (pns_a*(b-l.float())).view(-1))
        
        # define kl-loss
        loss = -T.sum(m*log_ps_a, 1)
        self.q.zero_grad()
        (w*loss).mean().backward()
        # TODO: clip gradient.
        self.optimizer.step()
        # update priorities of sampled transitions with kl-loss (instead of TD-loss)
        self.mem.update(idxs, loss.detach().cpu().numpy()) 