# !/usr/bin/env python
# -*- conding:utf-8 -*-
import numpy as np

class ReplayBuffer(object):
    r"""基础的强化学习回放单元
    A rudimentary replay buffer class for reinforcement learning. 
    The main api of this buffer are:
        :store_trasition - store transition tuple after every step.
        :sample - sample batches for training.
    """
    def __init__(self, input_shape, n_actions, memory_size):
        #memo counter
        self.mem_cntr = 0
        #exp size
        self.mem_size = memory_size
        # self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def add(self, s, a, r, s_, done):
        """Main function to store experience.
        Params:
            s (tensor): current or previous state.
            a (tensor): action value.
            r (tensor): reward value.
            s_ (tensor): next state.
            done (bool): indicator of env status.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = s
        self.next_state_memory[index] = s_
        self.reward_memory[index] = r
        self.terminal_memory[index] =  done
        self.action_memory[index] = a
        self.mem_cntr += 1

    def sample(self, batch_size):
        """Sampler for batch gradient algorithm.
        Param:
            batch_size (int): the sfdeize of batch set.
        Return:
            states (tensor): current or previous states.
            acions (tensor): action values.
            rewards (tensor): reward values.
            states_ (tensor): next states.
            terminals (bool): indicators of env status.
        """
        maxmem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(maxmem, batch_size)
        states = self.state_memory[batch]
        states_ = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminals = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminals

class PrioritizedReplayBuffer(object):
    def __init__(self):
        pass