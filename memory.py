# !/usr/bin/env python
# -*- conding:utf-8 -*-
import numpy as np

class ReplayBuffer(object):
    r"""A basic replay buffer class which stores (s, a, r, s_) for reinforcement 
    learning. See more at "https://www.youtube.com/watch?v=5fHngyN8Qhw&t=94s".

    The main api of this buffer are:
        :store_trasition - store transition tuple after every step.
        :sample - sample batches for training.
    """
    def __init__(self, input_shape, n_actions, memory_size, discrete=False):
        #memo counter
        self.mem_cntr = 0
        #exp size
        self.mem_size = memory_size
        # self.discrete = discrete
        if isinstance(input_shape, int):
            self.state_memory = np.zeros([self.mem_size, input_shape])
            self.next_state_memory =  np.zeros([self.mem_size, input_shape])
        else:
            state_shape=[self.mem_size]
            for size in input_shape:
                state_shape.append(size)
            self.state_memory = np.zeros(input_shape)
            self.next_state_memory = np.zeros(state_shape)
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros([self.mem_size, n_actions])
        self.reward_memory = np.zeros(self.mem_size)
        #whether finish
        self.terminal_memory =  np.zeros(self.mem_size, dtype=np.float32)

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
        self.terminal_memory[index] =  1 - int(done)  # if not done, continue.
        # if self.discrete:
        #     # use one-hot encode for discrete action space.
        #     actions = np.zeros(self.action_memory.shape[1])
        #     actions[a] = 1
        #     self.action_memory[index] = actions
        # else:
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

