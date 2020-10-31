# !/usr/bin/env python
# -*- conding:utf-8 -*-
from operator import index
from pickle import STRING
from typing import List, Tuple, Type, Union

from torch._C import dtype

import numpy as np
import torch as T
from torch.nn.functional import batch_norm

class BaseBuffer(object):
    def __init__(self, name:str) -> None:
        self.name = name
    
    def add(self, s:Type[T.Tensor], a:Type[T.Tensor], r:Type[T.Tensor], 
                s_:Type[T.Tensor]=None, done:Type[T.bool]=False) -> None:
        """Main function to store experience.
        Params:
            s (tensor): current or previous state.
            a (tensor): action value.
            r (tensor): reward value.
            s_ (tensor): next state.
            done (bool): indicator of env status.
        """
        raise NotImplementedError
    
    def sample(self, batch_size:int) -> tuple:
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
        raise NotImplementedError

# ----------------------------------------------------------

class ReplayBuffer(BaseBuffer):
    r"""基础的强化学习回放单元
    A rudimentary replay buffer class for reinforcement learning. 
    The main api of this buffer are:
        :store_trasition - store transition tuple after every step.
        :sample - sample batches for training.
    """
    def __init__(self, input_shape:Type[Union[List, Tuple]], memory_size:int) -> None:
        super(ReplayBuffer, self).__init__("Standard_Replay_Buffer")
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
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = s
        self.next_state_memory[index] = s_
        self.reward_memory[index] = r
        self.terminal_memory[index] =  done
        self.action_memory[index] = a
        self.mem_cntr += 1

    def sample(self, batch_size):
        maxmem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(maxmem, batch_size)
        states = self.state_memory[batch]
        states_ = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminals = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminals

# ----------------------------------------------------------

class SumTree(object):
    r"""构建基于和的二叉树结构的具优先度存储结构
    Build a sum tree edifice to store the priorities.
    """
    def __init__(self, in_dim, mem_size:int) -> None:
        Transition_dtype = np.dtype(
            [('timestep', np.int32), ('state', np.uint8, (in_dim[-2], in_dim[-1])), 
            ('action', np.int32), ('reward', np.float32), ('notdone', np.bool_)]
            )
        blank_trans = (0, np.zeros(*in_dim, dtype=np.flot32), 0, 0.0, False)
        self.index = 0
        self.size = mem_size
        self.full = False  # used to track actual capacity
        self.tree_start = 2**(mem_size-1).bit_length()-1 # put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float)
        self.data = np.array([blank_trans]*mem_size, dtype=Transition_dtype) # build memory array
        self.max = 1 # initial max value of value

    # Updates nodes values from current tree
    def update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1,2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)
    
    # Propagates changes up tree 
    def propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self.update_nodes(unique_parents)
        if parents[0] != 0:
            self.propagate(parents) # recursively update
    
     # Propagates single value given a specific index for efficiency
    def propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self.propagate_index(parent)
    
    # Updates values given tree indices globally
    def update(self, indices, values):
        self.sum_tree[indices] = values
        self.propagate(indices) # propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)
    
    # Updates single value given a specific index for efficiency
    def update_index(self, index, value):
        self.sum_tree[index] = value
        self.propagate_index(index)
        self.max = max(value, self.max)
    
    def append(self, data, value):
        self.data[self.index] = data
        self.update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)
    
    # Search and retrieve location of values in the sum tree
    def retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1,2], axis=1))
        if children_indices[0, 0] >= self. sum_tree.shape[0]:
            return indices
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)
        successor_indices = children_indices[successor_choices, np.arange(indices.size)]
        successor_values = values - successor_choices * left_children_values
        return self.retrieve(successor_indices, successor_values)
    
    # Search for values in sum tree and return values, data indices
    def find(self, values):
        indices = self.retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)
    
    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    # Return current total value
    def total(self):
        return self.sum_tree[0]

class PrioritizedReplayBuffer(BaseBuffer):
    def __init__(self, alpha:float, beta:float, device:Type[T.device], discout:float,
            in_dim:Type[Union[List, Tuple]], mem_size:int, multi_step:int) -> None:
        super(PrioritizedReplayBuffer, self).__init__("Prioritiezed_Replay_Buffer")
        self.device = device
        self.history = 1 # number of consecutive states processed simultaneously
        self.mem_size = mem_size
        self.n = multi_step
        self.alpha = alpha # priority_exponent
        self.beta = beta # initial priority_weight
        self.discout = discout # reward discout factor
        self.t = 0
        self.n_step_scaling = T.tensor(
            [self.discout ** i for i in range(self.n)], dtype=T.float32, device=self.device
            ) # discount-scaling vector for n-step returns
        self.transitions = SumTree(in_dim, mem_size)

    def add(self, s, a, r, done):
        self.transitions.append((self.t, s, a, r, not done), self.transitions.max) # give new memory maximum value to exploit
        self.t = 0 if done else self.t + 1
    
    # Returns the transitions with blank states where appropriate
    def _get_transitions(self, idxs):
        trans_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
        trans = self.transitions.get(trans_idxs)
        trans_firsts = trans['timestep'] == 0
        blank_msk = np.zeros_like(trans_firsts, dtype=np.bool_)
        for t in range(self.history - 2, -1, -1):
            blank_msk[:, t] = np.logical_or(blank_msk[:, t+1], trans_firsts[:, t+1])
        for t in range(self.history, self.history + self.n):
            blank_msk[:, t] = np.logical_or(blank_msk[:, t-1], trans_firsts[:, t])
        trans[blank_msk] = blank_msk
        return trans
    
    # Returns a valid sample from each segment
    def _get_sample_from_segments(self, batch_size:int, p_total:Union[int, float]):
        seg_len = p_total / batch_size
        seg_starts = np.arange(batch_size)
        valid = False
        while not valid:
            sample = np.random.uniform(0.0, seg_len, [batch_size]) + seg_starts
            probs, idxs, tree_idxs = self.transitions.find(sample)
            if np.all((self.transitions.index - idxs)) % self.mem_size > self.n) and \
                np.all((idxs - self.transitions.index) % self.mem_size >= self.history) and \
                    np.all(probs != 0):
                valid = True
        # retrieve all required data from t-h to t+n
        trans = self._get_transitions(idxs)
        
        # create un-discretised states and nth next states
        all_states = trans['state']
        s = T.tensor(all_states[:, :self.history], device=self.device, dtype=T.float32)
        s_ = T.tensor(all_states[:, self.n:self.n+self.history], device=self.device, dtype=T.float32)
        # discrete actions to be used as index
        a = T.tensor(np.copy(trans['action'][:, self.history-1], dtype=T.int64, device=self.device))
        # calculate truncated n-step discouted return R^{n}=Σ_{k=0}γ^{k}R_{t+k+1}
        r = T.tensor(no.copy(trans['reward'][:, self.history-1:-1]), dtype=T.float32, device=self.device)
        r = T.matmul(r, self.n_step_scaling)
        # Mask for done nth next states
        n_done = T.tensor(np.expand_dims(trans['notdone'][:, self.history+self.n-1], axis=1), dtype=T.float32, device=self.device)

        return probs, idxs, tree_idxs, s, a, r, s_, n_done

    def sample(self, batch_size:int):
        p_total = self.transitions.total()
        probs, idxs, tree_idxs, s, a, r, s_, n_done = self._get_sample_from_segments(batch_size, p_total)
        probs = probs / p_total # calculate normalized probabilities
        mem_size = self.mem_size if self.transitions.full else self.transitions.index
        w = (mem_size * probs) ** (-self.beta)
        w = T.tensor(w / w.max(), dtype=T.float32, device=self.device) # normalization
        
        return tree_idxs, s, a, r, s_, n_done, w
    
    def update(self, idxs:int, ps):
        ps = np.power(ps, self.alpha)
        self.transitions.update(idxs, ps)
    
    # set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self
    
    # return valid states for validation
    def __next__(self):
        if self.current_idx == self.mem_size:
            raise StopIteration
        trans = self.transitions.data[np.arange(self.current_idx-self.history+1, self.current_idx+1)]
        trans_firsts = trans['timestep'] == 0
        blnk_mask = np.zeros_like(trans_firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            blnk_mask[t] = np.logical_or(blnk_mask[t + 1], trans_firsts[t + 1]) # If future frame has timestep 0
        trans[blnk_mask] = blnk_mask
        state = T.tensor(trans['state'], dtype=T.float32, device=self.device)  # Agent will turn into batch
        self.current_idx += 1
        return state
