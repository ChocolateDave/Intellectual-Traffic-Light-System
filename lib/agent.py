# !/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
import tensorflow as tf
if int(tf.__version__[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


class BaseAgent(object):
    r"""Abstract Agent interface."""
    def initial_state(self):
        """Should create initial empty state for the agent. It will be called 
        for the start of the episode.

        Return: 
            Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """Convert observations and states into actions to take.
        
        Param:
            states (list): environment states to process.
            agent_states (list): states with the same length as observations.
        
        Return: 
            tuple of actions, states.
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def default_states_preprocessor(states):
    """Convert list of states into the form suitable for model. 
    By default we assume Variable.

    Param:
        states (list): numpy arrays with states
    
    Return:
        class tf.variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return tf.convert_to_tensor(np_states)


def float32_preprocessor(states):
    """Preconvert list of states into numpy array with dtype as np.float32 if needed.
    
    Param:
        states (list): numpy arrays with states.

    Return:
        class tf.variable
    """
    np_states = np.array(states, dtype=np.float32)
    return tf.convert_to_tensor(np_states)


class DQNAgent(BaseAgent):
    r"""DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector.
    """
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if tf.is_tensor(states):
                states = states.to(self.device)
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states


class TargetNet(object):
    """Wrapper around model which provides copy of it instead of trained weights.
    See https://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/model-based-RL-deepmind.pdf
    fo more information.
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """Blend params of target net with params from the model
        
        Param: 
            alpha (float): blending factor.
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)

if __name__ == "__main__":
    agent = DQNAgent()