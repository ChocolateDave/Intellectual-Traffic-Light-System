# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class ActionSelector(object):
    r"""Abstract object that converts score to actions.
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    r"""Basic argmax action selector. Choose the action with
    highest score.
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray), ValueError("Network output error")
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    r"""Action selector based on epsilon greedy algorithm.
    With step t, the action of agent is defined as follows:
        generate random number c with np.random
        if c >= epsilon:
            choose the optimal action a_t* with argmax selector
        else:
            randomly select an action from action set.

        See http://home.deib.polimi.it/restelli/MyWebSite/pdf/rl5.pdf 
        for more details.
    """
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions