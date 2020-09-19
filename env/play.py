# !/usr/bin/env python
# -*- coding: utf-8 -*-

#---------------------------------
# Project: simulation interaction
# Author: Juanwu Lu
# Date: 5 Sep. 2020
#---------------------------------

import matplotlib.pyplot as plt
import numpy as np
from wrapped_traffic_light import tl_v1

env = tl_v1()
obs = env.reset()
print('observation space:',obs.shape)
plt.ion()
f = False
while not f:
    plt.cla()
    plt.imshow(obs[0], cmap='gray', origin='lower')
    action = input("Enter the next action:")
    if not (isinstance(action, int) and action < env.action_size):
        print("Invalid input, choose random action")
        action = np.random.randint(0, env.action_size)
    obs, r, f, info = env.step(int(action))
    print("CurrentPhase:", info['cp'], '\t', "reward:",r)
    plt.pause(0.1)
plt.ioff()