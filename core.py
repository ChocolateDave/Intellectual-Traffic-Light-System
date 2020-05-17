# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ITLS master control
Author: Juanwu Lu
Facility: Tongji University
"""
import sys
sys.path.append("project/")
from brain import Brain
from config import DQNConfigs,SumoConfigs
import env as Env
import numpy as np 

def main():
    r"""智能信号灯系统主控函数
    Master control function for ITLS.
    """
    # 第一步：初始化系统 Step1: initialize system.
    env = Env.TrafficLight_v0(DQNConfigs)
    brain = Brain(DQNConfigs, env)
    obs, _, _, _ = env.reset()
    brain.currentState = obs
    # 第二步：实时交互与训练 Step2: play and train.
    while True:
        action = brain.get_action()
        state, reward, terminal = env.step(action)
        brain.interact(state, action, reward, terminal)

if __name__ == '__main__':
    main()
