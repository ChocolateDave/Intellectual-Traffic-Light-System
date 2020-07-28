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
import tensorflow as tf
import numpy as np 


def main():
    r"""智能信号灯系统主控函数
    Master control function for ITLS.
    """
    # 第一步：初始化系统 Step1: initialize system.
    env = Env.TrafficLight_v0(DQNConfigs)
    #create Session
    sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables())
    brain = Brain(DQNConfigs, env,sess)
    obs = env.reset()
    brain.currentState = obs
    # 第二步：实时交互与训练 Step2: play and train.
    while True:
        action = brain.get_action()
        state, reward, terminal,_ = env.step(action)
        if terminal:
            obs = env.reset()
            brain.currentState = obs
        brain.interact(state, action, reward, terminal)

        #loss = sess.run([brain.loss],feed_dict={brain.a:action})
        #print(reward,brain.loss,brain.timestep)

if __name__ == '__main__':
    main()
