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

with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)

# 把所有要显示的参数聚集在一起
PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

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
        LOSS,REWARD = brain.interact(state, action, reward, terminal)
        # 保存训练结果 Save network with certain frequency.
        if brain.timestep % brain.config.checkpoint == 0:
            summ = brain.sess.run(PERFORMANCE_SUMMARIES,
                                 feed_dict={LOSS_PH: LOSS,
                                            REWARD_PH: REWARD})
            brain.writer.add_summary(summ, brain.timestep)
        #loss = sess.run([brain.loss],feed_dict={brain.a:action})
        #print(reward,brain.loss,brain.timestep)

if __name__ == '__main__':
    main()
