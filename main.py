# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# -----------------------
# Project: Main
# Author: Juanwu Lu
# Date: 19 Sep. 2020
# -----------------------

import argparse
from rl.nnet import OWM_DQN
from rl.agent import DQNAgent, DuelingAgent, RLTAgent
from rl.config import AgentConfigs
from env.wrapped import tl_v1 as Env

def main(args):
    r"""智能信号灯系统主控函数
    Master control function for ITLS.
    Params:
        args: argument parsers.
    """

    # 第一步：初始化系统 Step1: Initialize system.
    env = Env(path=args.envpth)
    if args.appr == 'dueling':
        agent = DuelingAgent(AgentConfigs, env)
    elif args.appr == 'owm':
        agent = None
        pass
    elif args.appr == 'rlt':
        agent = RLTAgent(AgentConfigs, env)
    else:
        agent = DQNAgent(AgentConfigs, env)
    # 第二步：循环训练 Step2: Recursively training.
    agent.currentState = env.reset()
    while True:
        agent.interact()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--appr', default='rlt', type=str, required=False, help='Name of training approach')
    parser.add_argument('--envpth', default='./env/', type=str, required=False, help='Path to environment')
    main(parser.parse_args())
