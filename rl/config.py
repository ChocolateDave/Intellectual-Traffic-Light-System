# -*- coding: utf-8 -*-

# This files highlights all the necessary configurations of  the environment
# during training. To adjust the params, make sure you've checked the
# instructions. An inappropriate modification may cause error.
# Note: Do not hashtag any exsisting params!

# Agent configs
# Configs of reinforcement learning agent, which has significant effcts on 
# the final result of the training process. Treat them with cautions.

class AgentConfigs(object):
    # memory
    memory_size = 50000
    # epsilon-greedy hyperparams
    epsilon_start= 1.0
    epsilon_end= 0.1
    epsilon_frame= 50000
    epsilon_decay = (epsilon_start - epsilon_end)/epsilon_frame
    # network saver
    checkpoint = 1000
    # training hyperparams
    batch_size = 64
    data_format = 'NCHW'
    gamma = 0.99
    learn_initial = 50000
    lr = 0.01
    multi_step = 1
    optm_eps = 1e-8
    sync_tau = 1000
    # prioritized replay hyperparams
    prio_alpha = 0.6
    prio_beta = 0.4
    prio_frame = 50000
    # model params
    atoms = 51
    double = False
    h_features = 1000
    noise_std = 0.1
    prioritized = False
    vmin, vmax = 0, 100
    # resnet
    depth = [3, 4, 6, 3]
    #record
    SUMMARIES = 'summaries/'
    SAVER = 'output/'
    SAVED_FILE_NAME = '.ckpt'
