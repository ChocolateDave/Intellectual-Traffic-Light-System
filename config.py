# -*- coding: utf-8 -*-

# This files highlights all the necessary configurations of  the environment
# during training. To adjust the params, make sure you've checked the
# instructions. An inappropriate modification may cause error.
# Note: Do not hashtag any exsisting params!

# Agent configs
# Configs of reinforcement learning agent, which has significant effcts on 
# the final result of the training process. Treat them with cautions.

import xml.dom.minidom
from xml.dom.minidom import parse

class AgentConfigs(object):
    # memory
    memory_size = 10000
    # epsilon-greedy hyperparams
    epsilon_start= 1.0
    epsilon_end= 0.1
    epsilon_frame= 10000
    epsilon_decay = (epsilon_start - epsilon_end)/epsilon_frame
    # network saver
    checkpoint = 1000
    # training hyperparams
    batch_size = 64
    data_format = 'NHWC'
    gamma = 0.99
    learn_initial = 10000
    lr = 0.01
    sync_tau = 1000
    use_gpu = True
    # prioritized replay hyperparams
    prio_alpha = 0.6
    prio_beta = 0.4
    prio_frame = 100000
    # model params
    dueling =  False
    double = False
    prioritized = False
    UPDATE_FRAMES = 1000
    # owm
    owm = False
    owm_alphas = [[0.001,0.6]]
    #record
    SUMMARIES = 'summaries/'
    SAVER = 'output/'
    SAVED_FILE_NAME = '.ckpt'


# Environment configs
# Basic configurations of environments that realate to how the environment
# would interact with the agent. Do pay attention to the meaning of each param.
class EnvConfigs(object):
    downsample = 3
    env_name = 'ITLS-v1'
    frameskip = 25
    reward_range = None
    warm_up_dur = 60
    total_dur = 3600

# Sumo interactive env params.
# See https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html for more info on traci.
# See https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html for more info on actions.
# See http://www.mohurd.gov.cn/wjfb/201903/t20190320_239844.html for more info about vehicle conversions.
class SumoConfigs(EnvConfigs):
    # traffic light
    actions = {
        "None": None,
        "W, E": "srrsrrrgGrgGGGGrrsrrsrrrgGrgGGGGrr",
        "WL, EL": "srrsrrrgrGgrrrrGGsrrsrrrgrGgrrrrGG",
        "W, WL": "srrsrrrsrrsrrrrrrsrrsrrrgGGgGGGGGG",
        "E, EL": "srrsrrrgGGgGGGGGGsrrsrrrsrrsrrrrrr",
        "N, S": "gGrgGGrsrrsrrrrrrgGrgGGrsrrsrrrrrr",
        "NL, SL": "grGgrrGsrrsrrrrrrgrGgrrGsrrsrrrrrr",
        "N, NL": "gGGgGGGsrrsrrrrrrsrrsrrrsrrsrrrrrr",
        "S, SL": "srrsrrrsrrsrrrrrrgGGgGGGsrrsrrrrrr"
    }
    conversions = {
        "NMV": 0.36,
        "P": 1.0,
        "T": 1.2,
        "MP": 2.0,
        "MT": 3.0,
        "LP": 2.5,
        "LT": 4.0
    }
    # model details
    edges, ent_edges, ext_edges, lanearea = [], [], [], []
    with parse("./project/ITLS.net.xml") as tree:
        es = tree.documentElement.getElementsByTagName("edge")
        for e in es:
            _id = e.getAttribute("id")
            edges.append(_id)
            if _id[-1] == 'E':
                ent_edges.append(_id)
            if _id[-1] == 'X':
                ext_edges.append(_id)
    with parse("./project/ITLS.add.xml") as tree:
        es = tree.documentElement.getElementsByTagName("laneAreaDetector")
        for e in es:
            lanearea.append(e.getAttribute("id"))
    # observation
    shape = [-110, 68, -10, 210]
    width = shape[1] - shape[0]
    height = shape[3] - shape[2]
    # sumo binary
    sumo_binary = "sumo"
    sumo_config = "./project/ITLS.sumo.cfg"
    TLid = "ITLS"

# Aggregate DQN params.
class DQNConfigs(AgentConfigs, SumoConfigs):
    backend = 'tf'
    env_type = 'detailed'
    model = 'DQN'


# Configs calling function.
def get_config():
    config = DQNConfigs
    return config