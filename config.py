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
    memory_size = 10000
    # epsilon-greedy hyperparams
    epsilon_start= 1.0
    epsilon_end= 0.1
    epsilon_frame= 10000
    epsilon_decay = (epsilon_start - epsilon_end)/epsilon_frame
    # network saver
    checkpoint = 1
    # training hyperparams
    batch_size = 64
    data_format = 'NHWC'
    gamma = 0.99
    learn_initial = 10
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
    env_name = 'ITLS-v0'
    frameskip = 25
    reward_range = None
    warm_up_dur = 60
    total_dur = 3600

# Sumo interactive env params.
# See https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html for more info on traci.
# See https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html for more info on actions.
# See http://www.mohurd.gov.cn/wjfb/201903/t20190320_239844.html for more info about vehicle conversions.
class SumoConfigs(EnvConfigs):
    actions = {
        "None": None,
        "W, E": "grrgrrrgGrgGGGGrrgrrgrrrgGrgGGGGrGrGr",
        "WL, EL": "grrgrrrgrGgrrrrGGgrrgrrrgrGgrrrrGrrrr",
        "W, WL": "grrgrrrgrrgrrrrrrgrrgrrrgGGgGGGGGrrrr",
        "E, EL": "grrgrrrgGGgGGGGGGgrrgrrrgrrgrrrrrrrrr",
        "N, S": "gGrgGGrgrrgrrrrrrgGrgGGrgrrgrrrrrrGrG",
        "NL, SL": "grGgrrGgrrgrrrrrrgrGgrrGgrrgrrrrrrrrr",
        "N, NL": "gGGgGGGgrrgrrrrrrgrrgrrrgrrgrrrrrrrrr",
        "S, SL": "grrgrrrgrrgrrrrrrgGGgGGGgrrgrrrrrrrrr"
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
    edges = [
        ':ITLS_0', ':ITLS_1', ':ITLS_2', ':ITLS_3', ':ITLS_4', ':ITLS_6',
        ':ITLS_33', ':ITLS_34', ':ITLS_35', ':ITLS_36', ':ITLS_7', ':ITLS_8',
        ':ITLS_9', ':ITLS_10', ':ITLS_11', ':ITLS_15', ':ITLS_16', ':ITLS_37',
        ':ITLS_38', ':ITLS_39', ':ITLS_40', ':ITLS_41', ':ITLS_17', ':ITLS_18',
        ':ITLS_19', ':ITLS_20', ':ITLS_21', ':ITLS_23', ':ITLS_42', ':ITLS_43',
        ':ITLS_44', ':ITLS_45', ':ITLS_24', ':ITLS_25', ':ITLS_26', ':ITLS_27',
        ':ITLS_28', ':ITLS_32', ':ITLS_46', ':ITLS_47', ':ITLS_48', ':ITLS_49',
        ':ITLS_c0', ':ITLS_c1', ':ITLS_c2', ':ITLS_c3', ':ITLS_w0', ':ITLS_w1',
        ':ITLS_w2', ':ITLS_w3', ':gneJ19_w0', ':gneJ20_0', ':gneJ20_7', ':gneJ20_w0',
        ':gneJ20_w1', ':gneJ22_0', ':gneJ22_5', ':gneJ22_w0', ':gneJ22_w1', ':gneJ23_w0',
        ':gneJ24_0', ':gneJ24_5', ':gneJ24_w0', ':gneJ24_w1', ':gneJ25_w0', ':gneJ26_0',
        ':gneJ26_3', ':gneJ26_w0', ':gneJ26_w1', ':gneJ27_w0',
        'CAED', 'CAEE', 'CAEU', 'CAEX', 'CAWD', 'CAWE', 'CAWU', 'CAWX',
        'JSND', 'JSNE', 'JSNU', 'JSNX', 'JSSD', 'JSSE', 'JSSU', 'JSSX'
    ]
    shape = [-55, 65, -30, 78]
    width = shape[1] - shape[0]
    height = shape[3] - shape[2]
    display = False
    ent_lanes = ['CAEE', 'CAWE', 'JSNE', 'JSSE']
    if display:
        sumo_binary = "sumo-gui"
    else:
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