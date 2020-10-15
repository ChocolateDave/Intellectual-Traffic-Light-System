# !/usr/bin/env python
# -*- coding: utf-8 -*-

#------------------------------------------
# Project: traffic light training scenario 
# Author: Juanwu Lu
# Date: 5 Sep. 2020
#------------------------------------------

import os
import sys
from xml.dom.minidom import parse
import numpy as np
from . import utils

if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

# Env Params
# traffic light
actions = {
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

# observation
shape = [-110, 68, -10, 210]
width = shape[1] - shape[0]
height = shape[3] - shape[2]
downsample = 4
frameskip = 25

# simulation
total_dur = 3600
warm_up_dur = 60

class tl_v1(object):
    r"""The sumo interactive environment based on openai gym. By default, 
    the reward range for RL training is set to [-inf,+inf]. It can be changed 
    accordingly."""

    def __init__(self, path="./", evals=False):

        print('-' * 100)
        print("Loading env ... ")

        self.frameskip = frameskip
        self.actions = actions
        self.action_size = len(self.actions)
        self.downsample = downsample
        self.observation_space = [int(self.frameskip/5), int(height/self.downsample), int(width/self.downsample)]
        self.total_time = total_dur
        self.warm_up_time = warm_up_dur

        self.binary = "sumo-gui" if evals else "sumo"
        self.cfg = os.path.join(path, "scenario/ITLS.sumo.cfg")
        self.convs = conversions
        self.edges, self.ents, self.exts, self.id = \
            utils.get_net(os.path.join(path, "scenario/ITLS.net.xml"))
        self.lnarea = utils.get_add(
            os.path.join(path, "scenario/ITLS.add.xml"))

        self.trans_sec = 0.0  # phase transition penalty

        print(
            "Traffic Light Control v1"
            "\nObservation space: ", self.observation_space,
            "\nAction space: [0, %d)" % self.action_size,
            "\nDownsample: %d, Frameskip: %d" % (self.downsample, self.frameskip),
            "\nInits ..."
        )
        print('-' * 100)
        
    # Seeding
    def seed(self, seed=None):
        _, seed1 = utils.np_random(seed)
        seed2 = utils.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    # Simulation functions
    def warmUp(self):
        """Before training, we need to make sure there's enough vehicles 
        on the given lanes. Hence, a warm up sequence is required for 
        filling the scenario. """
        while traci.simulation.getTime() <= self.warm_up_time:
            traci.simulationStep()

    def getState(self):
        """Imitate input observation acquired through api 
        of obejct tracing network. (e.g. YOLO)

        Returns:
            state (object): an array describe current traffic state.
        """

        def scale(x, range_list):
            """Normalize conversion factors to specific range.

            Returns:
                x (object): array or list after normalization.
            """
            assert isinstance(x, np.ndarray), "Matrix Error!"
            return np.round((x-np.min(x)) / (np.max(x)-np.min(x)) *
                            (range_list[1]-range_list[0]) + range_list[0])

        state = np.zeros([self.observation_space[1], self.observation_space[2]])
        for edge in self.edges:
            current_veh = traci.edge.getLastStepVehicleIDs(edge)
            for veh in current_veh:
                pos = traci.vehicle.getPosition(veh)
                for vehtype in self.convs.keys():
                    if vehtype in veh:
                        # x and y have to minus offsets of left and down boundaries, respectively.
                        x = round((pos[0] + 110)/self.downsample)
                        y = round((pos[1] + 10)/self.downsample)
                        if x in range(state.shape[0]) and y in range(state.shape[1]):
                            state[x][y] += self.convs[vehtype]
        return state

    def isEpisode(self):
        """Justify if a scenario is finished.

        Return:
            done (bool)
        """
        if traci.simulation.getTime() >= self.total_time:
            traci.close()
            print('Scenario Finished!')
            print('-' * 100)
            return True
        return False

    def getReward(self):
        """ Get reward value after every action was taken.
        The reward function is designed to encourage a high
        speed while avoiding collisions.

        Return:
            reward (float): the reward value of action.
        """
        vdes, vs = [], []
        for edge in self.ents:
            vehids = traci.edge.getLastStepVehicleIDs(edge)
            for veh in vehids:
                vdes.append(traci.vehicle.getMaxSpeed(veh))
                vs.append(traci.vehicle.getSpeed(veh))
        vdes, vs = np.array(vdes), np.array(vs)
        reward = max(np.sum(vdes**2)-np.sum((vdes-vs)**2),0)/np.sum(vdes**2)
        return reward

    # return trasistion phases
    def phase_transition(self, phase_o, phase_d):
            """Generate requisite transition between different phases. 

            Params:
                phase_o (string): current phase of traffic light controller.
                phase_d (string): future phase of traffic light controller.

            Returns:
                yellow_phase (string): the yellow-light phase for transition.
                red_phase (string): the red-light phase for transition.
            """
            assert len(phase_o) == len(phase_d), "Phase expression error!"
            yellow_phase, red_phase = '', ''
            for i in range(len(phase_o)):
                if phase_o[i] != phase_d[i]:
                    if phase_o[i] == 'G':
                        yellow_phase += 'y'
                        red_phase += 'r'
                    else:
                        yellow_phase += phase_o[i]
                        red_phase += phase_o[i]
                else:
                    yellow_phase += phase_o[i]
                    red_phase += phase_o[i]
            return yellow_phase, red_phase

    # Main fuctions of gym
    def step(self, action):
        if isinstance(action, np.ndarray):
            # encode one-hot vector
            action = np.argmax(action)
        reward = 0.0
        state_list = []
        # Future phase
        if action:
            fp = list(self.actions.values())[action]
        else:
            fp = None
       
        # Current phase
        if fp:
            # Ignore None action for no change.
            cp = traci.trafficlight.getRedYellowGreenState(self.id)
            # Phase transition
            if cp != fp:
                yp, rp = self.phase_transition(cp, fp)
                traci.trafficlight.setRedYellowGreenState(self.id, yp)
                for _ in range(15):
                    # 3s amber
                    traci.simulationStep()
                traci.trafficlight.setRedYellowGreenState(self.id, rp)
                for _ in range(10):
                    # 2s full-red
                    traci.simulationStep()
                # penalty for transition
                reward -= 5.0 / self.trans_sec
                self.trans_sec = 0.0
            traci.trafficlight.setRedYellowGreenState(self.id, fp)
        
        # Observe
        for i in range(self.frameskip):
            traci.simulationStep()
            if i % 5 == 0:
                state = self.getState()
                state_list.append(state)
                reward += self.getReward()
                self.trans_sec += 1
        observation = np.stack(state_list)
        observation = np.reshape(observation, self.observation_space)
        done = self.isEpisode()
        return observation, reward, done, {'cp': fp}

    def reset(self, mode='training', seed=None):
        assert isinstance(self.binary, str), "Sumo binary error!"
        assert mode in ('training', 'evaluation'), "Unsupported mode!"

        if seed:
            assert isinstance(seed, int), "Invalid simulation seed!"
            self.seed = seed
            sumoCmd = [self.binary, '-c', self.cfg, '--start',
                       '--seed', seed, '--quit-on-end']
        else:
            sumoCmd = [self.binary, '-c', self.cfg,
                       '--start', '--random', '--quit-on-end']
        traci.start(sumoCmd, label=mode)
        self.scenario = traci.getConnection(mode)
        self.warmUp()
        state, _, _, _ = self.step(0)
        return state


# Reward deprecated
"""
delay = dict.fromkeys(self.ents, 0)
ql = dict.fromkeys(['W', 'E', 'S', 'N'], 0)
vol = 0
# get queue length
for lanearea in self.lnarea:
    _dir = lanearea[0]
    ql[_dir] += traci.lanearea.getJamLengthMeters(lanearea)
# get delay
for edge in self.ents:
    d = 0
    for indx in range(1, traci.edge.getLaneNumber(edge)):
        lane = edge + '_' + str(indx)
        d += 1 - (traci.lane.getLastStepMeanSpeed(lane)/traci.lane.getMaxSpeed(lane))
    delay[edge] += d * 0.2 / (traci.edge.getLaneNumber(edge) - 1)
# get through volume
for ext in self.exts:
    vol += traci.edge.getLastStepVehicleNumber(ext)
reward = 0.1 * vol - np.sum(list(delay.values())) - 10 * np.std(list(delay.values())) \
    - 0.005 * np.sum(list(ql.values())) - 0.01 * np.std(list(ql.values()))
"""