# !/usr/bin/env python
# -*- coding: utf-8 -*-

import traci
from sumolib import checkBinary
from settings import *
from lib import seeding
import numpy as np
import gym
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class Traffic_Light_Control(gym.Env):
    r"""The sumo interactive environment based on openai gym. By default, 
    the reward range for RL training is set to [-inf,+inf]. It can be changed 
    accordingly. """

    def __init__(self):
        super(Traffic_Light_Control, self).__init__()
        if isinstance(Frameskip, int):
            self.frameskip = Frameskip
        else:
            self.frameskip = np.random.randint(Frameskip[0], Frameskip[1])
        self.warm_up_time = Warm_up_time
        self.total_time = Total_time
        self.shape = [Area[1]-Area[0], Area[3]-Area[2]]

        self.observation_space = (Frameskip, self.shape[0], self.shape[1])
        self.action_space = len(Actions)
        if Reward_range:
            self.reward_range = Reward_range

    # Seeding
    def seed(self, seed=None):
        _, seed1 = seeding.np_random(seed)
        # Generate a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    # Simulation functions
    def warmUp(self):
        """ Before training, we need to make sure there's enough vehicles 
        on the given lanes. Hence, a warm up sequence is required for 
        filling the scenario. """
        while traci.simulation.getTime() <= Warm_up_time:
            traci.simulationStep()

    def getState(self):
        """Imitate input observation acquired through api 
        of obejctive tracing network (e.g. YOLO)
        
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

        state = np.zeros(self.shape)
        for edge in Edges:
            current_veh = traci.edge.getLastStepVehicleIDs(edge)
            for veh in current_veh:
                pos = traci.vehicle.getPosition(veh)
                for vehtype in Conversions.keys():
                    if vehtype in veh:
                        x = round(pos[1] + 55)
                        y = round(pos[0] + 30)
                        if x in range(state.shape[0]) and y in range(state.shape[1]):
                            state[x][y] += Conversions[vehtype]
        
        return state

    def isEpisode(self):
        """Justify if a scenario is finished
        
        Return:
            done (bool)
        """
        if traci.simulation.getTime() >= Total_time:
            print('********Scenario Finished!********')
            self.scenario.close()
            return True
        return False

    def getReward(self):
        """ Get reward value after every action was taken.
        The reward function is by default designed as a 
        combination of queue length and travel time.

        Return:
            reward (float): the reward value of action.
        """
        ql, tt = 0.0, 0.0
        for edge in Entrance:
            ql += traci.edge.getLastStepHaltingNumber(edge)
            tt += traci.edge.getTraveltime(edge)
        reward = -.4 * ql - .1 * tt 
        return reward

    # Main fuctions of gym
    def step(self, action):
        assert isinstance(action, int), "TypeError"
        assert action in range(len(Actions)), "IndexInvalid!"
        reward = 0.0
        state_list = []
        # Future phase
        fp = list(Actions.values())[action]

        def phase_transition(phase_o, phase_d):
            """A transition phase is required between different phases. 
            
            Args:
                phase_o (string): current phase of traffic light controller.
                phase_d (string): future phase of traffic light controller.

            Returns:
                yellow_phase (string): the yellow-light phase for transition.
                red_phase (string): the red-light phase for transition.
            """
            assert len(phase_o) == len(phase_d), "Phase expression error!"
            yellow_phase, red_phase = '', ''
            for i in range(len(phase_1)):
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
            
            # Current phase
            cp = traci.trafficlight.getRedYellowGreenState(TLid)
            if cp != fp:
                yp, rp = phase_transition(cp, fp)
                traci.trafficlight.setRedYellowGreenState(TLid,yp)
                for _ in range(15):
                    traci.simulationStep()
                traci.trafficlight.setRedYellowGreenState(TLid, rp)
                for _ in range(10):
                    traci.simulationStep()
            traci.trafficlight.setRedYellowGreenState(TLid, fp)
            for i in range(Frameskip):
                traci.simulationStep()
                reward += self.getReward()
                if i % 5 == 0:
                    state = self.getState()
                state_list.append(state)
            observation = np.stack(state_list)
            done = self.isEpisode()
            return observation, reward, done, {}
    
    def reset(self, mode='training', seed=None):
        assert isinstance(Sumobinary, str), "Sumo binary error!"
        assert mode in ('training', 'evaluation'), "Unsupported mode!"

        if seed:
            assert isinstance(seed, int), "Invalid simulation seed!"
            self.seed = seed
            sumoCmd = [Sumobinary, '-c', Sumopath, '--start',
                       '--seed', seed, '--quit-on-end']
        else:
            sumoCmd = [Sumobinary, '-c', Sumopath, '--start', '--random','--quit-on-end']
        traci.start(sumoCmd, label=mode)
        self.scenario = traci.getConnection(mode)
        self.warmUp()
        observation = self.getState()
        return observation