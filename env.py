# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
# Check sumo env
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")
import numpy as np
import seeding
import gym
import traci


class TrafficLight_v1(gym.Env):
    r"""The sumo interactive environment based on openai gym. By default, 
    the reward range for RL training is set to [-inf,+inf]. It can be changed 
    accordingly."""

    def __init__(self, config):
        super(TrafficLight_v1, self).__init__()
        if isinstance(config.frameskip, int):
            self.frameskip = config.frameskip
        else:
            self.frameskip = np.random.randint(config.frameskip[0], config.frameskip[1])
        self.actions = config.actions
        self.action_size = len(config.actions)
        self.downsample = config.downsample
        self.observation_space = [int(config.width/self.downsample), int(config.height/self.downsample), int(self.frameskip/5)]
        if config.reward_range:
            self.reward_range = config.reward_range
        self.total_time = config.total_dur
        self.warm_up_time = config.warm_up_dur
        # Sumo params
        self.binary = config.sumo_binary
        self.cfg = config.sumo_config
        self.convs = config.conversions
        self.edges = config.edges
        self.ents = config.ent_edges
        self.exts = config.ext_edges
        self.id = config.TLid
        self.lnarea = config.lanearea

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

        state = np.zeros([self.observation_space[0], self.observation_space[1]])
        for edge in self.edges:
            current_veh = traci.edge.getLastStepVehicleIDs(edge)
            for veh in current_veh:
                pos = traci.vehicle.getPosition(veh)
                for vehtype in self.convs.keys():
                    if vehtype in veh:
                        # x and y have to minus offsets of down and left boundaries, respectively.
                        x = round((pos[1] + 10)/self.downsample)
                        y = round((pos[0] + 110)/self.downsample)
                        if x in range(state.shape[0]) and y in range(state.shape[1]):
                            state[x][y] += self.convs[vehtype]
        return state

    def isEpisode(self):
        """Justify if a scenario is finished.

        Return:
            done (bool)
        """
        if traci.simulation.getTime() >= self.total_time:
            print('********Scenario Finished!********')
            #self.scenario.close()
            traci.close()
            return True
        return False

    def getReward(self):
        """ Get reward value after every action was taken.
        The reward function is by default designed as a linear 
        function wrt queue length, delay and through volume.

        Return:
            reward (float): the reward value of action.
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
        return reward

    # Main fuctions of gym
    def step(self, action):
        if isinstance(action, np.ndarray):
            # encode one-hot vector
            action = np.argmax(action)
        assert action in range(len(self.actions)), "IndexInvalid!"
        reward = 0.0
        state_list = []
        # Future phase
        fp = list(self.actions.values())[action]

        def phase_transition(phase_o, phase_d):
            """A transition phase is required between different phases. 

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
       
        # Current phase
        if fp:
            # Ignore None action for no change.
            cp = traci.trafficlight.getRedYellowGreenState(self.id)
            # Phase transition
            if cp != fp:
                yp, rp = phase_transition(cp, fp)
                traci.trafficlight.setRedYellowGreenState(self.id, yp)
                for _ in range(15):
                    traci.simulationStep()
                traci.trafficlight.setRedYellowGreenState(self.id, rp)
                for _ in range(10):
                    traci.simulationStep()
            traci.trafficlight.setRedYellowGreenState(self.id, fp)
        
        # Observe
        for i in range(self.frameskip):
            traci.simulationStep()
            if i % 5 == 0:
                state = self.getState()
                state_list.append(state)
                reward += self.getReward()
        observation = np.stack(state_list)
        observation = np.reshape(observation, self.observation_space)
        done = self.isEpisode()
        return observation, reward, done, {}

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


if __name__ == "__main__":
    import config
    import matplotlib.pyplot as plt
    config = config.SumoConfigs
    env = TrafficLight_v1(config)
    obs = env.reset()
    print('observation space:',obs.shape)
    plt.ion()
    f = False
    while not f:
        plt.cla()
        plt.imshow(obs[0], cmap='gray', origin='lower')
        action = np.random.randint(0, len(config.actions))
        obs, r, f, _ = env.step(action)
        print("reward:",r)
        plt.pause(0.1)
    plt.ioff()
