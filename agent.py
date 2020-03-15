# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
import time
# Not use tensorflow 2.0
if int(tf.__version__[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from copy import deepcopy
from datetime import datetime
from inspect import getmembers
from memory import ReplayBuffer

def class_vars(classobj):
    r"""Recursively retrieve class members and their values.
    :Param
        classobj (class): the target class object.
    :Return
        class_vars (dict): class members and their values.
    """
    return {k: v for k, v in getmembers(classobj)
                    if not k.startswith('__') and not callable(k) }
class BaseModel(object):
    r"""Abstract object representing a model"""
    def __init__(self, config):
        self._saver = None
        self.config = config

        self._attrs = class_vars(config)
        for attr in self._attrs:
            name = attr if not attr.startswith('__') else attr[1: ]
            setattr(self, name, getattr(self.config, attr))
        
    def save_model(self, step=None):
        """Save model params at certain step N.
        
        :Params
            step (int): the training step.
        """
        print(datetime.now(),": Saving checkpoints...")
        model_name=type(self).__name__

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        """Load pre-trained model for training or evaluation.

        :Return
            loaded (bool): workflow indicator
        """
        print(datetime.now(),": Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            file_name = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, file_name)
            print("# Load SUCCESS: %s"%file_name)
            return True
        else:
            print(datetime.now(),": Load FAILED!: %s" %self.checkpoint_dir)
            return False
        
    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)
        
    @property
    def model_dir(self):
        model_dir = self.config.env_name
        for k, v in self._attrs.items():
            if not k.startswith('__') and k not in ['display']:
                model_dir += "/%s= %s" % (k, ",".join([str(i) for i in v]) \
                                             if type(v) == list else v)
        return model_dir + '/'

    @property
    def saver(self):
        if self._saver==None:
            self._saver=tf.train.Saver(max_to_keep=10)
        return self._saver

class DQNAgent(BaseModel):
    r"""Multifunctional DQN agent for reinforcement learning.
    Modified based on Morvan_Zhou's naive dqn neural network. 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5_Deep_Q_Network/DQN_modified.py
    """
    def __init__(self, config, environment, sess):
        super(DQNAgent, self).__init__(config)
        self.sess = sess
        self.env = environment
        self.state_size, self.action_size = env.observation_space, env.action_space
        self.memory = ReplayBuffer(self.state_size, self.action_size, Memory_size)

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder(tf.int32, None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)
        
        self.build_nn()
    
    def build_nn(self):
        """Building the neural network for training.
        :Params
            input_shape (list): state size.
            n_actions (int): number of actions.
        :**kwargs
            dueling (bool): if or not use dueling DQN.
            double (bool): if or not use double DQN.
            prioritized (bool): if or not use prioritized replay buffer.
        """
        # ---------------- all inputs ---------------
        with tf.name_scope('placeholders'):
            with tf.name_scope('IO'):
                self.s = tf.placeholder(tf.float32, [None, self.state_size[1],self.state_size[2],self.state_size[0]], \
                     name='state')  # input_size = [batch, width, height, channels], which is different from torch.
                self.r = tf.placeholder(tf.float32, [None, ], name='reward')
                self.a = tf.placeholder(tf.int32, [None, self.action_size], name = 'action')
                self.s_ = tf.placeholder(tf.float32,[None, self.state_size[1], self.state_size[2],self.state_size[0]],\
                     name='next_state')
                # Preserve for OWM module:
                # self.alphas = tf.placehold(tf.float32, name='alpha')
        # ---------------- build prediction network ----------------
        with tf.variable_scope('pred_net'):
            with tf.name_scope('hidden_1'):
                w_conv1 = self.weight_variable([5, 5, self.state_size[0], 32])
                b_conv1 = self.bias_variable([32])
                h_conv1 = tf.nn.relu(self.conv2d(self.s, w_conv1, 4)+b_conv1)
                h_poo1 =  tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            with tf.name_scope('hidden_2'):
                w_conv2 = self.weight_variable([3, 3, 32, 64])
                b_conv2 = self.bias_variable([64])
                h_conv2 = tf.nn.relu(self.conv2d(h_poo1, w_conv2, 2) + b_conv2)
            with tf.name_scope('hidden_3'):
                w_conv3 = self.weight_variable([2, 2, 64, 64])
                b_conv3 = self.bias_variable([64])
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)
                h_conv3_flattend = tf.layers.flatten(h_conv3)
            with tf.name_scope('hidden_fc'):
                if self.dueling:
                    # Preserve for dueling q-learning
                    pass
                else:
                    w_fc1 = self.weight_variable([h_conv3_flattend.shape[1], 512])
                    b_fc1 = self.bias_variable([512])
                    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flattend, w_fc1) + b_fc1)
                    w_fc2 = self.weight_variable([512, n_actions])
                    b_fc2 = self.bias_variable([n_actions])
                    self.q_pred = tf.nn.bias_add(tf.matmul(h_fc1, w_fc2), b_fc2, name='q_pred')
            self.q_action = tf.argmax(self.q_pred, dimension=1)
        # Summary writer unit
        q_summary=[]
        avg_q = tf.reduce_mean(self.q, 0)
        for idx in xrange(self.action_size):
            q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
        self.q_summary = tf.summary.merge(q_summar, 'q_summary')
        # ---------------- build target network ----------------
        with tf.variable_scope('target_net'):
            with tf.name_scope('hidden_1'):
                w_conv1 = self.weight_variable([5, 5, self.state_size[0], 32])
                b_conv1 = self.bias_variable([32])
                h_conv1 = tf.nn.relu(self.conv2d(self.s, w_conv1, 4)+b_conv1)
                h_poo1 =  tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            with tf.name_scope('hidden_2'):
                w_conv2 = self.weight_variable([3, 3, 32, 64])
                b_conv2 = self.bias_variable([64])
                h_conv2 = tf.nn.relu(self.conv2d(h_poo1, w_conv2, 2) + b_conv2)
            with tf.name_scope('hidden_3'):
                w_conv3 = self.weight_variable([2, 2, 64, 64])
                b_conv3 = self.bias_variable([64])
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)
                h_conv3_flattend = tf.layers.flatten(h_conv3)
            with tf.name_scope('hidden_fc'):
                if self.dueling:
                    # Preserve for dueling q-learning
                    pass
                else:
                    w_fc1 = self.weight_variable([h_conv3_flattend.shape[1], 512])
                    b_fc1 = self.bias_variable([512])
                    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flattend, w_fc1) + b_fc1)
                    w_fc2 = self.weight_variable([512, n_actions])
                    b_fc2 = self.bias_variable([n_actions])
                    self.q_target = tf.nn.bias_add(tf.matmul(h_fc1, w_fc2), b_fc2, name='q_target')
        if self.double:
            self.q_target_idx = tf.placeholder(tf.int32, [None, None], 'target_output_idx')
            self.q_target_with_idx = tf.gather_nd(self.q_target, self.q_target_idx)
        # --------------- network synchronization --------------
        with tf.variable_scope('network_sync'):
            t_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            p_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_net')
            self.sync = tf.assign(t, p) for t, p in zip(t_param, p_param)
        # --------------- network optimizer ----------------
        with tf.variable_scope('optimizer'):
            if self.prioritized:
                # preserved for prioritized replay buffer
                pass
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='q_max_s_')
            self.q_target = tf.stop_gradient(q_target)
            a_indices = tf.stack([tf.range(n_actions, dtype=tf.int8), self.a], axis=1)
            self.q_pred_wrt_a = tf.gather_nd(params=self.q_pred, indices=a_indices)
            self.loss = tf.reduce_mean(tf.squred_difference(self.q_target, self.q_pred_wrt_am, name='TD_error'))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # --------------  summary writer --------------
        with tf.variable_scope('summary'):
            # Plot variables
            scalar_summary_tags =['average reward', 'average loss']
            self.summary_placeholders, self.summary_ops = {}, {}
            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar("%s" % tag, self.summary_placeholders[tag])
            # Histogram variables
            histogram_summar_tags = ['episode.reward', 'epsiode.actions']
            for tag in histogram_summar_tags:
                self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.histogram("%s" % tag, self.summary_placeholders[tag])
            # Initialize summary writer
            self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)
        # -------------- Initialize network --------------
        tf.initialize_all_variables().run()
        self._saver = tf.train.Saver(self.)

    def observe(self, state, reward, action, state_, terminal):
        """Interact with the environment and observe state,
        and then store memory fractions.
        :Params
            state (array): current state.
            reward (array): current reward.
            action (arrary): current action.
            terminal (boolean): if or not the scenario is done.
        """
        if self.reward_range and isinstance(self.reward_range, list):
            reward = np.clip(reward, a_min=min(self.reward_range),
                                             a_max=max(self.reward_range))
        
        self.memory.add(state, reward, action, state_, terminal)
        if self.step > self.learn_initial:
            self.q_learning()
            if self.step % self.sync_tau == 0:
                self.sync()

    def predict(self, state_, eval_ep = None):
        """Predict the optimal action given a certain state input.
        :Param
            state_ (array): current state.
        :Return
            action (int): optimal action.
        """
        epsilon = eval_ep or max(self.epsilon_final, 
                            self.epsilon_start - self.step / self.epsilon_frames)
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            action = self.q_action.eval({self.s: [state_]})[0]
        return action

    def get_action(self, score):
        r"""Action selector based on epsilon greedy algorithm.
        With step t, the action of agent is defined as follows:
            generate random number c with np.random
            if c >= epsilon:
                choose the optimal action a_t* with argmax selector
            else:
                randomly select an action from action set.

            See http://home.deib.polimi.it/restelli/MyWebSite/pdf/rl5.pdf 
            for more details.
        :Return
            action (int): action to take.
        """
        if np.random.uniform() < self.epsilon:
            action = np.argmax(score)
        else:
            action = np.random.randint(0, tf.shape(self.a)[0])
        return action

    def q_learning(self):
        """Main function of q-learning algorithm."""
        s, a, r, s_, done = self.memory.sample()
        if self.double:
            # Use double dqn
            pass
        else:
            pass

    def train(self):
        self.step = 0
        init_step = self.step_op.eval()
        init_time = time.time()

        ep_num, self.update_count, ep_reward = 0, 0, 0
        total_reward, self.total_loss = 0., 0.
        max_avg_ep_reward = 0
        ep_reward, actions = [], []

        state, reward, action, terminal = self.env.reset()
        while True:
            if self.step == self.learn_initial:
                ep_num, self.update_count, ep_reward = 0,0,0
                total_reward, self.total_loss = 0., 0.
                ep_reward, actions = [], []
            # 1. predict optimal action given state.
            action = self.predict(state)
            # 2. interact with the env.
            state_, reward, terminal, _ = self.env.step(action)
            # 3. observe.
            self.observe(state, reward, action, state_, terminal)
            # 4. store status
            if terminal:
                # scenario is done.
                state, reward, action, terminal = self.env.reset()
                ep_num += 1
                ep_reward.append(reward)
                ep_reward = 0.
            else:
                ep_reward += reward
            actions.append(action)
            total_reward += reward
            # 5. initial training
            if self.step >= self.learn_initial:
                if terminal:
                    avg_reward = total_reward / self.step
                    avg_loss = self.total_loss / self.step
                    try:
                        max_ep_reward = np.max(ep_reward)
                        min_ep_reward = np.min(ep_reward)
                        avg_ep_reward = np.mean(ep_reward)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0., 0., 0.
                    print('\n', datetime.now(), ' avg_r: %.4f, avg_l: %.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # episodes: %d' \
                                        % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, ep_num))
                    
                if (self.step - self.learn_initial) % self.checkpoint == 0:
                    self.save_model(self.step + 1)
                if self.step > self.learn_initial:
                    self
            self.step += 1


        """# Update target network with certain step.
        if self.step %  == 0:
            self.sess.run(self.target_replace_op)
        
        # Sample batches from experience buffer when exceed preset threshold.
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), \
                                                                        size=self.batch_size)
        samples = self.memory[sample_index, : ]

        _, cost = self.sess.run(
            [self.optimizer, self.loss],
            feed_dict={
                self.s: samples[:,  : self.state_size],
                self.a: samples[:, self.state_size],
                self.r: samples[:, self.state_size + 1],
                self.s_: samples[:, -self.state_size : ]
            }
        )

        # Update epsilon.
        self.epsilon = max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
        self.step += 1

        # Save network when reach checkpoint.
        if self.step % self.ckpt == 0:
            self.save_model(step=self.step)"""

    def play(self):
        """Benchmark function for testing training effects"""
        pass

    def weight_variable(self, shape):
        """Initialize layer weights with given shape.
        Params:
            shape (list): shape of the weight
        Return:
            weight (tf.Variable)
        """
        w_initializer=tf.random_normal_initializer(.1, .3)
        return tf.Variable(w_initializer(shape))
    
    def bias_variable(self, shape):
        """Initialize layer bias with given shape.
        Params:
            shape (list): shape of the weight
        Return:
            bias (tf.Variable)
        """
        b_initializer = tf.constant_initializer(.1)
        return tf.Variable(b_initializer(shape))
           
    def conv2d(self, x, w, s):
        """A simplified version of tf.nn.conv2d.
        Params:
            x (tensor): input placeholder.
            w (tensor): weight placeholder.
            s (int): stride, must be 1, 2 or 4.
        Return:
            convolution layer placeholder.
        """
        return tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='VALID')

if __name__ == "__main__":
    from env import TrafficLight_v0
    env = TrafficLight_v0()
    agent = DQNAgent(env)