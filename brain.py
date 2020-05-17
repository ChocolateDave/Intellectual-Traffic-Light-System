# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: CNS for Deep Reninforcement Learning
Author: Juanwu Lu
Facility: Tongji University
"""
from inspect import getmembers
import datetime
import os
import numpy as np
import tensorflow as tf

from memory import ReplayBuffer

# 不使用tensorflow 2.0训练
if int(tf.__version__[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


def class_vars(classobj):
    r"""从配置类中循环调用相关参数
    Recursively retrieve class members and their values.
    :Param
        classobj (class): the target class object.
    :Return
        class_vars (dict): class members and their values.
    """
    return {k: v for k, v in getmembers(classobj)
            if not k.startswith('__') and not callable(k)}


class BaseAgent(object):
    r"""深度强化学习的抽象控制主单元
    Abstract object representing a model"""

    def __init__(self, config):
        self._saver = None
        self.config = config

        self._attrs = class_vars(config)
        for attr in self._attrs:
            name = attr if not attr.startswith('__') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

    def save_model(self, step=None):
        """Save model params at certain step N.

        :Params
            step (int): the training step.
        """
        print(datetime.now(), ": Saving checkpoints...")
        model_name = type(self).__name__

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        """Load pre-trained model for training or evaluation.

        :Return
            loaded (bool): workflow indicator
        """
        print(datetime.now(), ": Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            file_name = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, file_name)
            print("# Load SUCCESS: %s" % file_name)
            return True
        else:
            print(datetime.now(), ": Load FAILED!: %s" % self.checkpoint_dir)
            return False

    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)

    @property
    def model_dir(self):
        model_dir = self.config.env_name
        for k, v in self._attrs.items():
            if not k.startswith('__') and k not in ['display']:
                model_dir += "/%s= %s" % (k, ",".join([str(i) for i in v])
                                             if type(v) == list else v)
        return model_dir + '/'

    @property
    def saver(self):
        if self._saver == None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver


class Brain(BaseAgent):
    r"""强化学习单元中枢主单元，基于原始DQN增设OWM单元等定制化内容
    The DRL network for Traffic light control system. """

    def __init__(self, config, env):
        super(Brain, self).__init__(config)
        self.config=config
        self.owm = config.owm
        self.memory = ReplayBuffer(env.observation_space,env.action_size,config.memory_size)
        self.timestep = 0
        self.actions = env.action_size
        self.epsilon=config.epsilon_start
        self.owm_alphas = config.owm_alphas
        self.state_size = env.observation_space



        with tf.name_scope('placeholders'):
            with tf.name_scope('IO'):
                # observation_space = [batch, width, height, channels], which is different from torch.
                self.s = tf.placeholder(tf.float32, [None, self.state_size], name='state')
                self.r = tf.placeholder(tf.float32, [None, ], name='reward')
                self.a = tf.placeholder(tf.int32, [None, env.action_size], name='action')
                self.s_ = tf.placeholder(tf.float32, [None, self.state_size], name='next_state')

        with tf.name_scope('nnet'):
            self.build_nn(owm=config.owm)

    def build_nn(self, owm=False):
        r"""神经网络主函数，客制化定义OWM模组
        Main function for building neural network.
        Params:
            owm(bool): whether to use orthogonal weight modification module.
        """
        # 神经网络 Neural Network
        with tf.name_scope('hidden_conv1'):
            w_conv1 = self.weight_variable([5, 5, self.config.frameskip, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(self.s, w_conv1, 4)+b_conv1)
            h_poo1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[
                                    1, 2, 2, 1], padding="SAME")
        with tf.name_scope('hidden_conv2'):
            w_conv2 = self.weight_variable([3, 3, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_poo1, w_conv2, 2) + b_conv2)
        with tf.name_scope('hidden_conv3'):
            w_conv3 = self.weight_variable([2, 2, 64, 64])
            b_conv3 = self.bias_variable([64])
            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)
            h_conv3_flattend = tf.reshape(h_conv3, [-1, 1920])

        # 全连接层 fully connected layer
        with tf.name_scope('hidden_fc'):
            w_fc1 = self.weight_variable([1920, 512])
            b_fc1 = self.bias_variable([512])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flattend, w_fc1) + b_fc1)
        if self.owm:
            # 为全连接层配置正交矩阵 P matrix for fully connected layer.
            with tf.name_scope('P_matrix'):
                # 初始化P矩阵为同权重第一维度大小的单位矩阵 initialize p mat to identity
                self.P1 = tf.Variable(tf.eye(int(1920)))
                # 求输入矩阵的平均值 mean of inputs
                x_mu = tf.reduce_mean(h_conv3_flattend, 0, keep_dims=True)
                # 计算P矩阵更新值 calculate P update
                k = tf.matmul(self.P1, tf.transpose(x_mu))
                delta_P1 = tf.divide(tf.matmul(k, tf.transpose(k)), self.owm_alphas[0][0] + tf.matmul(x_mu, k))
                # 动态更新P矩阵 apply update to P
                self.update_p1 = tf.assign_sub(self.P1, delta_P1)
        # 输出层 output layer
        with tf.name_scope('hidden_output'):
            w_fc2 = self.weight_variable([512, self.action_size])
            b_fc2 = self.bias_variable([self.action_size])
            self.q = tf.nn.bias_add(tf.matmul(h_fc1, w_fc2), b_fc2, name='q')
        if self.owm:
            # 为输出层更新配置正交矩阵 P matrix for output layer.
            with tf.name_scope('P_matrix'):
                self.P2 = tf.Variable(tf.eye(int(512)))
                x_mu = tf.reduce_mean(h_fc1, 0, keep_dims=True)
                k = tf.matmul(self.P2, tf.transpose(x_mu))
                delta_P2 = tf.divide(tf.matmul(k, tf.transpose(k)), self.owm_alphas[0][1] + tf.matmul(x_mu, k))
                self.update_p2 = tf.assign_sub(self.P2, delta_P2)
            
        # 训练调节器 Optimiser
        with tf.name_scope('optimiser'):
            q_action = tf.reduce_mean(tf.mul(self.q, self.a), reduction_indices = 1)
            self.q_target = tf.placeholder(tf.float32, [None,])
            self.cost = tf.reduce_mean(tf.square(self.q_target - q_action))
            if self.owm:
                # 初始化训练调控器 Initialize optimiser.
                optimiser = tf.train.AdamOptimizer(self.lr)
                # 通过BP算法计算梯度值 calculates ΔW by BP method.
                grads = optimiser.compute_gradients(self.cost, var_list=[w_fc1, w_fc2])
                # 依据OWM算法修改梯度值 modify ΔW with p matrix.
                with tf.name_scope('OWM'):
                    grads_fc = [self.owm(self.P1, grads[0])]
                    grads_output = [self.owm(self.P2, grads[1])]
                self.trainStep = optimiser.apply_gradients([grads_fc[0], grads_output[0]])
            else:
                self.trainStep = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
    
    def train(self):
        r"""智能体训练主函数
        Main function for training the agent.
        """
        # 第一步：随机从经验池中调取批训练数据  Step1: randomly obtain minibatch from replay buffer
        s_batch, a_batch, r_batch, ns_batch, terminals = self.memory.sample(self.config.batch_size)
        
        #第二步：计算目标Q值 Step2: calculate target Q value
        q_target = []
        q_batch = self.q.eval(feed_dict={self.s:ns_batch})
        for i in range(0, self.batchsize):
            terminal = terminals[i]
            if terminal:
                q_target.append(r_batch[i])
            else:
                q_target.append(r_batch[i] + self.gamma * np.max(q_batch))
        
        #第三步：训练网络 Step3: train neural network
        self.trainStep.run(feeddict={
            self.q_target:q_target,
            self.a: a_batch,
            self.s: s_batch
        })

        # 保存训练结果 Save network with certain frequency.
        if self.timestep %  self.config.checkpoint == 0:
            self.save_model(step=self.timestep)
    
    def interact(self, state, action, reward, terminal):
        r"""智能体与环境交互的决策主函数，存储记忆并决定是否自我训练
        Main function to determine interaction policies, store transition and train.
        Params:
            state(tensor): next observation from environment.
            action(tensor): action that has taken.
            reward(tensor): reward value.
            terminal(bool): indicate whether a task is finished.
        """
        self.memory.add(self.currentState, action, reward, state, terminal)
        if self.timestep > self.config.learn_initial:
            self.train()
        self.currentState = state
        self.timestep += 1

    def get_action(self):
        r"""智能体动作决策函数，采用贪心算法
        Main function to determine action by ε-greedy algorithm.
        """
        q = self.q.eval(feed_dict={self.s: [self.currentState]})[0]
        action = np.zeros(self.actions)
        action_indx = 0
        if np.random.random() <= self.epsilon:
            action_indx = np.random.randint(self.actions)
            action[action_indx] = 1
        else:
            action_indx = np.argmax(q)
            action[action_indx] = 1
        # 更新ε update epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon-self.config.epsilon_decay)
        return action

    def weight_variable(self, shape):
        """Initialize layer weights with given shape.
        Params:
            shape (list): shape of the weight
        Return:
            weight (tf.Variable)
        """
        w_initializer = tf.random_normal_initializer(.1, .3)
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

    def owm(self, P, g_v, lr=1.0):
        """With given gradients by BP method, return modified gradients with p matrices.
        Params:
            P (tensor): p matrix.
            g_v (tuple): (gradient, variable) pairs.
            lr (float32): predefied learning rate.
        Return:
            g_ (tensor): modified gradients.
            g_v[1] (tensor): variable names.
        """
        g_ = lr*tf.matmul(P, g_v[0])
        return g_, g_v[1]
