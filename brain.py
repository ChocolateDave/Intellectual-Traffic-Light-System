# !/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------
# File: Cerebral Network
# Author: Juanwu Lu
# Facility: Tongji University
# -----------------------
import numpy as np
import torch as T
import torch.nn as nn

class Naive_DQN(nn.Module):
    r"""基础深度Q网络结构
    Naive deep Q network for reinforcement learning by V Minh et al.
    Further details on:
    "https://sites.google.com/a/deepmind.com/dqn/"
    """
    def __init__(self, input_shape, n_action):
        super(Naive_DQN, self).__init__()        
        # Naive dqn structure
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_layer = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )
    
    def _get_conv_out(self, shape):
        r"""计算降维后卷积层输出的列向量长度
        Return the flatten size of convolutional layer output.
        Params:
            shape (list): shape of input matrix for conv-layers.
        Return:
            length (int): the size of the flattened vector.
        """
        _output = self.convolutional_layer(T.zeros(1, *shape))
        return int(np.prod(_output.size()))

    def forward(self, x):
        x = x.float()
        conv_out = self.convolutional_layer(x).view(x.size(0), -1)
        action = self.fc_layer(conv_out)
        return action

class Dueling_DQN(nn.Module):
    r"""竞争型深度Q网络结构
    Dueling DQN for training by Ziyu Wang et al.
    Further details on:
    "https://arxiv.org/abs/1511.06581"
    """
    def __init__(self, input_shape, n_action, lr):
        super(Naive_DQN, self).__init__()
        # Dueling dqn structure
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.val_layer = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.adv_layer = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )
    
    def _get_conv_out(self, shape):
        r"""计算降维后卷积层输出的列向量长度
        Return the flatten size of convolutional layer output.
        Params:
            shape (list): shape of input matrix for conv-layers.
        Return:
            length (int): the size of the flattened vector.
        """
        _output = self.convolutional_layer(torch.zeros(1, *shape))
        return int(np.prod(_output.size()))

    def forward(self, x):
        x = x.float()
        conv_out = self.convolutional_layer(x).view(x.size(0), -1)
        val = self.val_layer(conv_out)
        adv = self.adv_layer(conv_out)
        return val + adv - adv.mean()

class OWM_DQN(nn.Module):
    r"""基于正交修正模组的DQN
    DQN with implementation of OWM。
    Further details on:
    ""
    """
    def __init__(self, input_shape, n_action, lr):
        super(OWM_DQN, self).__init__()
        # Transformation
        self.padding = nn.ReplicationPad2d(1)
        self.drop = nn.Dropout(0.2)

        # DQN structure
        self.conv1 = nn.Conv2d(input_shape[0], 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def _get_conv_out(self, shape):
        r"""计算降维后卷积层输出的列向量长度
        Return the flatten size of convolutional layer output.
        Params:
            shape (list): shape of input matrix for conv-layers.
        Return:
            length (int): the size of the flattened vector.
        """
        _input = torch.zeros(1, *shape)
        _output = self.padding(_input)
        _output = self.drop(nn.ReLU(self.conv1(_output)))
        _output = self.padding(_output)
        _output = self.drop(nn.ReLU(self.conv2(_output)))
        _output = self.padding(_output)
        _output = self.drop(nn.ReLU(self.conv3(_output)))
        return int(np.prod(_output.size()))
    
    def forward(self, x):
        h_list, x_list = [], []
        # Gates
        x = self.padding(x.float())
        x_list.append(T.mean(x,0,True))
        con1_o = self.drop(nn.ReLU(self.conv1(x)))

        con1_o = self.padding(con1_o)
        x_list.append(T.mean(con1_out,0,True))
        con2_o = self.drop(nn.ReLU(self.conv2(con1_o)))

        con2_o = self.padding(con2_o)
        x_list.append(T.mean(con2_o,0,True))
        con3_o = self.drop(nn.ReLU(self.conv3(con2_o)))

        h = con3_o.view(x.size(0), -1)
        h_list.append(T.mean(h, 0, True))
        h = nn.ReLU(self.fc1(h))
        
        h_list.append(T.mean(h, 0, True))
        action = self.fc2(h)
        return action, h_list, x_list
    