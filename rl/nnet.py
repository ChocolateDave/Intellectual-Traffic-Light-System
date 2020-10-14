# !/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------
# File: Neural Network
# Author: Juanwu Lu
# Date: 5 Sep. 2020
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
        self.name = 'naive_dqn'

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

# ----------------------------------------------------------

class Dueling_DQN(nn.Module):
    r"""竞争型深度Q网络结构
    Dueling DQN for training by Ziyu Wang et al.
    Further details on:
    "https://arxiv.org/abs/1511.06581"
    """
    def __init__(self, input_shape, n_action, lr):
        super(Dueling_DQN, self).__init__()
        self.name = 'dueling_dqn'
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
        _output = self.convolutional_layer(T.zeros(1, *shape))
        return int(np.prod(_output.size()))

    def forward(self, x):
        x = x.float()
        conv_out = self.convolutional_layer(x).view(x.size(0), -1)
        val = self.val_layer(conv_out)
        adv = self.adv_layer(conv_out)
        return val + adv - adv.mean()

# ----------------------------------------------------------

class OWM_DQN(nn.Module):
    r"""基于正交修正模组的DQN
    DQN with implementation of OWM。
    Further details on:
    ""
    """
    def __init__(self, input_shape, n_action, lr):
        super(OWM_DQN, self).__init__()
        self.name = 'owm_dqn'
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
        _input = T.zeros(1, *shape)
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
        x_list.append(T.mean(con1_o,0,True))
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

# ----------------------------------------------------------
class Dummy_Block(nn.Module):
    
    expansion = 1

    def __init__(self, in_dim, o_dim, stride=1):
        super(Dummy_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, o_dim, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(o_dim)
        self.conv2 = nn.Conv2d(o_dim, o_dim, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(o_dim)
    
    def Swish(self, x):
        return x * T.sigmoid(x)
    
    def forward(self, x):
        residual = x

        out = self.Swish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        return self.Swish(out)

class RLT(nn.Module):
    r"""智能信号灯系统完整神经网络(基于ResNet理念)
    Neural Network for robust and light-weight traffic light.
    """
    def __init__(self, input_shape, block, layers, n_action):
        super(RLT, self).__init__()
        self.name = 'rlt'
        # Main edifice
        self.in_dim = 256
        self.conv1 = nn.Conv2d(input_shape[0], 256, 5, 2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.res_layers = self._make_layer(block, 256, layers[0])
        res_out_size = self._get_res_out(input_shape)
        self.prj = nn.Linear(res_out_size, 256)
        self.val = nn.Linear(256, 1)
        self.act = nn.Linear(256, n_action) 

    def _get_res_out(self, shape):
        r"""计算降维后卷积层输出的列向量长度
        Return the flatten size of convolutional layer output.
        Params:
            shape (list): shape of input matrix for conv-layers.
        Return:
            length (int): the size of the flattened vector.
        """
        _output = self.Swish(self.bn1(self.pool(self.conv1(T.zeros(1, *shape)))))
        _output = self.res_layers(_output)
        return int(np.prod(_output.size()))

    def _make_layer(self, block, o_dim, blocks, stride=1):
        layers = []
        layers.append(block(256, o_dim, stride))
        self.in_dim = o_dim * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_dim, o_dim))
        
        return nn.Sequential(*layers)

    def Swish(self, x):
        return x * T.sigmoid(x)

    def forward(self, x):
        x = x.float()
        x = self.Swish(self.bn1(self.pool(self.conv1(x))))
        x = self.res_layers(x)
        x = x.view(x.size(0), -1)
        val = self.val(self.Swish(self.prj(x)))
        adv = self.act(self.Swish(self.prj(x)))
        return val + adv - adv.mean()

