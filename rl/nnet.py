# !/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------
# File: Neural Network
# Author: Juanwu Lu
# Date: 5 Sep. 2020
# -----------------------
from typing import List, Optional, Tuple, Type, Union

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

class Res_Block(nn.Module):    
    expansion = 1
    def __init__(self, in_dim: int, o_dim: int, stride: int = 1):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, o_dim, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(o_dim)
        self.conv2 = nn.Conv2d(o_dim, o_dim, 3, 1, padding=1, bias=False)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x: T.Tensor):
        residual = x
        out = self.leaky_relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))        
        out += residual
        return self.leaky_relu(out)

class RLT(nn.Module):
    r"""智能信号灯系统神经网络ver 1.0(基于ResNet理念)
    Neural Network for robust and light-weight traffic light.
    """
    def __init__(self, input_dim: Type[Union[List, Tuple]], num_layers: List[int], 
            h_features: int, n_actions: int) -> None:
        super(RLT, self).__init__()
        self.name = 'rlt'
        # Main edifice
        self.in_dim = 256
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_dim[0], 256, 3, 2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.res_layers = self._make_layer(Res_Block, 256, num_layers[0])
        self.conv2 = nn.Conv2d(256, 2, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(256, 1, 1, 1, bias=False)
        self.fc1 = nn.Linear(1440, h_features)
        self.fc2 = nn.Linear(h_features, 1)
        self.fc3 = nn.Linear(720, n_actions)
        self.bn2 = nn.BatchNorm2d(2)
        self.bn3 = nn.BatchNorm2d(1)
        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                f = 1 / np.sqrt(m.weight.data.size()[0])
                nn.init.uniform_(m.weight, -f, f)
                nn.init.uniform_(m.bias, -f, f)

    def _make_layer(self, block: Type[Res_Block], 
                        o_dim: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(block(256, o_dim, stride))
        self.in_dim = o_dim * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_dim, o_dim))
        return nn.Sequential(*layers)

    def forward(self, x: T.Tensor):
        x = x.float()
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.res_layers(x)

        x_val = self.leaky_relu(self.bn2(self.conv2(x)))
        x_adv = self.leaky_relu(self.bn3(self.conv3(x)))

        val = self.fc2(self.leaky_relu(
                        self.fc1(x_val.view(x_val.size(0), -1))
                        ))
        adv = self.fc3(x_adv.view(x_adv.size(0), -1))
        return val + adv - adv.mean()

# ----------------------------------------------------------

class NoisyLinear(nn.Module):
    r"""A NoisyLinear layer with bias combines a deterministic and noisy stream
    Compared with epsilon-greedy, Noisy network can ignore the noisy stream
    at different rates in different parts of the state space, allowing 
    exploration with a form of self-annealing, according to 
    "Rainbow: Combining Improvements in Deep Reinforcement Learning"
    """
    def __init__(self, in_dim: int, o_dim: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_dim
        self.out_features = o_dim
        self.std_init = std_init
        self.w_mu = nn.Parameter(T.zeros(o_dim, in_dim))
        self.w_sig = nn.Parameter(T.zeros(o_dim, in_dim))
        self.register_buffer('w_eps', T.zeros(o_dim, in_dim))
        self.b_mu = nn.Parameter(T.zeros(o_dim))
        self.b_sig = nn.Parameter(T.zeros(o_dim))
        self.register_buffer('b_eps', T.zeros(o_dim))
        self.reset_params()
        self.reset_noise
        
    def reset_params(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.w_mu.data.uniform_(-mu_range, mu_range)
        self.w_sig.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.b_mu.data.uniform_(-mu_range, mu_range)
        self.b_sig.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        # standardize noise scale with x = sign(x) * sqrt(abs(x))
        x = T.randn(size, device=self.w_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.w_eps.copy_(eps_out.ger(eps_in))
        self.b_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            return nn.functional.linear(
                x, self.w_mu+self.w_sig*self.w_eps, self.b_mu+self.b_sig*self.b_eps
                )
        else:
            return nn.functional.linear(x, self.w_mu, self.b_mu)

class res_block(nn.Module):
    expansion: int = 1
    def __init__(self, in_dim: int, o_dim: int, 
                transform: Optional[nn.Module] = None, stride: int = 1) -> None:
        super(res_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_dim, o_dim, 3, stride=stride, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(o_dim, o_dim, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(o_dim)
        self.transform = transform
        self.leaky_relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        residual = x
        out = self.leaky_relu(self.bn(self.conv_1(x)))
        out = self.bn(self.conv_2(out))
        if self.transform is not None:
          residual = self.transform(residual)
        out += residual
        return self.leaky_relu(out)

class ReITA(nn.Module):
    r"""智能信号灯系统神经网络ver 2.0(基于ResNet + Rainbow理念)
    Neural Network for robust and light-weight traffic light.
    """
    def __init__(self, input_dim: Type[Union[List, Tuple]], num_layers: List[int], 
            atoms: int, h_features: int, noise_std: float, n_actions: int) -> None:
        super(ReITA, self).__init__()
        self.name = 'reita'

        # Core edifice
        # convolutional decoder
        self.dim = 64
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv_1 = nn.Conv2d(input_dim[0], self.dim, 3, 2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.dim)
        self.conv_2 = self._make_block(res_block, 64, num_layers[0])
        self.conv_3 = self._make_block(res_block, 128, num_layers[1], stride=2)
        self.conv_4 = self._make_block(res_block, 256, num_layers[2], stride=2)
        self.conv_5 = self._make_block(res_block, 512, num_layers[3], stride=2)
        
        # linear
        self.atoms = atoms
        self.n_actions = n_actions
        conv_out_size = self._get_conv_out(input_dim)
        print("flatten:", conv_out_size)
        self.fc_h_v = NoisyLinear(conv_out_size, h_features, std_init=noise_std)
        self.fc_h_a = NoisyLinear(conv_out_size, h_features, std_init=noise_std)
        self.fc_o_v = NoisyLinear(h_features, self.atoms, std_init=noise_std)
        self.fc_o_a = NoisyLinear(
            h_features, self.n_actions * self.atoms, std_init=noise_std
            )
            
        self.reset_conv()

    def _make_block(self, block: Type[res_block], o_dim: int, 
                        num_block: int, stride:int = 1) -> nn.Sequential:
        if stride != 1 or self.dim != o_dim * block.expansion:
            transform = nn.Sequential(
                nn.Conv2d(self.dim, o_dim*block.expansion, 1, stride),
                nn.BatchNorm2d(o_dim * block.expansion)
            )
        else:
            transform = None
        layers = []
        layers.append(block(self.dim, o_dim, transform, stride))
        self.dim = o_dim * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.dim, o_dim))
        return nn.Sequential(*layers)

    def _get_conv_out(self, shape):
        _output = self.leaky_relu(self.bn_1(self.conv_1(T.zeros(1, *shape))))
        _output = self.conv_5(self.conv_4(self.conv_3(self.conv_2(_output))))
        return int(np.prod(_output.size()))

    def forward(self, x, log_=False):
        # flow in decoder
        x = x.float()
        x = self.leaky_relu(self.bn_1(self.conv_1(x)))
        x = self.conv_5(self.conv_4(self.conv_3(self.conv_2(x))))
        x = x.view(x.size(0), -1)
        
        # flow in dueling network
        v = self.fc_o_v(self.leaky_relu(self.fc_h_v(x)))
        a = self.fc_o_a(self.leaky_relu(self.fc_h_a(x)))
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.n_actions, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        if log_:
            q = nn.functional.log_softmax(q, dim=2)
        else:
            q = nn.functional.softmax(q, dim=2)
        return q
        
    def reset_conv(self):
        # Initialize conv2d weights & biases
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def reset_noise(self):
        for n, m in self.named_children():
            if 'fc' in n:
                m.reset_noise()
