import tensorflow as tf
import argparse
import collections
import copy
import numpy as np
import os,sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append("./lib")
sys.path.append("./common")

import env as Env
from tensorboardX import SummaryWriter

from lib import agent, action, experience, parameters, saver, tracker, wrappers

# Global Variable:
params = parameters.Constants
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

#build up Neutral Network
'''
Deep Q learning  off policy
'''
class DeepNetwork:
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay=0.1,e_greedy=0.1,replace_target_iter=300,memory_size=300,batch_size=32,e_greedy_increment=None,output_graph=False,):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        self.eposilon = 0 if e_greedy_increment is not None else self.eposilon_max

        #calculate total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        #store your network param
        self._build_net()
        t_params = tf.get_collection('target_net_params')# target network params
        e_params = tf.get_collection('eval_net_params')#eval network params
        self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]#更新目标网络参数

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self,inputs,regularizer):
        '''
        build eval network
        '''
        self.s = tf.placeholder()
        # -------------- 创建 eval 神经网络, 及时提升参数 --------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions],
                                       name='Q_target')  # 用来接收 q_target 的值, 这个之后会通过计算得到
        self.activation_l = 'RELU'
        self.inputs = inputs

        #-------------- Lenet结构 --------------
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 256, \
                tf.random_normal_initializer(0.2, 0.3), tf.constant_initializer(0.1)  # config of layers

            # eval_net 的第一层. collections 是在更新 target_net 参数时会用到,n_l1=256是我决定的
            with tf.variable_scope('l1-conv1'):
                # 第一层为卷积层，过滤器大小为5x5，深度为6，不够用0补全。步长为1。
                conv1_weights = tf.get_variable('weight', [5, 5, c, 6],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv1_biases = tf.get_variable('bias', [6], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(inputs, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
                relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

                # w1 = tf.get_variable('w1', [self.n_features, 256], initializer=w_initializer, collections=c_names)
                # b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # eval_net 的第二层. collections 是在更新 target_net 参数时会用到.第二层为池化层，步长为1，不够用0补全。
            with tf.variable_scope('l2-pool1'):
                pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                # conv1 = tf.layers.con2d(l1,filters=16,kernel_size=5,strides=1,padding='same',activation=tf.nn.relu)
                # w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                # b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                # self.q_eval = tf.matmul(l1, w2) + b2

            with tf.variable_scope('l3-conv2'):
                conv2_weights = tf.get_variable('weight', [5, 5, 6, 16],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv2_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
                relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
                # l1 = tf.layers.dense(inputs,units=100,activation=tf.nn.relu)

            with tf.variable_scope('l4-pool2'):
                pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
                # l1 = tf.layers.dense(inputs,units=100,activation=tf.nn.relu)

            # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为5×5×16的矩阵，然而第五层全连接层需要的输入格式
            # 为向量，所以我们需要把代表每张图片的尺寸为5×5×16的矩阵拉直成一个长度为5×5×16的向量。
            # 举例说，每次训练64张图片，那么第四层池化层的输出的size为(64,5,5,16),拉直为向量，nodes=5×5×16=400,尺寸size变为(64,400)

            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool2, [-1, nodes])

            # 第五层：全连接层，nodes=5×5×16=400，400->120的全连接
            # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×400->64×120
            # 训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题。
            # 这和模型越简单越不容易过拟合思想一致，和正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，以此达到避免过拟合思想一致。
            # 最后训练时没有采用dropout，dropout项传入参数设置成了False，因为训练和测试写在了一起没有分离，不过可以尝试。
            with tf.variable_scope('l5-fc1'):
                fc1_weights = tf.get_variable('weight', [nodes, 512],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                # if regularizer != None:
                #     tf.add_to_collection('losses', regularizer(fc1_weights))
                fc1_biases = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.1))
                fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

                # if train:
                #     fc1 = tf.nn.dropout(fc1, 0.5)

            # 第六层：全连接层，120->84的全连接
            # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84
            with tf.variable_scope('l6-fc2'):
                fc2_weights = tf.get_variable('weight', [512, 256],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                # if regularizer != None:
                #     tf.add_to_collection('losses', regularizer(fc2_weights))
                fc2_biases = tf.get_variable('bias', [256], initializer=tf.truncated_normal_initializer(stddev=0.1))
                fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

                # if train:
                #     fc2 = tf.nn.dropout(fc2, 0.5)

            # 第七层：全连接层（近似表示），84->10的全连接
            # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
            # 即得到最后的分类结果。
            with tf.variable_scope('l7-fc3'):
                fc3_weights = tf.get_variable('weight', [256, 8],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                # if regularizer != None:
                #     tf.add_to_collection('losses', regularizer(fc3_weights))
                fc3_biases = tf.get_variable('bias', [8], initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.q_eval = tf.nn.relu(fc2, fc3_weights) + fc3_biases
            return self.q_eval


        with tf.variable_scope('loss'):  # 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):  # 梯度下降,Adams
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)



        '''
        build target network///先不管
        '''
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # target_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # target_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2


    def store_transition(self):

        pass

    def choose_action(self):
        batch_size, n_actions = self.q_eval.shape
        actions = self.selector(self.q_eval)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions

    def learn(self):

        pass

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--name", default='network_checkpoint', type=str, dest='name', help='train period name')
#     parser.add_argument("--resume", default = "checkpoint.pth", type = str, dest='path', help= 'path to latest checkpoint')
#     parser.add_argument("--skip", default=180, type=int, dest='frameskip', help='frameskip of env')
#     parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
#     args = parser.parse_args()
#     Train(args.cuda, args.name, args.path, args.frameskip)
