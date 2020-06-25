#######################################################################################################
# 2020-06-25
# Author: ioaniu/Jeff Young
# Modified from https://https://github.com/PaddlePaddle/PARL/blob/develop/examples/DQN/cartpole_model.py
#######################################################################################################

import parl
from parl import layers

class PaddleModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 64
        hid2_size = 64
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='sigmoid')

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
