# ddpg.py
# ioaniu / Jeff Young
# 2020-6-25


import os
import numpy as np

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放
from copy import deepcopy


# from parl.algorithms import DDPG
class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 boundaries=None,
                 actor_lrvalue=None,
                 critic_lrvalue=None):
        """  DDPG algorithm
        
        Args:
            model (parl.Model): actor and critic 的前向网络.
                                model 必须实现 get_actor_params() 方法.
            gamma (float): reward的衰减因子.
            tau (float): self.target_model 跟 self.model 同步参数 的 软更新参数
            actor_lr (float): actor 的学习率
            critic_lr (float): critic 的学习率
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)

        # assert isinstance(actor_lr, float)
        # assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.boundaries = boundaries
        self.actor_lrvalue = actor_lrvalue
        self.critic_lrvalue = critic_lrvalue

        self.model = model
        self.target_model = deepcopy(model)

    def predict(self, obs):
        """ 使用 self.model 的 actor model 来预测动作
        """
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 用DDPG算法更新 actor 和 critic
        """
        actor_cost = self._actor_learn(obs)
        critic_cost = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        return actor_cost, critic_cost

    def _actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.value(obs, action)
        cost = layers.reduce_mean(-1.0 * Q)

        # optimizer = fluid.optimizer.AdamOptimizer(self.actor_lrvalue)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=fluid.layers.piecewise_decay(boundaries=self.boundaries, values=self.actor_lrvalue),
                    regularization=fluid.regularizer.L2Decay(1e-4))

        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        next_action = self.target_model.policy(next_obs)
        next_Q = self.target_model.value(next_obs, next_action)

        terminal = layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        target_Q.stop_gradient = True

        Q = self.model.value(obs, action)
        cost = layers.square_error_cost(Q, target_Q)
        cost = layers.reduce_mean(cost)
        
        # optimizer = fluid.optimizer.AdamOptimizer(self.critic_lrvalue)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=fluid.layers.piecewise_decay(boundaries=self.boundaries, values=self.critic_lrvalue),
                    regularization=fluid.regularizer.L2Decay(1e-4))

        optimizer.minimize(cost)
        return cost

    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        """ self.target_model从self.model复制参数过来，可设置软更新参数
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(
            self.target_model,
            decay=decay,
            share_vars_parallel_executor=share_vars_parallel_executor)