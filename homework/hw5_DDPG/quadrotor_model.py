# quadrotor_model.py
# ioaniu / Jeff Young
# 2020-6-25


import parl
from parl import layers

class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):

        hid = self.fc1(obs)
        hid = self.fc2(hid)
        logits = self.fc3(hid)
        return logits

class CriticModel(parl.Model):
    def __init__(self):
        hid1_size = 128
        hid2_size = 128
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='tanh')
        self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)
        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        hid = self.fc2(hid)
        Q = self.fc3(hid)
        Q = layers.squeeze(Q, axes=[1])
        return Q


class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()