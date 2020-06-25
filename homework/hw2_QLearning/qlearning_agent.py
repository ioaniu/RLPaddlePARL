# qlearning_agent.py
# ioaniu / Jeff Young
# 2020-6-25

import numpy as np 

class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n      # 动作维度，有几个动作可选
        self.lr = learning_rate # 学习率
        self.gamma = gamma      # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        # 利用epsilon贪心算法，使其一定概率下，随机选择动作。
        if(np.random.uniform(0, 1) < (1.0 - self.epsilon)):
            action = self.predict(obs); #查表按模型预测来选择动作
        else:
            action = np.random.choice(self.act_n);#随机选择一个动作.
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]   #获取输入观察值下可能的Q值
        maxQ = np.max(Q_list)     #获取最大值
        action_list = np.where(Q_list == maxQ)[0]  # 获取maxQ值对应的action，可能有多个
        action = np.random.choice(action_list)#随机选择maxQ值对应的action中的一个作为输出action
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        #根据obs 和action，获得预测Q值
        predict_Q = self.Q[obs, action]
        # 是否结束？
        if done:
            target_Q = reward # 奖励
        else:
            # Q Learning，计算目标Q值
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :]) # Q Learning
                    
        # 更新Q值公式
        self.Q[obs, action] = self.Q[obs, action] + self.lr * (target_Q - predict_Q)


    # 保存Q表格数据到文件
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')
    
    # 从文件中读取数据到Q表格中
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')