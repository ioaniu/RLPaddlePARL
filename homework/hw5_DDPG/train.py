# train.py
# ioaniu / Jeff Young
# 2020-6-25


import numpy as np

from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境

from ddpg import DDPG
from quadrotor_agent import QuadrotorAgent
from quadrotor_model import QuadrotorModel

# 分段衰减，即由给定step数分段呈阶梯状衰减，每段内学习率相同
BOUNDARIES = [50000, 1000000, 200000]
ACTOR_LR_VALUE = [1e-3, 1e-4, 1e-5, 1e-6]
CRITIC_LR_VALUE = [1e-2, 1e-3, 1e-4, 1e-5]

# ACTOR_LR = 0.0002   # Actor网络更新的 learning rate
# CRITIC_LR = 0.001   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward


NOISE = 0.05         # 动作噪声方差
OFFSET_FACTOR = 0.2  # 补偿系数：输出5个action，其中第一个有主值，后四个有补偿值


def run_episode(env, agent, rpm, render=False):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        
        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, NOISE), -1.0, 1.0)

        # action_tmp = action[0] +action[-(len(action)-1):,] * OFFSET_FACTOR;
        # action = np.append(action[0],action_tmp)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        # action = action_mapping(action, env.action_space.low[0],
        #                         env.action_space.high[0])        
        # next_obs, reward, done, info = env.step(action)

        main_action = action[0]
        sub_action = action[1:]
        sub_action = main_action + sub_action * OFFSET_FACTOR;
        sub_action = np.clip(sub_action, -1.0, 1.0)
        sub_action = action_mapping(sub_action, env.action_space.low[0],
                                env.action_space.high[0])        
        next_obs, reward, done, info = env.step(sub_action)

        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if render:
            env.render()

        if done:
            break
    return total_reward, steps

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)

            # action_tmp = action[0] +action[-(len(action)-1):,] * OFFSET_FACTOR;
            # action = np.append(action[0],action_tmp)
            
            # #输出限制在 [-1.0, 1.0] 范围内
            # action = np.clip(action, -1.0, 1.0)
            # action = action_mapping(action, env.action_space.low[0], 
            #                         env.action_space.high[0])
            # next_obs, reward, done, info = env.step(action)

            main_action = action[0]
            sub_action = action[1:]
            sub_action = main_action + sub_action * OFFSET_FACTOR;
            sub_action = np.clip(sub_action, -1.0, 1.0)
            sub_action = action_mapping(sub_action, env.action_space.low[0],
                                    env.action_space.high[0])        
            next_obs, reward, done, info = env.step(sub_action)

            obs = next_obs
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                        
            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)



# 创建飞行器环境
env = make_env("Quadrotor", task="hovering_control")
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# 输出action增加1维作为main action, 其它为补偿值
act_dim = act_dim + 1

# 根据parl框架构建agent
model = QuadrotorModel(act_dim)
algorithm = DDPG(
    model, gamma=GAMMA, tau=TAU,boundaries=BOUNDARIES, actor_lrvalue=ACTOR_LR_VALUE, critic_lrvalue=CRITIC_LR_VALUE)
agent = QuadrotorAgent(algorithm, obs_dim, act_dim)


# parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)



# 启动训练
test_flag = 0
total_steps = 0

# # 加载模型
# ckpt = 'model_dir/steps_best.ckpt' 
# agent.restore(ckpt)

best_reward = None

while total_steps < TRAIN_TOTAL_STEPS:
    train_reward, steps = run_episode(env, agent, rpm, render=True)
    total_steps += steps
    # logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward

    if total_steps // TEST_EVERY_STEPS >= test_flag: # 每隔一定step数，评估一次模型
        while total_steps // TEST_EVERY_STEPS >= test_flag:
            test_flag += 1
 
        evaluate_reward = evaluate(env, agent, render=True)
        logger.info('Steps {}, Test reward: {}'.format(
            total_steps, evaluate_reward)) # 打印评估的reward

        # 每评估一次，就保存一次模型，以训练的step数命名
        ckpt = 'model_dir/steps_{}.ckpt'.format(total_steps)
        agent.save(ckpt)
        
        # 保存最好的一次模型
        if best_reward < evaluate_reward or best_reward == None:
            best_reward = evaluate_reward
            
            ckpt = 'model_dir/steps_best.ckpt'.format(total_steps)
            agent.save(ckpt)
