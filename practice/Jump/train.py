#######################################################################################################
# 2020-06-25
# ioaniu/Jeff Young
# Modified from 
#    https://https://https://github.com/shivaverma/Orbit/tree/master/Paddle/agent.py
#    https://https://github.com/PaddlePaddle/PARL/blob/develop/examples/DQN/replay_memory.py
#######################################################################################################

import numpy as np
import parl
from parl.utils import logger

from jump import JumpGame

from jump_model import JumpModel
from jump_agent import JumpAgent

from replay_memory import ReplayMemory
import matplotlib.pyplot as plt



LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 100000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 64
LEARNING_RATE = 0.01
GAMMA = 0.99  # discount factor of reward



def run_episode(agent, env, rpm,render=False):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        
#        print(action)
        reward, next_obs, isOver = env.step(action)
        # next_obs, reward, isOver = env.step(action)
        rpm.append((obs, action, reward, next_obs, isOver))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_isOver) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_isOver)

        total_reward += reward
        obs = next_obs
        if render:
            env.render()
        if isOver:
            print("episode: {}, score: {}".format(step, total_reward))
            break
            
        if step > 1000:
        	break
    return total_reward


def evaluate(agent, env, render=False):
    # test part, run 5 episodes and average
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        isOver = False
        while not isOver:
            action = agent.predict(obs)
            if render:
                env.render()
            reward, obs, isOver = env.step(action)
            episode_reward += reward
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
	
    env = JumpGame()
    action_dim = 2
    obs_shape = len(env.reset())

    rpm = ReplayMemory(MEMORY_SIZE)

    model = JumpModel(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(
        model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = JumpAgent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.1,  # explore
        e_greed_decrement=1e-6
    )  # probability of exploring is decreasing during training
    
    # 加载模型
#    save_path = 'jump_best_model.ckpt'
#    agent.restore(save_path)

    while len(rpm) < MEMORY_WARMUP_SIZE:  # warm up replay memory
        run_episode(agent, env, rpm)

    max_episode = 1000  

    # start train
    episode = 0
    
    train_total_reward = []
    best_eval_reward = 0
    while episode < max_episode:
        # train part
        for i in range(0, 50):
            total_reward = run_episode(agent, env, rpm)
            episode += 1
            train_total_reward.append(total_reward)  
            print(total_reward)
            # plt.plot(train_total_reward)
            # plt.draw()
            # plt.pause(0.001)
            # plt.show()
            

        eval_reward = evaluate(agent, env)
        logger.info('episode:{}    test_reward:{}'.format(
            episode, eval_reward))
        if best_eval_reward < eval_reward:
            best_eval_reward = eval_reward
            save_path = 'jump_best_model.ckpt'
            agent.save(save_path)
            
    # 训练结束，保存模型
    save_path = 'jump_model.ckpt'
    agent.save(save_path)


if __name__ == '__main__':
    main()