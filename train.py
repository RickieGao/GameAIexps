import tensorflow as tf
import numpy as np
import environment
import PolicyGradient
import matplotlib.pyplot as plt
import os
import time
import sys
# sys.path.append("D:\\project\\pycharm project\\comm\\demo.py")
# sys.path.append("D:\\project\\pycharm project\\comm\\PolicyGradient.py")
# np.random.seed(1000)
# tf.set_random_seed(1000)
NUM_CHANNELS =3
NUM_USERS =2
NUM_BS = 2
ATTEMPT_PROB1=0.6
ATTEMPT_PROB2=0.8
pmax=7

learning_rate=0.0001  #learning rate
gamma=0.95          #discount factor

action_space = 12
obs_space = 16
MAX_EPISODES = 500
MAX_EP_STEPS = 10

#reseting default tensorflow computational graph
tf.reset_default_graph()
#initializing the environment
env = environment.env_network(NUM_BS,NUM_CHANNELS,NUM_USERS,ATTEMPT_PROB1,ATTEMPT_PROB2,pmax)
#initializing policy gradient network
mainPG = PolicyGradient.PGmodel(learning_rate=learning_rate,n_actions=action_space, n_features=obs_space,reward_decay = 0.95 )
# print(env.action) #输出1*12维的0或1数组
# print(env.action_space)#输出[ 0  1  2  3  4  5  6  7  8  9 10 11]，12个动作的下标
# print(env.obs)

#save object to save the checkpoints of the PGmodel to disk
saver = tf.train.Saver()

#initializing the sessions
sess = tf.Session()

#initializing all the tensorflow variables
sess.run(tf.global_variables_initializer())

##########################################################################
####                      main simulation loop                    ########
for i in range(MAX_EPISODES):
    # 获取回合 i_episode 第一个整个系统 observation和信道状态的信息
    obs, channel_state, res_power1,res_power2 = env.reset()
    print('the whole system state')
    print(obs)
    print('打印出信道初始状态')
    print(channel_state)
    for j in range(MAX_EP_STEPS):
        # to sample random actions for each user
        action = env.sample()
        print('action sample')
        print(action)

        # choosing action with max probability
        act = mainPG.choose_action(obs)  # 选行为
        print('选动作')
        print(act)
        action[act] = 1
        print('打印出动作')
        print(action)

        # 获取下一个state
        obs_,channel_state_, reward, rest_power1,rest_power2,done = env.step(action,channel_state,res_power1,res_power2)
        print('下一个state')
        print(obs_)
        print('下一刻信道状态信息')
        print(channel_state_)
        print('回报')
        print(reward)
        # 保存这一组记忆
        mainPG.store_transition(obs, action, reward)

        if done:
            ep_rs_sum = np.sum(mainPG.ep_rs)
            print('@@@@')
            print(ep_rs_sum)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            print("episode:", i, "  reward:", running_reward )

            vt = mainPG.learn()
            print('balabalabala')

            if i == 0:
                plt.plot(vt)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
        obs = obs_
        channel_state = channel_state_


