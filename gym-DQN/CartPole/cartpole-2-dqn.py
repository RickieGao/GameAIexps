# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 20:12:03 2019

@author: zhangfan
"""
#导入openAI gym 以及 TensorFlow
import gym
import numpy as np
import tensorflow as tf

#用gym.make('CartPole-v0')导入gym定义好的环境，对于更复杂的问题则需要自定义环境
env = gym.make('CartPole-v0')

#第一步不用agent，采用随机策略进行对比
env.reset() #初始化环境
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    obsevation, reward, done, _ = env.step(np.random.randint(0, 2))
    #np.random.randint创建随机action，env.step执行action
    reward_sum += reward
    #最后一个action也获得奖励
    if done:
        random_episodes += 1
        print("Reward for this episodes was:", reward_sum)
        reward_sum = 0  #重置reward
        env.reset()

# PART2:AGENT. Agent是一个简单的ANN with one hidden layer.
# 可以使用更复杂的深度神经网络
H = 50  # 50个neure
batch_size = 25
learning_rate = 0.1
D = 4  # observation维度为4
gamma = 0.99  # discount rate

# 定义策略网络具体结构：输入observation，输出选择action的概率
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
# 隐藏层使用ReLu激活
W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)
# 输出层使用Sigmoid将输出转化为概率

# 定义优化器，梯度占位符，采用batch training更新参数
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1_grad = tf.placeholder(tf.float32, name="batch_grad1")
W2_grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1_grad, W2_grad]
tvars = tf.trainable_variables()
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


# 计算每一个action的折现后的总价值
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# 计算损失函数
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
# action的潜在价值
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
# loglik为action的对数概率，P(act=1) = probablility,P(act=0) = 1-probablility
# action=1，loglik = tf.log(probability)
# action=0，loglik = tf.log(1-probability)
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)
# tvars用于获取全部可训练参数，tf.gradients求解参数关于loss的梯度


xs = []  # observation的列表
ys = []  # label的列表， label = 1 - action
drs = []  # 每个action的reward
reward_sum = 0  # 累计reward
episode_num = 1  # 每次实验index
total_episodes = 10000  # 总实验次数

# 创建会话
with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)  # 初始化状态
    observation = env.reset()  # 重置环境

    gradBuffer = sess.run(tvars)
    # 创建存储参数梯度的缓冲器，执行tvars获取所有参数
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0  # 将所有参数全部初始化为零
    # 进入实验循环
    while episode_num <= total_episodes:
        # 当某batch平均reward>100时，对环境进行展示
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        # 将observation变形为网络输入格式
        x = np.reshape(observation, [1, D])
        # 计算action=1的概率tfprob
        tfprob = sess.run(probability, feed_dict={observations: x})
        # （0,1）随机抽样，若随机值小于tfprob，action=1
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)  # 将observation加入列表xs
        y = 1 - action
        ys.append(y)  # 将label加入列表ys
        observation, reward, done, info = env.step(action)
        # env.step执行action，获取observation,reward,done,info
        reward_sum += reward
        drs.append(reward)  # 将reward加入列表drs

        # done=True 即实验结束
        if done:
            episode_num += 1  # 一次实验结束，index+1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []

            # 计算每一步的总价值，并标准化为均值为0标准差为1的分布
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # 将epx epy epr 输入神经网络，newGrads求梯度
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad
                # 将梯度叠加进gradBuffer

            # 当试验次数达到batch_size整数倍时
            if episode_num % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1_grad: gradBuffer[0], W2_grad: gradBuffer[1]})
                # updateGrads将gradBuffer梯度更新到模型参数中

                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                    # 清空gradBuffer，为下一个batch做准备

                print('Average reward for episode %d : %f.' % (episode_num, reward_sum / batch_size))
                if reward_sum / batch_size > 200:
                    print("Task solved in", episode_num, 'episodes!')
                    break
                reward_sum = 0

            observation = env.reset()
            # 每次实验结束，重置任务环境
