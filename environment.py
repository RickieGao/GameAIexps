import tensorflow as tf
import numpy as np
import math
import  time
import sys
from PolicyGradient import PGmodel
import math
# from gym import spaces, logger
# from gym.utils import seeding
import numpy as np
from numpy import linalg as LA

# TIME_SLOTS = 1
NUM_CHANNELS =3
NUM_USERS =2
NUM_BS = 2
ATTEMPT_PROB1=0.6
ATTEMPT_PROB2=0.8
# GAMMA = 0.90
pmax=6

class env_network:
    def __init__(self, num_channels=3, num_users=2, num_bs=2, transmit_power=1, attempt_prob1=0.6, attempt_prob2=0.8):
        self.NUM_CHANNELS = num_channels  # 信道数
        self.NUM_USERS = num_users  # 用户数
        self.NUM_BS = num_bs  # 基站数
        self.pmax = 7  # 基站存储能量的最大值
        self.init_power = 1
        self.transmit_power = transmit_power
        self.ATTEMPT_PROB1 = attempt_prob1
        self.ATTEMPT_PROB2 = attempt_prob2
        self.REWARD = 1#单步奖励值
        self.step_count = 0  # which step

        self.rest_power1 = 0
        self.rest_power2 = 0
        # self.harvest_power1 = None
        # self.harvest_power2 = None

        self.action_space = np.arange(self.NUM_CHANNELS*self.NUM_USERS*self.NUM_BS)#定义动作空间
        self.action = np.random.randint(0,2,size=self.NUM_BS*self.NUM_CHANNELS*self.NUM_USERS)
        self.obs = None          #系统状态
        self.channel_state_matrix = np.zeros(self.NUM_BS*self.NUM_CHANNELS*self.NUM_USERS)#描述信道占用状态
        # self.channel_obs = []  #信道状态
        # self.obs_space = np.arange((self.NUM_BS*self.NUM_CHANNELS*self.NUM_USERS),np.int32)

        # self.h_matrix = np.zeros((num_bs,num_channels,num_users))#信道占用状态转移矩阵，0表示占用，1表示空闲
        # self.generate_channel_data()
        # self.generate_energy_power()
    def render(self):
        pass

    def sample(self):
        x = np.zeros(self.NUM_BS*self.NUM_CHANNELS*self.NUM_USERS)
        return x

    def reset(self):
        channel_state_matrix = np.random.randint(0, 2, size=[self.NUM_BS, self.NUM_CHANNELS, self.NUM_USERS]).flatten()
        rest_power1 = np.random.randint(0, self.pmax)
        rest_power2 = np.random.randint(0, self.pmax)
        harvest_power1 = math.floor(0.8 * self.init_power + np.random.random())
        harvest_power2 = math.floor(1.2 * self.init_power + np.random.random())
        obs = np.hstack([rest_power1, rest_power2, channel_state_matrix, harvest_power1, harvest_power2])
        return obs,channel_state_matrix,rest_power1,rest_power2

    def step(self,action,channel_state,res_power1,res_power2):
        self.action = action
        print('00000000')
        print(self.action)
        self.channel_state_matrix = channel_state
        print('11111111')
        print(self.channel_state_matrix)
        self.rest_power1 = res_power1
        self.rest_power2 = res_power2
        # 两个基站当前时刻分别吸收的能量
        harvest_power1 = math.floor(0.8 * self.init_power + np.random.normal(0, 1))
        harvest_power2 = math.floor(1.2 * self.init_power + np.random.normal(0, 1))
        #12个action对应的reward
        reward = np.zeros([self.NUM_BS,self.NUM_CHANNELS, self.NUM_USERS]).flatten()

        if self.rest_power1 == 0 and self.rest_power2 >= 2 * self.transmit_power:
            self.action[0:6] = 0
            for j in range(self.NUM_BS * self.NUM_CHANNELS * self.NUM_USERS):
                if self.channel_state_matrix[j] * self.action[j] == 1:
                    reward[j] = 12
                else: reward[j] = -1
            self.rest_power1 = harvest_power1
            self.rest_power2 = min(self.rest_power2 + harvest_power2 - 2*self.transmit_power, self.pmax)


        if self.rest_power2 == 0 and self.rest_power1 >= 2 * self.transmit_power:
            self.action[6:12] = 0
            for j in range(self.NUM_BS * self.NUM_CHANNELS * self.NUM_USERS):
                # print('second')
                # print(j)
                # print(channel_state_matrix[j] * self.action[j])
                if self.channel_state_matrix[j] * self.action[j] == 1:
                    reward[j] = 12
                else: reward[j] = -1
                # print(reward)
            self.rest_power1 = min(self.rest_power1 + harvest_power1 - 2 * self.transmit_power, self.pmax)
            self.rest_power2 = harvest_power2


        if self .pmax > self.rest_power1 and self .pmax > self.rest_power2 and self .transmit_power  <= self.rest_power1 and self .transmit_power  <= self.rest_power2:
            for j in range(self.NUM_BS * self.NUM_CHANNELS * self.NUM_USERS):
                if self.channel_state_matrix[j] * self.action[j] == 1:
                    reward[j] = 12
                else: reward[j] = -1
                # print(reward)
            self.rest_power1 = min(self.rest_power1 + harvest_power1 - self.transmit_power, self.pmax)
            self.rest_power2 = min(self.rest_power2 + harvest_power2 - self.transmit_power, self.pmax)

        if self.rest_power2 == 0 and self.rest_power1 >= self.transmit_power and self.rest_power1 < 2 * self.transmit_power :
            self.action[6:12] = 0
            for j in range(self.NUM_BS * self.NUM_CHANNELS * self.NUM_USERS):
                # print(channel_state_matrix[j] * self.action[j])
                if self.channel_state_matrix[j] * self.action[j] == 1:
                    reward[j] = 12
                else:
                    reward[j] = -1
                # print(reward)
            self.rest_power1 = min(self.rest_power1 + harvest_power1 - self.transmit_power,self.pmax)
            self.rest_power2 = harvest_power2

        if self.rest_power1 == 0 and self.rest_power2 >= self.transmit_power and self.rest_power2 < 2 * self.transmit_power :
            self.action[0:6] = 0
            for j in range(self.NUM_BS * self.NUM_CHANNELS * self.NUM_USERS):
                # print(channel_state_matrix[j] * self.action[j])
                if self.channel_state_matrix[j] * self.action[j] == 1:
                    reward[j] = 12
                else:
                    reward[j] = -1
                # print(reward)
            self.rest_power1 = harvest_power1
            self.rest_power2 = min(self.rest_power2 + harvest_power2 - self.transmit_power, self.pmax)

        done = (self.transmit_power <= self.pmax) and (self.rest_power1 >= self.transmit_power) and (self.rest_power1 <= self.pmax) and (self.rest_power2 >= self.transmit_power) and (self.rest_power2 <= self.pmax)
        # abort =(self.rest_power1 == 0) or (self.rest_power1 < self.transmit_power) or (self.rest_power2 == 0) or (self.rest_power2 < self.transmit_power)

        for i in range(self.NUM_BS * self.NUM_CHANNELS * self.NUM_USERS):
            if self.channel_state_matrix[i] == 0:
               prob1 = np.random.uniform(0, 1)
               prob1 = round(prob1, 2)
               if prob1 <= self.ATTEMPT_PROB1:
                   self.channel_state_matrix[i] = 0
               else:
                   self.channel_state_matrix[i] = 1
            else:
               prob2 = np.random.uniform(0, 1)
               prob2 = round(prob2, 2)
               if prob2 <= self.ATTEMPT_PROB2:
                   self.channel_state_matrix[i] = 1
               else:
                   self.channel_state_matrix[i] = 0
        print('22222222')
        print(self.channel_state_matrix)

        self.obs = np.hstack([self.rest_power1, self.rest_power2, self.channel_state_matrix, harvest_power1, harvest_power2])

        # if abort == True:
        #     done = False
        #     reward = [-2 for i in reward[0:12]]
        # elif done:
        #     reward = [1 for i in reward[0:12]]

        return self.obs,self.channel_state_matrix, reward, self.rest_power1,self.rest_power2,done











