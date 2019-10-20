import gym
import pygame
import cv2
import random
import numpy as np
# clock = pygame.time.Clock()
# list1 = [[295, 298], [283, 174], [293, 173], [305, 297]]
# list2 = np.asarray(list1)
# list3 = list2.tolist()
# list4 = sorted(list3)
# env = gym.make("Skiing-v0")
# action = env.action_space
# print(action)
# print(env.observation_space)
# observation = env.reset()
# env.render()
# print(observation)

# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     # observation = env.reset()
#     env.reset()
#     for t in range(100):
#         env.render()
#         clock.tick(30)
#         # print(observation)
#         # action = env.action_space.sample()
#         action = 1
#         """
#         skiing:
#         0:do nothing
#         1:right
#         2:left
#         CartPole:
#
#         """
#         observation, reward, done, info = env.step(action)
#         print("/reward", reward, "/done", done)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# ob = env.reset()
# ob = cv2.resize(ob, (80, 80))
# ret, x_t = cv2.threshold(ob, 1, 255, cv2.THRESH_BINARY)
# for t in range(10):
#     print(random.randrange(3))
ob_colored = cv2.imread(r"/home/gaozh/projects/gym-DQN/image_process_lab/sample1.png")
cv2.imshow("test", ob_colored)
cv2.waitKey(0)
ob = cv2.cvtColor(cv2.resize(ob_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
x_t1 = cv2.resize(ob, (80, 80))
ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
x_t1 = np.reshape(x_t1, (80, 80, 1))

print(type(x_t1))
