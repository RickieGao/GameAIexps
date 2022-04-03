import csv
from decimal import Decimal
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
WEIGHT_R = 0.8


def read_file(file_name):
	with open(file_name, "r") as file:
		reader = csv.reader(file)
		next(reader)
		recorder1, recorder2 = [], []
		for row in reader:
			recorder1.append(int(row[0]))
			recorder2.append(float(Decimal(row[1]).quantize(Decimal('0.00'))))
	return recorder1, recorder2


def smooth(target, wight):
	smoothed = []
	last = target[0]
	for value in target:
		smoothed_val = last * wight + (1 - wight) * value
		smoothed.append(smoothed_val)
		last = smoothed_val
	return smoothed


# DQN_timesteps, DQN_Max_Q = read_file(r"plane\run-Exp10Graph2-tag-50kper_average_qMax_50kper_average_qMax.csv")
# rule_timesteps, rule_Max_Q = read_file(r"plane\run-Exp13Graph-tag-50kper_average_qMax_50kper_average_qMax.csv")
DQN_episode, DQN_reward = read_file(r"grid/log_eps-greedy_traces_9.csv")
rule_episode, rule_reward = read_file(r"grid/log_eps-greedy_traces_8.csv")

# smoothed_DQN_Max_Q = smooth(DQN_Max_Q, WEIGHT_Q)
# smoothed_rule_Max_Q = smooth(rule_Max_Q, WEIGHT_Q)
smoothed_DQN_reward = smooth(DQN_reward, WEIGHT_R)
smoothed_rule_reward = smooth(rule_reward, WEIGHT_R)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.figure()
plt.xlabel("游戏局数", fontsize=16)
plt.ylabel("单局游戏奖励", fontsize=16)
plt.title("格子世界", fontsize=16)
dqn, = plt.plot(DQN_episode, smoothed_DQN_reward, color="#cc3311")
dqn_rule, = plt.plot(rule_episode, smoothed_rule_reward, color="#0077bb")
plt.legend(handles=[dqn, dqn_rule], labels=['Baseline', 'SML'],  loc='lower right')

plt.grid(ls='--')
smoothed_DQN_reward_std = np.std(smoothed_DQN_reward)
smoothed_rule_reward_std = np.std(smoothed_rule_reward)
plt.fill_between(DQN_episode, smoothed_DQN_reward - smoothed_DQN_reward_std,
					smoothed_DQN_reward + smoothed_DQN_reward_std, facecolor='#cc3311', alpha=0.25)
plt.fill_between(rule_episode, smoothed_rule_reward - smoothed_rule_reward_std,
					smoothed_rule_reward + smoothed_rule_reward_std, facecolor='#0077bb', alpha=0.25)
# plt.axvline(x=1480, color='black', linestyle="--")
# plt.axhline(y=650, color='black', linestyle="--")
# plt.figure()
# plt.title("Average Q on AircraftShoot")
# plt.xlabel("Training Epochs")
# plt.ylabel("Average Action Value (Q)")
# dqn_q, = plt.plot(DQN_timesteps, smoothed_DQN_Max_Q, color="#cc3311")
# rule_q, = plt.plot(rule_timesteps, smoothed_rule_Max_Q, color="#0077bb")
# plt.legend(handles=[dqn_q, rule_q], labels=['original DQN', 'our model'],  loc='lower right')
plt.show()
