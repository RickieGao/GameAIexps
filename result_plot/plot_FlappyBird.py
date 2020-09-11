import csv
from decimal import Decimal
import matplotlib
from numpy import trapz
from matplotlib import pyplot as plt
import numpy as np

WEIGHT_R = 0.9
WEIGHT_Q = 0.7


def read_file(file_name):
	with open(file_name, "r") as file:
		reader = csv.reader(file)
		next(reader)
		recorder1, recorder2 = [], []
		for row in reader:
			recorder1.append(int(row[1]))
			recorder2.append(float(Decimal(row[2]).quantize(Decimal('0.00'))))
	return recorder1, recorder2


def smooth(target, wight):
	smoothed = []
	last = target[0]
	for value in target:
		smoothed_val = last * wight + (1 - wight) * value
		smoothed.append(smoothed_val)
		last = smoothed_val
	return smoothed


def mute(x, y, xlimit):
	coordinate = list(zip(x, y))
	coordinate = [(x, y) for (x, y) in coordinate if x <= xlimit]
	return coordinate


def mute_y(x, y, ylimit):
	coordinate = list(zip(x, y))
	coordinate = [(x, y) for (x, y) in coordinate if y <= ylimit]
	return coordinate


DQN_timesteps, DQN_Max_Q = read_file(r"bird\DQN-tag-50kper_average_qMax_50kper_average_qMax.csv")
rule_timesteps, rule_Max_Q = read_file(r"bird\run-DQN_rule2true-tag-50kper_average_qMax_50kper_average_qMax.csv")
DQN_episode, DQN_reward = read_file(r"bird\run-DQN-tag-reward_per_life_reward_per_life.csv")
rule_episode, rule_reward = read_file(r"bird\run-DQN_rule-tag-reward_per_life_reward_per_life.csv")

DQN_timesteps = DQN_timesteps[:len(rule_timesteps)]
DQN_Max_Q = DQN_Max_Q[:len(rule_Max_Q)]

smoothed_DQN_Max_Q = smooth(DQN_Max_Q, WEIGHT_Q)
smoothed_rule_Max_Q = smooth(rule_Max_Q, WEIGHT_Q)
smoothed_DQN_reward = smooth(DQN_reward, WEIGHT_R)
smoothed_rule_reward = smooth(rule_reward, WEIGHT_R)

# DQN_muted_unsmoothed_coordinate = mute_y(DQN_episode, DQN_reward, 80)
# rule_muted_unsmoothed_coordinate = mute_y(rule_episode, rule_reward, 80)
# muted_unsmoothed_DQN_episode = [x[0] for x in DQN_muted_unsmoothed_coordinate]
# muted_unsmoothed_DQN_reward = [x[1] for x in DQN_muted_unsmoothed_coordinate]
# muted_unsmoothed_rule_episode = [x[0] for x in rule_muted_unsmoothed_coordinate]
# muted_unsmoothed_rule_reward = [x[1] for x in rule_muted_unsmoothed_coordinate]

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.figure()
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Average Reward", fontsize=20)
plt.title("Flappy bird", fontsize=20)
dqn, = plt.plot(DQN_episode, smoothed_DQN_reward, color="#cc3311")
dqn_rule, = plt.plot(rule_episode, smoothed_rule_reward, color="#0077bb")

plt.grid(ls='--')
# dqn_unsmoothed, = plt.plot(muted_unsmoothed_DQN_episode, muted_unsmoothed_DQN_reward, color="#cc3311", alpha=0.3)
# dqn_rule_unsmoothed, = plt.plot(muted_unsmoothed_rule_episode, muted_unsmoothed_rule_reward, color="#0077bb", alpha=0.3)
smoothed_DQN_reward_std = np.std(smoothed_DQN_reward)
smoothed_rule_reward_std = np.std(smoothed_rule_reward)
plt.fill_between(DQN_episode, smoothed_DQN_reward - smoothed_DQN_reward_std,
					smoothed_DQN_reward + smoothed_DQN_reward_std, facecolor='#cc3311', alpha=0.25)
plt.fill_between(rule_episode, smoothed_rule_reward - smoothed_rule_reward_std,
					smoothed_rule_reward + smoothed_rule_reward_std, facecolor='#0077bb', alpha=0.25)
plt.legend(handles=[dqn, dqn_rule], labels=['Baseline', 'SML'],  loc='upper left')
plt.axvline(x=26700, color='black', linestyle="--")
# plt.axvline(x=29000, color='black', linestyle="--")
# plt.axhline(y=80, color='black', linestyle="--")
plt.axhline(y=44, color='black', linestyle="--")
# plt.axhline(y=3, color='black', linestyle="--")
# plt.figure()
# plt.title("Average Q on Flappy Bird", fontsize=20)
# plt.xlabel("Training Epochs", fontsize=20)
# plt.ylabel("Average Q Value", fontsize=20)
# dqn_q, = plt.plot(DQN_timesteps, smoothed_DQN_Max_Q, color="#cc3311")
# rule_q, = plt.plot(rule_timesteps, smoothed_rule_Max_Q, color="#0077bb")
# plt.legend(handles=[dqn_q, rule_q], labels=['DQN', 'RIL'],  loc='lower right')

# DQN_muted_coordinate = mute(DQN_episode, smoothed_DQN_reward, 20000)
# rule_muted_coordinate = mute(rule_episode, smoothed_rule_reward, 20000)
# muted_DQN_episode = [x[0] for x in DQN_muted_coordinate]
# muted_DQN_reward = [x[1] for x in DQN_muted_coordinate]
# muted_rule_episode = [x[0] for x in rule_muted_coordinate]
# muted_rule_reward = [x[1] for x in rule_muted_coordinate]
#
# DQN_integrate = trapz(muted_DQN_reward, muted_DQN_episode)
# rule_integrate = trapz(muted_rule_reward, muted_rule_episode)
# print(DQN_integrate)
# print(rule_integrate)

plt.show()
