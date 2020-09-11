from decimal import Decimal
import matplotlib
import csv
from matplotlib import pyplot as plt
from numpy import trapz
import numpy as np

WEIGHT_R = 0.997
WEIGHT_Q = 0.99
WEIGHT_R2 = 0.95


def read_file(file_name):
	with open(file_name, "r") as file:
		recorder = []
		for lines in file:
			recorder.append(float(Decimal(lines).quantize(Decimal('0.00'))))
	return recorder


def read_csvfile(file_name):
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


max_q = read_file(r"cartpole/exp01dqn_4w/max_q_array.txt")
reward = read_file(r"cartpole/exp01dqn_4w/reward_array.txt")
time_line_q = read_file(r"cartpole/exp01dqn_4w/time_line_q.txt")
time_line_r = read_file(r"cartpole/exp01dqn_4w/episodes.txt")

rule_max_q = read_file(r"cartpole/exp02rule_6w/max_q_array.txt")
rule_reward = read_file(r"cartpole/exp02rule_6w/reward_array.txt")
rule_time_line_q = read_file(r"cartpole/exp02rule_6w/time_line_q.txt")
rule_time_line_r = read_file(r"cartpole/exp02rule_6w/episodes.txt")

# reve_max_q = read_file(r"cartpole/exp03reve_4w/max_q_array.txt")
# reve_reward = read_file(r"cartpole/exp03reve_4w/reward_array.txt")
# reve_time_line_q = read_file(r"cartpole/exp03reve_4w/time_line_q.txt")
# reve_time_line_r = read_file(r"cartpole/exp03reve_4w/episodes.txt")

smoothed_reward_DQN = smooth(reward, WEIGHT_R)
smoothed_reward_rule = smooth(rule_reward, WEIGHT_R)
# smoothed_reward_reve = smooth(reve_reward, WEIGHT_R)
# time_line_q = [i*1000 for i in time_line_q]

# smoothed2_reward_DQN = smooth(reward, WEIGHT_R2)
# smoothed2_reward_rule = smooth(rule_reward, WEIGHT_R2)
# DQN_muted_unsmoothed_coordinate = mute_y(time_line_r, smoothed2_reward_DQN, 20)
# rule_muted_unsmoothed_coordinate = mute_y(rule_time_line_r, smoothed2_reward_rule, 20)
# muted_unsmoothed_DQN_episode = [x[0] for x in DQN_muted_unsmoothed_coordinate]
# muted_unsmoothed_DQN_reward = [x[1] for x in DQN_muted_unsmoothed_coordinate]
# muted_unsmoothed_rule_episode = [x[0] for x in rule_muted_unsmoothed_coordinate]
# muted_unsmoothed_rule_reward = [x[1] for x in rule_muted_unsmoothed_coordinate]

smoothed_max_q = smooth(max_q, WEIGHT_Q)
smoothed_rule_max_q = smooth(rule_max_q, WEIGHT_Q)
# smoothed_reverse_Max_Q = smooth(reve_max_q, WEIGHT_Q)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.figure()
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Average Reward", fontsize=20)
plt.title("Cart pole", fontsize=20 )
dqn, = plt.plot(time_line_r, smoothed_reward_DQN, color="#cc3311")
dqn_rule, = plt.plot(rule_time_line_r, smoothed_reward_rule, color="#0077bb")

# plt.ylim((0, 70))
# plt.xlim((0, 31500))
plt.grid(ls='--')
smoothed_DQN_reward_std = np.std(smoothed_reward_DQN)
smoothed_rule_reward_std = np.std(smoothed_reward_rule)
plt.fill_between(time_line_r, smoothed_reward_DQN - smoothed_DQN_reward_std,
					smoothed_reward_DQN + smoothed_DQN_reward_std, facecolor='#cc3311', alpha=0.25)
plt.fill_between(rule_time_line_r, smoothed_reward_rule - smoothed_rule_reward_std,
					smoothed_reward_rule + smoothed_rule_reward_std, facecolor='#0077bb', alpha=0.25)
# dqn_unsmoothed, = plt.plot(muted_unsmoothed_DQN_episode, muted_unsmoothed_DQN_reward, color="#cc3311", alpha=0.3)
# dqn_rule_unsmoothed, = plt.plot(muted_unsmoothed_rule_episode, muted_unsmoothed_rule_reward, color="#0077bb", alpha=0.3)

# rule_reverse, = plt.plot(reve_time_line_r, smoothed_reward_reve, color="#082E54")
# plt.legend(handles=[dqn, dqn_rule, rule_reverse], labels=['DQN', 'RIL', 'reversed rule'],  loc='lower right')
plt.legend(handles=[dqn, dqn_rule], labels=['Baseline', 'SML'],  loc='lower right')
# plt.axvline(x=1300, color='black', linestyle="--")
plt.axvline(x=600, color='black', linestyle="--")
plt.axhline(y=11.6, color='black', linestyle="--")
# plt.axhline(y=2, color='black', linestyle="--")
# plt.axhline(y=19.2, color='black', linestyle="--")
# plt.figure()
# plt.xlabel("Training Epochs", fontsize=20)
# plt.ylabel("Average Q Value", fontsize=20)
# plt.title("Average Q on CartPole", fontsize=20)
# dqn_q, = plt.plot(time_line_q, smoothed_max_q, color="#cc3311")
# dqn_rule_q, = plt.plot(rule_time_line_q, smoothed_rule_max_q, color="#0077bb")
# rule_reverse_q, = plt.plot(reve_time_line_q, smoothed_reverse_Max_Q, color="#BDFCC9")
# plt.legend(handles=[dqn_q, dqn_rule_q, rule_reverse_q], labels=['DQN', 'RIL', 'reversed rule'],  loc='lower right')

# DQN_muted_coordinate = mute(time_line_r, smoothed_reward_DQN, 3600)
# rule_muted_coordinate = mute(rule_time_line_r, smoothed_reward_rule, 3600)
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
