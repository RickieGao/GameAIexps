from decimal import Decimal
import matplotlib
import csv
from matplotlib import pyplot as plt

WEIGHT_R = 0.998
WEIGHT_Q = 0.998


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


max_q = read_file(r"breakout\max_q_array.txt")
reward = read_file(r"breakout\reward_array.txt")
time_line_q = read_file(r"breakout\time_line_q.txt")
time_line_r = read_file(r"breakout\time_line_r.txt")

rule_max_q = read_file(r"breakout\exp5_max_q_array.txt")
rule_reward = read_file(r"breakout\exp5_reward_array.txt")
rule_time_line_q = read_file(r"breakout\exp5_time_line_q.txt")
rule_time_line_r = read_file(r"breakout\exp5_time_line_r.txt")

# reverse_timesteps, reverse_Max_Q = read_csvfile(r"breakout\run-Exp11Graph-tag-50kper_average_qMax_50kper_average_qMax.csv")
# reverse_episode, reverse_reward = read_csvfile(r"breakout\run-Exp11Graph-tag-reward_per_life_reward_per_life.csv")

smoothed_reward_DQN = smooth(reward, WEIGHT_R)
smoothed_reward_rule = smooth(rule_reward, WEIGHT_R)
time_line_q = [i*1000 for i in time_line_q]

smoothed_max_q = smooth(max_q, WEIGHT_Q)
smoothed_rule_max_q = smooth(rule_max_q, WEIGHT_Q)

# smoothed_reverse_Max_Q = smooth(reverse_Max_Q, 0.9)
# smoothed_reverse_reward = smooth(reverse_reward, 0.99)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.figure()
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Average Reward", fontsize=20)
plt.title("Average Reward on Breakout", fontsize=20)
dqn, = plt.plot(time_line_r, smoothed_reward_DQN, color="#cc3311")
dqn_rule, = plt.plot(rule_time_line_r, smoothed_reward_rule, color="#0077bb")
# rule_reverse, = plt.plot(reverse_episode, smoothed_reverse_reward)
# plt.legend(handles=[dqn, dqn_rule, rule_reverse], labels=['DQN', 'RIL', 'reversed rule'],  loc='lower right')
# plt.axvline(x=20750, color='black', linestyle="--")
# plt.axvline(x=30500, color='black', linestyle="--")
# plt.axhline(y=55, color='black', linestyle="--")
# plt.axhline(y=26.5, color='black', linestyle="--")
plt.figure()
plt.xlabel("Training Epochs", fontsize=20)
plt.ylabel("Average Q Value", fontsize=20)
plt.title("Average Q on Breakout", fontsize=20)
dqn_q, = plt.plot(time_line_q, smoothed_max_q, color="#cc3311")
dqn_rule_q, = plt.plot(rule_time_line_q, smoothed_rule_max_q, color="#0077bb")
# rule_reverse_q, = plt.plot(reverse_timesteps, smoothed_reverse_Max_Q)
# plt.legend(handles=[dqn_q, dqn_rule_q, rule_reverse_q], labels=['DQN', 'RIL', 'reversed rule'],  loc='lower right')
plt.show()
