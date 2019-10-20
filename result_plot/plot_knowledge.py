import csv
from decimal import Decimal
import matplotlib
from matplotlib import pyplot as plt

WEIGHT_R = 0.998
WEIGHT_Q = 0.6


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


DQN_timesteps, DQN_Max_Q = read_file(r"knowledge\run-Exp17Graph-tag-50kper_average_qMax_50kper_average_qMax.csv")
rule_timesteps, rule_Max_Q = read_file(r"knowledge\run-Exp18Graph-tag-50kper_average_qMax_50kper_average_qMax.csv")
DQN_episode, DQN_reward = read_file(r"knowledge\run-Exp17Graph-tag-reward_per_life_reward_per_life.csv")
rule_episode, rule_reward = read_file(r"knowledge\run-Exp18Graph-tag-reward_per_life_reward_per_life.csv")

smoothed_DQN_Max_Q = smooth(DQN_Max_Q, WEIGHT_Q)
smoothed_rule_Max_Q = smooth(rule_Max_Q, WEIGHT_Q)
smoothed_DQN_reward = smooth(DQN_reward, WEIGHT_R)
smoothed_rule_reward = smooth(rule_reward, WEIGHT_R)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.figure()
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Average Reward", fontsize=20)
plt.title("Average Reward on muted Space War", fontsize=20)
dqn, = plt.plot(DQN_episode, smoothed_DQN_reward, color="#cc3311")
dqn_rule, = plt.plot(rule_episode, smoothed_rule_reward, color="#0077bb")
plt.legend(handles=[dqn, dqn_rule], labels=['DQN', 'RIL'],  loc='lower right')
plt.axvline(x=1700, color='black', linestyle="--")
# plt.axvline(x=2050, color='black', linestyle="--")
# plt.axhline(y=380, color='black', linestyle="--")
plt.axhline(y=325, color='black', linestyle="--")
plt.figure()
plt.title("Average Q on muted Space War", fontsize=20)
plt.xlabel("Training Epochs", fontsize=20)
plt.ylabel("Average Q Value", fontsize=20)
dqn_q, = plt.plot(DQN_timesteps, smoothed_DQN_Max_Q, color="#cc3311")
rule_q, = plt.plot(rule_timesteps, smoothed_rule_Max_Q, color="#0077bb")
plt.legend(handles=[dqn_q, rule_q], labels=['DQN', 'RIL'],  loc='lower right')
plt.show()
