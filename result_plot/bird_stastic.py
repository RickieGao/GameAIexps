import csv
from decimal import Decimal


def read_file(file_name):
	with open(file_name, "r") as file:
		reader = csv.reader(file)
		next(reader)
		recorder1, recorder2 = [], []
		for row in reader:
			recorder1.append(int(row[1]))
			recorder2.append(float(Decimal(row[2]).quantize(Decimal('0.00'))))
	return recorder1, recorder2


DQN_timesteps, DQN_Max_Q = read_file(r"bird\DQN-tag-50kper_average_qMax_50kper_average_qMax.csv")
rule_timesteps, rule_Max_Q = read_file(r"bird\run-DQN_rule-tag-50kper_average_qMax_50kper_average_qMax.csv")
DQN_episode, DQN_reward = read_file(r"bird\run-DQN-tag-reward_per_life_reward_per_life.csv")
rule_episode, rule_reward = read_file(r"bird\run-DQN_rule-tag-reward_per_life_reward_per_life.csv")

reve_timesteps, reve_Max_Q = read_file(r"bird\run-DQN_rule_reve_count_graph-tag-50kper_average_qMax_50kper_average_qMax.csv")
reve_episode, reve_reward = read_file(r"bird\run-DQN_rule_reve_count_graph-tag-reward_per_life_reward_per_life.csv")

DQN_Q = dir(zip(DQN_timesteps, DQN_Max_Q))
DQN_r = dir(zip(DQN_episode, DQN_reward))
rule_Q = dir(zip(rule_timesteps, rule_Max_Q))
rule_r = dir(zip(rule_episode, rule_reward))
reve_Q = dir(zip(reve_timesteps, reve_Max_Q))
reve_r = dir(zip(reve_episode, reve_reward))
