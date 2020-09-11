from matplotlib import pyplot as plt
import numpy as np
from numpy import trapz


def load_data(path, is_smooth=False):
    if is_smooth:
        return smooth(np.nan_to_num(np.load(path)), 0.90)
    else:
        return np.load(path)


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


if __name__ == '__main__':
    PLOT_REWARDS = True
    PLOT_Q_VALUES = False
    SAVE_PLOT = False
    DQN_DATA_DIR = "./pong/DQN"
    DRIL_DATA_DIR = "./pong/DRIL"
    WEIGHT = 0.93

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300

    if PLOT_REWARDS:
        # DQN
        dqn_episode_data = load_data(DQN_DATA_DIR + "/episode_pong_dqn.npy")
        dqn_reward_data_mata = load_data(DQN_DATA_DIR + "/reward_pong_dqn.npy", True)

        # DRIL
        dril_episode_data = load_data(DRIL_DATA_DIR + "/episode_pong_dril.npy")
        dril_reward_data_mata = load_data(DRIL_DATA_DIR + "/reward_pong_dril.npy", True)

        # smooth
        dqn_reward_data = smooth(dqn_reward_data_mata, WEIGHT)
        dril_reward_data = smooth(dril_reward_data_mata, WEIGHT)

        smoothed_DQN_reward_std = np.std(dqn_reward_data)
        smoothed_rule_reward_std = np.std(dril_reward_data)

        # DQN_muted_unsmoothed_coordinate = mute_y(dqn_episode_data, dqn_reward_data_mata, 21)
        # rule_muted_unsmoothed_coordinate = mute_y(dril_episode_data, dril_reward_data_mata, 21)
        # muted_unsmoothed_DQN_episode = [x[0] for x in DQN_muted_unsmoothed_coordinate]
        # muted_unsmoothed_DQN_reward = [x[1] for x in DQN_muted_unsmoothed_coordinate]
        # muted_unsmoothed_rule_episode = [x[0] for x in rule_muted_unsmoothed_coordinate]
        # muted_unsmoothed_rule_reward = [x[1] for x in rule_muted_unsmoothed_coordinate]

        # dqn_reward_data = [x + 22 for x in dqn_reward_data]
        # dril_reward_data = [x + 22 for x in dril_reward_data]

        fig_r, ax_r = plt.subplots()
        plt.xlabel("Episode", fontsize=20)
        plt.ylabel("Average Reward", fontsize=20)
        plt.title("Pong", fontsize=20)
        dqn, = plt.plot(dqn_episode_data, dqn_reward_data, color="#cc3311")
        dqn_rule, = plt.plot(dril_episode_data, dril_reward_data, color="#0077bb")

        # dqn_unsmoothed, = plt.plot(muted_unsmoothed_DQN_episode, muted_unsmoothed_DQN_reward, color="#cc3311",
        #                            alpha=0.3)
        # dqn_rule_unsmoothed, = plt.plot(muted_unsmoothed_rule_episode, muted_unsmoothed_rule_reward, color="#0077bb",
        #                                 alpha=0.3)
        # plt.ylim((0, 70))
        # plt.xlim((0, 31500))
        plt.grid(ls='--')
        plt.fill_between(dqn_episode_data, dqn_reward_data - smoothed_DQN_reward_std,
                         dqn_reward_data + smoothed_DQN_reward_std, facecolor='#cc3311', alpha=0.25)

        plt.fill_between(dril_episode_data, dril_reward_data - smoothed_rule_reward_std,
                         dril_reward_data + smoothed_rule_reward_std, facecolor='#0077bb', alpha=0.25)

        plt.legend(handles=[dqn, dqn_rule], labels=['Baseline', 'SML'], loc='lower right')
        # plt.axvline(x=1250, color='black', linestyle="--")
        plt.axvline(x=830, color='black', linestyle="--")
        # plt.axvline(x=2000, color='black', linestyle="--")
        plt.axhline(y=18, color='black', linestyle="--")
        # plt.axhline(y=11, color='black', linestyle="--")
        # plt.axhline(y=19.2, color='black', linestyle="--")
        if SAVE_PLOT:
            # plt.savefig("./Pong_Avg_Rewards.png", transparent=True)
            plt.savefig("./Pong_Avg_Rewards.png")
        DQN_muted_coordinate = mute(dqn_episode_data, dqn_reward_data, 1200)
        rule_muted_coordinate = mute(dril_episode_data, dril_reward_data, 1200)
        muted_DQN_episode = [x[0] for x in DQN_muted_coordinate]
        muted_DQN_reward = [x[1] for x in DQN_muted_coordinate]
        muted_rule_episode = [x[0] for x in rule_muted_coordinate]
        muted_rule_reward = [x[1] for x in rule_muted_coordinate]

        DQN_integrate = trapz(muted_DQN_reward, muted_DQN_episode)
        rule_integrate = trapz(muted_rule_reward, muted_rule_episode)
        print(DQN_integrate)
        print(rule_integrate)

        plt.show()

    if PLOT_Q_VALUES:
        # DQN
        dqn_episode_data = load_data(DQN_DATA_DIR + "/episode_pong_dqn.npy")
        dqn_q_value_data = load_data(DQN_DATA_DIR + "/q_value_pong_dqn.npy", True)

        # DRIL
        dril_episode_data = load_data(DRIL_DATA_DIR + "/episode_pong_dril.npy")
        dril_q_value_data = load_data(DRIL_DATA_DIR + "/q_value_pong_dril.npy", True)

        fig_r, ax_r = plt.subplots()
        plt.xlabel("Training Epochs", fontsize=18)
        plt.ylabel("Average Q Value", fontsize=18)
        plt.title("Average Q on Pong", fontsize=20)
        dqn, = plt.plot(dqn_episode_data, dqn_q_value_data, color="#cc3311")
        dqn_rule, = plt.plot(dril_episode_data, dril_q_value_data, color="#0077bb")
        plt.legend(handles=[dqn, dqn_rule], labels=['DQN', 'RIL'], loc='lower right')
        if SAVE_PLOT:
            # plt.savefig("./Pong_Avg_Q_Values.png", transparent=True)
            plt.savefig("./Pong_Avg_Q_Values.png")
        plt.show()
