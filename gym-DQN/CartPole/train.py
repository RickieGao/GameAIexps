import gym, os
from os import path
import DQAgent

env = gym.make('CartPole-v0')

root_path = './results/'

agent_opts = {
    # hyperparameters
    'REPLAY_BATCH_SIZE': 32,
    'LEARNING_BATCH_SIZE': 1,
    'DISCOUNT': .9,
    'MAX_STEPS': 500,
    'REPLAY_MEMORY_SIZE': 500,
    'LEARNING_RATE': 0.01,

    # ann specific
    'EPSILON_START': .9,
    'EPSILON_DECAY': .5,
    'MIN_EPSILON': 0.1,

    # saving and logging results
    'AGGREGATE_STATS_EVERY': 1,
    'SHOW_EVERY': 1,
    'COLLECT_RESULTS': True,
    'SAVE_EVERY_EPOCH': True,
    'SAVE_EVERY_STEP': False,
    'BEST_MODEL_FILE': f'{root_path}best_model.h5',
}

model_opts = {
    'num_layers': 1,
    'default_nodes': 20,
    'dropout_rate': 0.5,
    'model_type': 'ann',
    'add_dropout': False,
    'add_callbacks': False,
    'nodes_per_layer': [24],

    # cnn options
    'filter_size': 3,
    'pool_size': 2,
    'stride_size': None,
}


# Train models
def train_model(agent_opts, model_opts):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    agent.train(n_epochs=24, render=False)
    agent.save_weights(f'{root_path}cartpole')
    agent.show_plots('cumulative')
    agent.show_plots('loss')
    env.close()


# Evaluate model
def evaluate_model(agent_opts, model_opts, best_model=True):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    if best_model:
        filename = agent_opts.get('BEST_MODEL_FILE')[:-3]
        agent.load_weights(filename)
    else:
        agent.load_weights('./agent/results/cnnagent')
    results = agent.evaluate(100, render=False, verbose=False)
    print(f'Average Reward: {sum(sum(results, [])) / len(results)}')
    results = agent.evaluate(10, render=True, verbose=True)


train_model(agent_opts, model_opts)
evaluate_model(agent_opts, model_opts, best_model=True)