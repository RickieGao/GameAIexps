import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# reproducible
# np.random.seed(1)
# tf.set_random_seed(1)


class PGmodel:
    def __init__(
            self,
            n_actions,   #action space
            n_features,  #state space
            learning_rate=0.0001,#learning rate
            reward_decay=0.95,#reward's decay rate
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []#The observations, actions, and rewards of a trajectory

        self._build_net()#build policy net

        self.sess = tf.Session()#initlize session

        if output_graph:  #whether output tensordoard file or not
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32,shape=[None, self.n_features], name="observations")#observation
            self.tf_acts = tf.placeholder(tf.int32, shape=[None, ], name="actions_num")#actions
            self.tf_vt = tf.placeholder(tf.float32, shape=[None, 12], name="actions_value")## state-action  value (by computing reward )

        # fc1，
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,#output neurons
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2，output action's value
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,#output neurons
            activation=None,#add softmax later
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # use softmax to convert to probability，One softmax process is performed on all output behaviors, that is, the value of each action is converted to prob
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        #传统意义上来说，PG是没有误差的概念的，因为它是进行反向传递后能使下次选这个动作的概率增加一点，然后再乘以增加的幅度，将这个做反向传递
        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            #选出对应action的probability
            # 加负号是因为我们是想要使概率越来越大或概率乘以奖励越来越大，这是一个maximize的过程，但是tf只有minimize这个功能，所以加一个负号把maximize变成minimize
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)#tf是最小化误差的系统

     #直接根据输出的动作的概率分布来选择动作，直接调用numpy中的np.random.choice(range)就可以
    def choose_action(self, observation):
        """从最后一层softmax层输出的概率数组中选择动作"""
        # state = state[np.newaxis, :]  # （n_inputs,)转化成(1, n_inputs)

        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})#所有action的概率
        print('打印出概率')
        print(prob_weights)
        # 按照概率选择动作, size=None时默认返回单个值
        action = np.random.choice(range(prob_weights.shape[1]),size=2, p=prob_weights.ravel())  # select action w.r.t the actions prob根据概率来选action
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
        discounted_ep_rs = self.print_value()
        print('))))')
        print(discounted_ep_rs.shape)
        print('((((')

    def learn(self):
        """训练过程"""
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        print('********')
        print(discounted_ep_rs_norm.shape)
        print('*********')
        print(self.tf_vt.shape)
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm, # shape=[None, ]
        })

        # 重置序列数据
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data回合结束时将其置空，下一回合再继续往里面存储
        return discounted_ep_rs_norm

    '''我们之前存储的奖励是当前状态s采取动作a获得的即时奖励，
    而当前状态s采取动作a所获得的真实奖励应该是即时奖励加上未来直到episode结束的奖励贴现和。'''
    #累积回报函数v的处理
    def _discount_and_norm_rewards(self):
        """credit assignment 技巧，只考虑该动作之后的回报，并对回报进行归一化"""
        # discount episode rewards
        #折扣回报和
        discounted_ep_rs = np.zeros_like(self.ep_rs,dtype=np.float64)
        running_add = 0
        #reversed 返回的是列表的反序，这样就得到了贴现求和值
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards # 将折扣后的回报Normalization，归一化处理
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def print_value(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        return discounted_ep_rs


