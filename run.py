import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.misc import imresize
import gym
import gym_chrome_dino


class Conv2D:

    def __init__(self, mi, mo, filtersz=5, stride=2, f=tf.nn.relu):
        self.W = tf.Variable(tf.random_normal(shape=(filtersz, filtersz, mi, mo)))
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.f = f
        self.stride = stride
        self.params = [self.W, self.b]

    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        return self.f(conv_out)


class HiddenLayer:
    
    def __init__(self, Mi, Mo, f=tf.nn.relu):
        self.W = tf.Variable(tf.random_normal((Mi, Mo)))
        self.b = tf.Variable(np.zeros((1, Mo)).astype(np.float32))
        self.f = f
        self.params = [self.W, self.b]
        
    def forward(self, X):
        output = self.f(tf.matmul(X, self.W) + self.b)
        return output
        
        
class DQN:

    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, max_exp=50000, min_exp=5000, batch_sz=32):
        self.batch_sz = batch_sz
        self.max_exp = max_exp
        self.min_exp = min_exp
        self.replay_buffer = {'s':[], 'a':[], 'r':[], 's_p':[], 'done':[]}
        self.K = K
        self.conv_layers = []
        num_input_filters = 4
        final_height = 100
        final_width = 100
        for num_output_filters, filtersz, stride in conv_layer_sizes:
            layer = Conv2D(num_input_filters, num_output_filters, filtersz, stride)
            self.conv_layers.append(layer)
            num_input_filters = num_output_filters
            old_height = final_height
            new_height = int(np.ceil(old_height / stride))
            final_height = int(np.ceil(final_height / stride))
            final_width = int(np.ceil(final_width / stride))
        self.layers = []
        flattened_ouput_size = final_height * final_width * num_input_filters
        M1 = flattened_ouput_size
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(layer)
        self.params = []
        for layer in (self.conv_layers + self.layers):
            self.params += layer.params
        self.X = tf.placeholder(tf.float32, shape=(None, 4, 100, 100), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.A = tf.placeholder(tf.int32, shape=(None,), name='actions')
        Z = self.X / 255.0
        Z = tf.transpose(Z, [0, 2, 3, 1])
        for layer in self.conv_layers:
            Z = layer.forward(Z)
        Z = tf.reshape(Z, [-1, flattened_ouput_size])
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat
        indices = tf.range(self.batch_sz) * tf.shape(self.predict_op)[1] + self.A
        selected_action_values = tf.gather(tf.reshape(self.predict_op, [-1]), indices)
        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.RMSPropOptimizer(0.001, 0.99, 0.0, 1e-6).minimize(cost)
        
    def train(self, target_network, gamma):
        if len(self.replay_buffer['s']) < self.min_exp:
            return
        idx = np.random.choice(len(self.replay_buffer['s']), self.batch_sz, False)
        states = [self.replay_buffer['s'][i] for i in idx]
        actions = [self.replay_buffer['a'][i] for i in idx]
        next_states = [self.replay_buffer['s_p'][i] for i in idx]
        rewards = [self.replay_buffer['r'][i] for i in idx]
        dones = [self.replay_buffer['done'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = []
        for next_q, r, done in zip(next_Q, rewards, dones):
            if done:
                targets.append(r)
            else:
                Qval = r + next_q*gamma
                targets.append(Qval)
        self.session.run(self.train_op, {self.X:states, self.G:targets, self.A:actions})
        
    def predict(self, states):
        output = self.session.run(self.predict_op, {self.X:states})
        return output
    
    def sample_action(self, state, eps):
        state = np.expand_dims(state, 0)
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            action = np.argmax(self.predict(state)[0])
            return action
    
    def set_session(self, session):
        self.session = session
    
    def add_experience(self, s, a, r, s_p, done):
        if len(self.replay_buffer['s']) >= self.max_exp:
            self.replay_buffer['s'].pop(0)
            self.replay_buffer['a'].pop(0)
            self.replay_buffer['r'].pop(0)
            self.replay_buffer['s_p'].pop(0)
            self.replay_buffer['done'].pop(0)
        self.replay_buffer['s'].append(s)
        self.replay_buffer['a'].append(a)
        self.replay_buffer['s_p'].append(s_p)
        self.replay_buffer['r'].append(r)
        self.replay_buffer['done'].append(done)
    
    def copy_from(self, eval_model):
        ops = []
        for i in range(len(self.params)):
            val = self.session.run(eval_model.params[i])
            op = self.params[i].assign(val)
            ops.append(op)
        self.session.run(ops)

def update_state(seq_state, state):
    if len(seq_state) >= 4:
        seq_state.pop(0)
    state = downsample_image(state)
    seq_state.append(state)
    return seq_state
        
def play_one(env, enet, tnet, gamma, eps, copy_period):
    s = env.reset()
    seq_state = []
    for i in range(4):
        seq_state = update_state(seq_state, s)
    total_reward = 0
    done = False
    iters = 0
    while not done:
        if (iters % copy_period) == 0:
            tnet.copy_from(enet)
        a = enet.sample_action(seq_state, eps)
        p_seq_state = seq_state
        s, r, done, _ = env.step(a)
        total_reward += r
        seq_state = update_state(seq_state, s)
        enet.add_experience(p_seq_state, a, r, seq_state, done)
        enet.train(tnet, gamma)
        iters += 1
    return total_reward


def downsample_image(A):
    B = A[22:]
    B = B.mean(axis=2)
    B = imresize(B, size=(100, 100), interp='nearest')
    return B

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

env = gym.make('ChromeDinoNoBrowser-v0')
n_actions = 3 
sess = tf.InteractiveSession()
conv_size = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
hidd_size = [512]
enet = DQN(n_actions, conv_size, hidd_size)
tnet = DQN(n_actions, conv_size, hidd_size)
sess.run(tf.global_variables_initializer())
enet.set_session(sess)
tnet.set_session(sess)
saver = tf.train.Saver()
saver.restore(sess, './dino.ckpt')
# print("Populating experience replay buffer...")
# s = env.reset()
# seq_state = []
# for i in range(4):
#     seq_state = update_state(seq_state, s)
# for i in range(5000):
#     action = np.random.choice(n_actions)
#     obs, reward, done, _ = env.step(action)
#     next_seq_state = update_state(seq_state, obs)
#     enet.add_experience(seq_state, action, reward, next_seq_state, done)
#     if done:
#         obs = env.reset()
#         obs_small = downsample_image(obs)
#         for i in range(4):
#             seq_state = update_state(seq_state, s)
#     else:
#         seq_state = next_seq_state
# print("Replay buffer population procedure complete.")
print("Initializing training procedure now.")
N = 100
totalrewards = np.empty(N)
costs = np.empty(N)
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    if eps < 0.1:
        eps = 0.1
    print("Playing another episode {}...".format(n))
    totalreward = play_one(env, enet, tnet, 0.99, eps, 100)
    print("Episode finished.")
    totalrewards[n] = totalreward
    print("Attained reward this episode is: ", totalreward)
    if n % 10 == 0:
        print("Episode: ", n, "Epsilon: ", eps, "Avg.Reward (Last 100): ", totalrewards[max(0, n-100):(n+1)].mean())
    saver.save(sess, './dino.ckpt')
print("Avg. Reward for the last 100 Episodes: ", totalrewards[-100:].mean())
print("Total Steps: ", totalrewards.sum())
plt.plot(totalrewards)
plt.title("Rewards")
plt.show()
plot_running_avg(totalrewards)