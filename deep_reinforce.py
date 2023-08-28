######FIrst part of the code

# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
N = 100  # maximum population size
a = .5 / N  # birth rate
b = .5 / N  # death rate

lifetime = 1000
x = np.zeros(lifetime)
x[0] = 25

# %%
for t in range(lifetime - 1):
    if 0 < x[t] < N - 1:
        # Is there a birth?
        birth = np.random.rand() <= a * x[t]*(x[t]<N-1)
        # Is there a death?
        death = np.random.rand() <= b * x[t]
        # We update the population size.
        x[t + 1] = x[t] + 1 * birth - 1 * death
    # The evolution stops if we reach $0$ or $N$.
    else:
        x[t + 1] = x[t]

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(x, lw=2)

# %%
ntrials = 100
x = np.random.randint(size=ntrials,
                      low=0, high=N)

def simulate(x, lifetime):
    """Run the simulation."""
    for _ in range(lifetime - 1):
        # Which trials to update?
        upd = (0 < x) & (x < N - 1)
        # In which trials do births occur?
        birth = 1 * (np.random.rand(ntrials) <= a * x * (x<N-1))
        # In which trials do deaths occur?
        death = 1 * (np.random.rand(ntrials) <= b * x)
        # We update the population size for all trials
        x[upd] += birth[upd] - death[upd]

bins = np.linspace(0, N, 25)

lifetime_list = [10, 1000, 10000]
fig, axes = plt.subplots(1, len(lifetime_list),
                         figsize=(12, 3),
                         sharey=True)
for i, lifetime in enumerate(lifetime_list):
    ax = axes[i]
    simulate(x, lifetime)
    ax.hist(x, bins=bins)
    ax.set_xlabel("Population size")
    if i == 0:
        ax.set_ylabel("Histogram")
    ax.set_title(f"{lifetime} time steps")

# %%
initial = 25
lifetime = 1000
jumptimes = np.zeros(1)
y = np.ones(1)*initial
t = 0 
while t < lifetime:
    s = np.random.exponential(1,1)
    jumptimes = np.append(jumptimes,np.array([s+t]))
    t = t + s
    if 0 < y[-1] < N - 1:
        # Is there a birth?
        birth = np.random.rand() <= a * y[-1]
        # Is there a death?
        death = np.random.rand() <= b * y[-1]
        # We update the population size.
        helper = y[-1] + 1 * birth - 1 * death
    # The evolution stops if we reach $0$ or $N$.
    else:
        helper = y[-1]
    y = np.append(y,np.array([helper]))

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(jumptimes,y, lw=2)

# %%
#define stochastic matrix
A = np.array([[-5, 4, 1], [4, -5, 1], [2, 2, -4]])

# %%
#define jump measures
from scipy import stats

jumpmeasures = []
for i in range(3):
    values = [j for j in range(3)]
    xk = np.array(values)
    pk = np.array([-A[i,j]/A[i,i]*(j!=i) for j in values])
    custm = stats.rv_discrete(values=(xk, pk))
    jumpmeasures = jumpmeasures + [custm]

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.plot(xk, jumpmeasures[0].pmf(xk), 'ro', ms=12, mec='r')
plt.show()

# %%
lifetime = 5
initial = 0
jumptimes = np.zeros(1)
y = np.ones(1)*initial
t = 0 
while t < lifetime:
    tau = np.random.exponential(-1/A[int(y[-1]),int(y[-1])],1)
    jumptimes = np.append(jumptimes,np.array([tau+t]))
    t = t + tau
    helper = jumpmeasures[int(y[-1])].rvs(size=1)
    y = np.append(y,np.array([helper]))

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(jumptimes,y,'.')

# %%
lifetime = 20
initial = 0
jumptimes = np.zeros(1)
y = np.ones(1)*initial
t = 0 
def control(t,y):
    if t < 10:
        return 1*(2*y+1)
    else:
        return 1*(2*y+1)

while t < lifetime:
    A = control(t,y[-1])*np.array([[-5, 4, 1], [4, -5, 1], [2, 2, -4]])
    tau = np.random.exponential(-1/A[int(y[-1]),int(y[-1])],1)
    jumptimes = np.append(jumptimes,np.array([tau+t]))
    t = t + tau
    jumpmeasures = []
    for i in range(3):
        values = [j for j in range(3)]
        xk = np.array(values)
        pk = np.array([-A[i,j]/A[i,i]*(j!=i) for j in values])
        custm = stats.rv_discrete(values=(xk, pk))
        jumpmeasures = jumpmeasures + [custm]
    helper = jumpmeasures[int(y[-1])].rvs(size=1)
    y = np.append(y,np.array([helper]))

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(jumptimes,y,'.')




######Second part of the code


import numpy as np
import gym
from gym import wrappers


def run_episode(env, policy, gamma = 1.0, render = True):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [run_episode(env, policy, gamma = gamma, render = False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v

# %%
env_name  = 'FrozenLake8x8-v0'
gamma = 1.0
env = gym.make(env_name)
env=env.unwrapped
optimal_v = value_iteration(env, gamma);
policy = extract_policy(optimal_v, gamma)
policy_score = evaluate_policy(env, policy, gamma, n=1000)
print('Policy average score = ', policy_score)

# %%
"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
from gym import wrappers


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

# %%
env_name  = 'FrozenLake8x8-v0'
env = gym.make(env_name)
env = env.unwrapped
optimal_policy = policy_iteration(env, gamma = 1.0)
scores = evaluate_policy(env, optimal_policy, gamma = 1.0)
print('Average scores = ', np.mean(scores))

# %%
"""
Q-Learning example using OpenAI gym MountainCar enviornment
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np

import gym
from gym import wrappers

n_states = 40
iter_max = 10000

initial_lr = 1.0 # Learning rate
min_lr = 0.003
gamma = 1.0
t_max = 10000
eps = 0.02

def run_episode(env, policy, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b

# %%
env_name = 'MountainCar-v0'
env = gym.make(env_name)
env.seed(0)
np.random.seed(0)
print ('----- using Q Learning -----')
q_table = np.zeros((n_states, n_states, 3))
for i in range(iter_max):
    obs = env.reset()
    total_reward = 0
        ## eta: learning rate is decreased at each step
    eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
    for j in range(t_max):
        a, b = obs_to_state(env, obs)
        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(env.action_space.n)
        else:
            logits = q_table[a][b]
            logits_exp = np.exp(logits)
            probs = logits_exp / np.sum(logits_exp)
            action = np.random.choice(env.action_space.n, p=probs)
        obs, reward, done, _ = env.step(action)
        total_reward += (gamma ** j) * reward
            # update q table
        a_, b_ = obs_to_state(env, obs)
        q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
        if done:
            break
    if i % 100 == 0:
        print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
solution_policy = np.argmax(q_table, axis=2)
solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it

#run_episode(env, solution_policy, True)

# %% [markdown]
# # Deep Reinforcement Learning

# %% [markdown]
# So far we were fully in the field of optimal control without any appearance of deep learning techniques. It is particularly interesting to think of exploring an unknown environment, learning a Q function increasingly well but storing the information in a deep neural networks. In terms of the HJB equation this amounts to solving the equation by a deep neural network.

# %% [markdown]
# There are basically two approaches: learning the $Q$ function and learning the policy $ \pi $ (often in a relaxed version). One can see this from the point of view of the HJB equation, which we take in the simplest case (one player, $c=0$):
# 1. (Value iteration) Approximate solutions of the HJB equation by neural networks. i.e. choose a value function as neural network and run one step of value iteration.
# 2. (Policy iteration) Approximate policies by neural networks.

# %% [markdown]
# Previous algorithms were just implementations of solving fixed point problems by value or policy iteration, this can also be done by learning technology yielding surprising and not yet understood effects. It is not clear why this works so well and, in contrast to some classical learning tasks, there is little regularity involved.
# 
# However, also very directed approaches are efficient, see for instance: in the sequel the game Cartpole is shown from several angles and a very direct approach to learn an efficient strategy is shown, we follow here the great blog entry by [Greg Surma](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288).

# %%
import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

# %%
env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500
score_requirement = 60
intial_games = 10000

# %%
def play_a_random_game_first():
    for step_index in range(goal_steps):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("Step {}:".format(step_index))
        print("action: {}".format(action))
        print("observation: {}".format(observation))
        print("reward: {}".format(reward))
        print("done: {}".format(done))
        print("info: {}".format(info))
        if done:
            break
    env.reset()

# %%
play_a_random_game_first()

# %% [markdown]
# You can read at the [Cartpole documentation](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) what the numbers do precisely mean. Now we create a set of random strategies which were up to some extend successful. Notice that you have to install from some [packages](https://github.com/gsurma/cartpole).

# %%
def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(intial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)
            
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = observation
            score += reward
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])
        
        env.reset()

    print(accepted_scores)
    print(len(accepted_scores))
    
    return training_data

# %%
training_data = model_data_preparation()


# %%
def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    
    model.fit(X, y, epochs=10)
    return model

# %%
trained_model = train_model(training_data)

# %%
scores = []
choices = []
for each_game in range(100):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        # Uncomment below line if you want to see how our bot is playing the game.
        env.render()
        #print('Step:', step_index)
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
        if done:
            break
    #print('Game:', each_game)
    env.reset()
    scores.append(score)

print(scores)
print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))

# %%
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()

# %%
#cartpole()

# %% [markdown]
# How could a Q learning algorithm look like for portfolio optimization:
# 1. first we have to look at a time dependent version of Q learning, i.e. time gets a state: the value function is defined as
# $$
# V(t,x) : = \sup_\pi E_{t,x} \big [ R(X_T^\pi) \big ] \, ,
# $$
# which satisfies the HJB equation
# $$
# V(t,x) = R(x) + \int_t^T \sup_{u} A^u V(s,x) ds
# $$
# whence the Q function
# $$
# Q(t,x,u) = P^u_{\Delta} V(t+\Delta,.)(x) \, , \; \sup_u Q(t,x,u) = V(t,x) \, , Q(t,x,u) = R(x)
# $$
# along a grid of mesh $ \Delta $. This is just backwards induction if one runs it backwards in time.
# 2. when starting with an arbitrary Q function $Q^{(0)}$ depending on time $ t $, state $ x $ and action $ u $ (and with $ Q^{(0)}(T,x,u) = R(x) $), then
# $$
# Q^{(n+1)}(t,x,u) = (1-\alpha) Q^{(n)}(t,x,u) + \alpha \sup_u (Q^{(n)}(t+\Delta,x',u)
# $$
# where $ x' $ is sampled from $ P_\Delta^u $ with respect to an action which optimizes $ Q^{(n)}(t,x,u) $ at $ (t,x) $.
# 3. Policy iteration instead starts with a policy $ \pi^{(n)} $ depending on the states $ (t,x) $. We calculate
# $$
# V^{\pi^{(n)}}(t+\Delta,x) : = E_{t+\Delta,x} \big [ R(X_T^{\pi^{(n)}}) \big ]
# $$
# and define
# $$
# \pi^{(n+1)}(t,x) := \operatorname{argmax}P_\Delta^u V^{\pi^{(n)}}(t+\Delta,.)(x) \, ,
# $$
# which possibly improves the value function at $ (t,x) $.

# %%





