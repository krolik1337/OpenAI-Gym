#%%
import gym
import numpy as np
import math
import matplotlib as mpl 
import matplotlib.pyplot as plt

class CartPoleQAgent():
    def __init__(self, buckets=(1, 1, 6, 3), num_episodes=1000, min_lr=0.1, min_epsilon=0.01, discount=1.0, decay=20):
        # declaring number of buckets to split continuous data, number of training episodes, minimal learning rate
        # minimal epsilon (eplore/exploit ratio), discount, rate of decay, environment and table for rewards
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.epsilon = 1.
        self.learning_rate = 1.
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        self.rewarr = []
        self.env = gym.make('CartPole-v1')

        # [position, velocity, angle, angular velocity]
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]

        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

    # discretize state to specified number of buckets
    def discretize_state(self, state):
        discretized = []
        for i in range(len(state)):
            scaling = (state[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_state = int(round((self.buckets[i] - 1) * scaling))
            new_state = min(self.buckets[i] - 1, max(0, new_state))
            discretized.append(new_state)
        return tuple(discretized)

    #choose random action with probability of epsilon
    def choose_action(self, state):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample() 
        else:
            return np.argmax(self.Q_table[state])

    #update state according to equation
    def update_q(self, state, action, reward, new_state):
        self.Q_table[state][action] += self.learning_rate * (reward + self.discount *
            np.max(self.Q_table[new_state]) - self.Q_table[state][action])

    #decrease epsilon
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    #decrease learning rate
    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    #generate q-table for specified number of episodes, save gained rewards
    def train(self):
        for e in range(self.num_episodes):
            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False
            rewsum = 0
            tick = 0
            while not done and tick<=200:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state
                rewsum = rewsum+1
                tick +=1
            self.rewarr.append(rewsum)  

    #run one episode without training using current q-table
    def run(self):
        t = 0
        done = False
        current_state = self.discretize_state(self.env.reset())
        while not done:
                self.env.render()
                t = t+1
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                current_state = new_state
        return t
            

#%%

agent = CartPoleQAgent()
t = agent.run()
print("Not trained agent survived", t, "ticks.", sep=" ")
agent.train()
agent.env.close()
#%%
X = np.linspace(1, len(agent.rewarr), len(agent.rewarr))
fig, ax = plt.subplots()
ax.scatter(X, agent.rewarr, 7, color = 'C2')
ax.axhline(y=195, color='r')
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
fig.show()
#%%
t = agent.run()
print("Agent trained for", agent.num_episodes, "episodes survived", t, "ticks.", sep=" ")
agent.env.close()
# %%
