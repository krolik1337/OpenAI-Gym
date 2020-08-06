#%%
import gym #import biblioteki gym
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

class Agent():
    def __init__(self, env):
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete

        if self.is_discrete:
            self.action_size = env.action_space.n
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape

    def get_action(self, state):
        if self.is_discrete:
            action = np.random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
        return action

class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate = 0.01):
        super().__init__(env)
        self.epsilon = 1.0
        self.state_size = len(env.observation_space.low)
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()
    
    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])

    def get_action(self, state):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if np.random.random()<self.epsilon else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience
        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next)
        q_update = q_target - q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update
        if done:
            self.epsilon = self.epsilon * 0.99


def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

buckets = [3, 3, 6, 3]
env = gym.make("CartPole-v1") #stworzenie środowiska o podanej nazwie
agent = QAgent(env) #inicjalizacja agenta
state = discretize(env.reset())

#%%
print(agent.q_table)
for episode in range(1000):
    for ts in range(200):
        #env.render() #wyświetlenie wyniku
        action = agent.get_action(state) #wybranie akcji na podstawie stanu
        next_state, reward, done, info = env.step(action) #zapisanie wyników akcji
        new_state = discretize(next_state)
        agent.train((state, action, next_state, reward, done))
        state = next_state
        if done == True:
            break
    state = env.reset()
    print("Episode", episode+1, "ended with reward", ts, sep=" ")
    rewards.append(ts+1)
print("Average reward for", episode+1, "episodes was", sum(rewards)/len(rewards), sep=" ")
#%%
X = np.linspace(1, 1000, 1000)
fig, ax = plt.subplots()
ax.bar(X, rewards, color = 'C1')
fig.show()
# %%
