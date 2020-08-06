#%%
import gym #import biblioteki gym
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1") #stworzenie środowiska o podanej nazwie
class Agent():
    def get_action(self, state):
        pole_velocity = state[3] #zapisanie kąta odchylenia słupa
        action = 0 if pole_velocity < 0 else 1 #wybranie akcji (1 lub 0)
        return action
agent = Agent() #inicjalizacja agenta
state = env.reset()
rewards = []
#%%
for episode in range(10):
    for ts in range(200):
        #env.render() #wyświetlenie wyniku
        action = agent.get_action(state) #wybranie akcji na podstawie stanu
        state, reward, done, info = env.step(action) #zapisanie wyników akcji
        if done == True:
            break
    state = env.reset()
    print("Episode", episode+1, "ended with reward", ts, sep=" ")
    rewards.append(ts+1)
print("Average reward for", episode+1, "episodes was", sum(rewards)/len(rewards), sep=" ")
#%%
X = np.linspace(1, 10, 10)
fig, ax = plt.subplots()
ax.bar(X, rewards, color = 'C1')
fig.show()

# %%
