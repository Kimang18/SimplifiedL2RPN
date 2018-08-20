__author__ = "kimangkhun"

import numpy as np
import matplotlib.pyplot as plt

"""DQN pendant 5 heures"""

EPISODES = 3000
legends = ["DAgger", "A2C", "DDQN", "DQN", "Random", "Donothing", "Greedy_Prior", "Solver"]
avg_returns = []

avg_returns.append(np.loadtxt("Data/average_returns_dagger.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_a2c.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_ddqn.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_dqn.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_random.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_donothing.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_greedy_prior.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_greedy.txt"))

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(avg_returns)):
    plt.plot(range(EPISODES), avg_returns[i])
plt.ylabel("Moving average total rewards")
plt.xlabel("Number of episodes")
plt.title("Moving average total rewards over 3000 episodes, gamma = 1.0")
plt.legend(legends, loc='best')

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(0, len(avg_returns)-1):
    plt.plot(range(EPISODES), avg_returns[i])
plt.ylabel("Moving average total rewards")
plt.xlabel("Number of episodes")
plt.title("Moving average total rewards over 3000 episodes, gamma = 1.0")
plt.legend(["DAgger", "A2C", "DDQN", "DQN", "Random", "Donothing", "Greedy_Prior"], loc='best')

avg_timesteps = []

avg_timesteps.append(np.loadtxt("Data/average_timesteps_dagger.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_a2c.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_ddqn.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_dqn.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_random.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_donothing.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_greedy_prior.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_greedy.txt"))


fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(avg_returns)):
    plt.plot(range(EPISODES), avg_timesteps[i])
plt.ylabel("Moving average timesteps")
plt.xlabel("Number of episodes")
plt.title("Moving average timesteps over 3000 episodes, gamma = 1.0")
plt.legend(legends, loc='best')
plt.show()