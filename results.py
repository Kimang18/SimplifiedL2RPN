__author__ = "kimangkhun"

import numpy as np
import matplotlib.pyplot as plt

EPISODES = 1000
avg_returns = []
avg_returns.append(np.loadtxt("Data/avg_returns_greedy.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_a2c.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_ddqn.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_dqn.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_random.txt"))
avg_returns.append(np.loadtxt("Data/average_returns_donothing.txt"))

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(avg_returns)):
    plt.plot(range(EPISODES), avg_returns[i])
plt.ylabel("Average returns")
plt.xlabel("Number of episodes")
plt.title("Average returns over 1000 episodes, gamma = 0.95")
plt.legend(["Greedy", "A2C", "DDQN", "DQN", "Random", "Donothing"], loc='best')

avg_timesteps = []
avg_timesteps.append(np.loadtxt("Data/avg_timesteps_greedy.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_a2c.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_ddqn.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_dqn.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_random.txt"))
avg_timesteps.append(np.loadtxt("Data/average_timesteps_donothing.txt"))

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(avg_returns)):
    plt.plot(range(EPISODES), avg_timesteps[i])
plt.ylabel("Average timesteps")
plt.xlabel("Number of episodes")
plt.title("Average timesteps over 1000 episodes, gamma = 0.95")
plt.legend(["Greedy", "A2C", "DDQN", "DQN", "Random", "Donothing"], loc='best')
plt.show()