__author__ = "kimangkhun"

if __name__=="__main__":
    from environment.game import Environment
    from collections import deque
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    env = Environment()
    env.seed(1)  # env.seed(3) to get the same result in video
    s = env.reset()
    mobile_returns = deque(maxlen=200)
    mobile_timesteps = deque(maxlen=200)
    avg_returns = []
    avg_timesteps = []
    EPISODES = 3000
    for i_ep in tqdm(range(EPISODES)):
        total_rewards = 0.0
        for h in range(200):
            #env.render()
            action = env.action_sample()
            sprime, reward, done, info = env.step(action)
            total_rewards += reward
            if done:
                #print("Epi: \t{}, Ts: \t{}, total rewards: \t{}".format(i_ep, h, total_rewards))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()
                break
            elif h == 199:
                #print("Epi: \t{}, Ts: \t{}, total rewards: \t{}".format(i_ep, h, total_rewards))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()

    plt.ioff()

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_returns)
    plt.ylabel("average return")
    plt.xlabel("number of episodes")
    plt.title("Average return over episodes")
    plt.xlim(0, EPISODES)
    #plt.ylim(-250, 250)

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_timesteps)
    plt.ylabel("number of timesteps")
    plt.xlabel("number of episodes")
    plt.title("Average timesteps over episodes")
    plt.xlim(0, EPISODES)
    plt.ylim(-250, 250)
    plt.show()
    np.savetxt("Data/average_returns_random.txt", avg_returns, fmt='%.3f')
    np.savetxt("Data/average_timesteps_random.txt", avg_timesteps, fmt='%.3f')