__author__ = 'kimangkhun'

if __name__=="__main__":
    """
    Note that I've already trained the agent
    If you want to retrain it, just comment line 22, and uncomment the line 52 to save your agent
    If you want to continue to train the agent and save the new agent, just uncomment the line 52 to save your
    retrained agent
    """
    from environment.game import Environment
    from agent.Agent import Actor_Critic
    from collections import deque
    import matplotlib.pyplot as plt
    import numpy as np

    env = Environment()
    env.seed(1)                     # env.seed(3) to get the same result in video
    s = env.reset()
    n_features = len(s)
    n_action_shape = env.amt_lines
    ag = Actor_Critic(n_features, n_action_shape)
    ag.load_weights("Data")         # I already trained a2c for 1000 episodes
    mobile_returns = deque(maxlen=200)
    mobile_timesteps = deque(maxlen=200)
    avg_returns = []
    avg_timesteps = []
    EPISODES = 1000
    for i_ep in range(EPISODES):
        total_rewards = 0.0
        for h in range(200):
            env.render()
            action = ag.choose_action(s)
            sprime, reward, done, info = env.step(action)
            ag.learn_step(s, action, reward, sprime, done)     # actor critic - online learning
            total_rewards += reward
            if done:
                print("Epi: \t{}, Ts: \t{}, total rewards: \t{}".format(i_ep, h, total_rewards))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()
                break
            elif h == 199:
                print("Epi: \t{}, Ts: \t{}, total rewards: \t{}".format(i_ep, h, total_rewards))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()

    #ag.save_weights("Data")                                    # save weights after training

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_returns)
    plt.ylabel("average return")
    plt.xlabel("number of episodes")
    plt.title("Average return over episodes")
    plt.xlim(0, EPISODES)
    plt.ylim(-250, 250)

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_timesteps)
    plt.ylabel("number of timesteps")
    plt.xlabel("number of episodes")
    plt.title("Average timesteps over episodes")
    plt.xlim(0, EPISODES)
    plt.ylim(-250, 250)
    plt.show()
    np.savetxt("Data/average_returns_a2c.txt", avg_returns, fmt='%.3f')
    np.savetxt("Data/average_timesteps_a2c.txt", avg_timesteps, fmt='%.3f')
