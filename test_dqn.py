__author__ = 'kimangkhun'


if __name__=="__main__":
    from environment.game import Environment
    from agent.Agent import DQN
    from collections import deque
    import matplotlib.pyplot as plt
    import numpy as np

    env = Environment()
    env.seed(1)
    s = env.reset()
    n_features = len(s)
    ag = DQN(n_features, env.amt_lines)
    #ag.load_weights("Data/dqn_weight")
    mobile_returns = deque(maxlen=200)
    mobile_timesteps = deque(maxlen=200)
    batch_size = 32
    avg_returns = []
    avg_timesteps = []
    EPISODES = 1000
    for i_ep in range(EPISODES):
        total_rewards = 0.0
        s = np.reshape(s, [1, ag.feature_size])
        for h in range(200):
            action = ag.choose_action(s)
            sprime, reward, done, info = env.step(action)
            if sprime is not None:
                sprime = np.reshape(sprime, [1, ag.feature_size])
            ag.remember(s, action, reward, sprime, done)
            total_rewards += reward
            if done:
                print("Epi: \t{}, Ts: \t{}, total rewards: \t{}, epsilon: \t{}".format(i_ep, h, total_rewards, ag.epsilon))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()
                break
            elif h == 199:
                print("Epi: \t{}, Ts: \t{}, total rewards: \t{}, epsilon: \t{}".format(i_ep, h, total_rewards, ag.epsilon))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()

            if len(ag.memory) > batch_size:
                ag.replay(batch_size)
        if i_ep % 10 == 0:
            if ag.epsilon > ag.epsilon_min:
                ag.epsilon *= ag.epsilon_decay


    ag.save_weights("Data/dqn_weight")

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
    np.savetxt("Data/average_returns_dqn.txt", avg_returns, fmt='%.3f')
    np.savetxt("Data/average_timesteps_dqn.txt", avg_timesteps, fmt='%.3f')