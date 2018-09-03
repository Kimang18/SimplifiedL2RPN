
if __name__=="__main__":
    from environment.game import Environment
    from agent.Agent import DAgger, Greedy
    from collections import deque
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    env = Environment()
    #env.seed(3)
    s = env.reset()
    obs_shape = s.shape
    n_action_shape = env.amt_lines
    ag = DAgger(obs_shape)
    mobile_returns = deque(maxlen=200)
    mobile_timesteps = deque(maxlen=200)
    avg_returns = []
    avg_timesteps = []
    EPISODES = 3000
    env.seed(1)
    RENDER = True
    # for i_ep in range(EPISODES):
    current_ts = 0
    i_ep = 0
    ## Collect data for DAgger
    human = Greedy()
    while len(ag.memory) < 2048:
        total_rewards = 0.0
        h = 0
        while True:
            env_action = np.zeros(17)
            action, _ = human.choose_action(s, env)
            env_action[11:] = np.copy(action)
            sprime, reward, done, info = env.step(env_action)
            ag.remember(s, action)
            total_rewards += reward
            if done:
                print("TS: \t{}, total rewards: \t{}".format(h, total_rewards))
                s = env.reset()
                break
            elif h == 199:
                print("TS: \t{}, total rewards: \t{}".format(h, total_rewards))
                s = env.reset()
                break
            s = np.copy(sprime)
            h += 1
    ag.train()
    env.seed(1)
    while i_ep < EPISODES:
        total_rewards = 0.0
        h = 0
        while True:
            """
            if RENDER:
                q_value = ag.prob_action(s)
                env.render(show=True, q_value=q_value, str_action=ag.action_space._str_actions)
            """
            env_action = np.zeros(17)
            action = ag.choose_action(s, env)
            env_action[11:] = np.copy(action)
            sprime, reward, done, info = env.step(env_action)
            total_rewards += reward
            if done:
                print("Epi: \t{}, Ts: \t{}, total rewards: \t{}, curr_ts: \t{}".format(i_ep, h, total_rewards,
                                                                                       current_ts))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()
                current_ts += h + 1
                break
            elif h == 199:
                # RENDER = False
                print("Epi: \t{}, Ts: \t{}, total rewards: \t{}, curr_ts: \t{}".format(i_ep, h, total_rewards,
                                                                                       current_ts))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()
                current_ts += h + 1
                break
            s = np.copy(sprime)
            h += 1

        i_ep += 1
        # ag.learn()
        #if i_ep % 10 == 0:
        #    ag.train()
        if h < 199:
            ag.train()
        #    ag.gamma *= 0.99
        if i_ep % 500 == 0:
            ag.save_weights("Data/dagger_weight")

    ag.save_weights("Data/dagger_weight")

    plt.ioff()

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_returns)
    plt.ylabel("average return")
    plt.xlabel("number of episodes")
    plt.title("Average return over episodes")
    plt.xlim(0, EPISODES)

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_timesteps)
    plt.ylabel("number of timesteps")
    plt.xlabel("number of episodes")
    plt.title("Average timesteps over episodes")
    plt.xlim(0, EPISODES)

    plt.show()

    np.savetxt("Data/average_returns_dagger3.txt", avg_returns, fmt='%.3f')
    np.savetxt("Data/average_timesteps_dagger3.txt", avg_timesteps, fmt='%.3f')

