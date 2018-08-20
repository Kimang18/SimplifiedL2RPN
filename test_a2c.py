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
    from tqdm import tqdm
    import pandas as pd
    df = pd.DataFrame()
    for i in range(6):

        env = Environment()
        #env.seed(3) #to get the same result in video
        s = env.reset()
        obs_shape = s.shape
        n_action_shape = env.amt_lines
        ag = Actor_Critic(obs_shape, n_action_shape)
        print(obs_shape)
        #ag.load_weights("Data")
        batch_size = 32
        mobile_returns = deque(maxlen=200)
        mobile_timesteps = deque(maxlen=200)
        avg_returns = []
        avg_timesteps = []
        EPISODES = 3000
        env.seed(1)
        RENDER = True
        #for i_ep in range(EPISODES):
        current_ts = 0
        i_ep = 0
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
                action = ag.choose_action(s)
                env_action[11:] = np.copy(action)
                sprime, reward, done, info = env.step(env_action)
                ag.learn_step(s, action, reward, sprime, done)     # actor critic - online learning
                #ag.remember(s, action, reward, sprime, done)
                total_rewards += reward
                if done:
                    print("Epi: \t{}, Ts: \t{}, total rewards: \t{}, curr_ts: \t{}".format(i_ep, h, total_rewards, current_ts))
                    mobile_returns.append(total_rewards)
                    mobile_timesteps.append(h)
                    avg_returns.append(np.mean(mobile_returns))
                    avg_timesteps.append(np.mean(mobile_timesteps))
                    s = env.reset()
                    current_ts += h+1
                    break
                elif h == 199:
                    #RENDER = False
                    print("Epi: \t{}, Ts: \t{}, total rewards: \t{}, curr_ts: \t{}".format(i_ep, h, total_rewards, current_ts))
                    mobile_returns.append(total_rewards)
                    mobile_timesteps.append(h)
                    avg_returns.append(np.mean(mobile_returns))
                    avg_timesteps.append(np.mean(mobile_timesteps))
                    s = env.reset()
                    current_ts += h+1
                    break
                s = np.copy(sprime)
                h += 1

            i_ep += 1
            #ag.replay(batch_size)

            #if i_ep % 500 == 0:
                #ag.save_weights("Data")
        df[i] = avg_returns
    df.to_csv("Data/multi_run_a2c.csv", sep=";", index=False)

    """

    ag.save_weights("Data")  # save weights after training
    np.savetxt("Data/average_returns_a2c.txt", avg_returns, fmt='%.3f')
    np.savetxt("Data/average_timesteps_a2c.txt", avg_timesteps, fmt='%.3f')


    plt.ioff()

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_returns)
    plt.ylabel("average return")
    plt.xlabel("number of episodes")
    plt.title("Average return over episodes")
    plt.xlim(0, EPISODES)
    #plt.ylim(-, 250)

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_timesteps)
    plt.ylabel("number of timesteps")
    plt.xlabel("number of episodes")
    plt.title("Average timesteps over episodes")
    plt.xlim(0, EPISODES)
    #plt.ylim(-250, 250)
    plt.show()
    """



