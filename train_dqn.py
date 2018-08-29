__author__ = 'kimangkhun'


if __name__=="__main__":
    from environment.game import Environment
    from agent.Agent import DQN
    from collections import deque
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    df = pd.DataFrame()
    for i in range(6):

        env = Environment()    # ironment()
        #env.seed(3)
        s = env.reset()
        obs_shape = s.shape
        ag = DQN(obs_shape)
        #ag.load_weights("Data/dqn_weight")
        #ag.epsilon = 0.01
        mobile_returns = deque(maxlen=200)
        mobile_timesteps = deque(maxlen=200)
        batch_size = 32
        avg_returns = []
        avg_timesteps = []
        EPISODES = 3000
        #max_timestep = 100000
        RENDER = True
        env.seed(1)
        #for i_ep in range(EPISODES):
        current_ts = 0
        i_ep = 0
        while i_ep < EPISODES:
            total_rewards = 0.0
            h = 0
            while True: #for h in range(200):
                """
                if RENDER:
                    q_value = ag.quality(s)
                    env.render(show=True, q_value=q_value, str_action=ag.action_space._str_actions)
                """
                env_action = np.zeros(17)
                action = ag.choose_action(s)
                env_action[11:] = np.copy(action)
                sprime, reward, done, info = env.step(env_action)
                ag.remember(s, action, reward, sprime, done)
                total_rewards += reward
                if done:
                    print("Epi: \t{}, Ts: \t{}, total rewards: \t{}, epsilon: \t{}, curr_ts: \t{}".format(i_ep, h,
                                                                                                          total_rewards,
                                                                                                          ag.epsilon,
                                                                                                          current_ts))
                    mobile_returns.append(total_rewards)
                    mobile_timesteps.append(h)
                    avg_returns.append(np.mean(mobile_returns))
                    avg_timesteps.append(np.mean(mobile_timesteps))
                    s = env.reset()
                    current_ts += h + 1
                    break
                elif h == 199:
                    #RENDER = True
                    print("Epi: \t{}, Ts: \t{}, total rewards: \t{}, epsilon: \t{}, curr_ts: \t{}".format(i_ep, h,
                                                                                                          total_rewards,
                                                                                                          ag.epsilon,
                                                                                                          current_ts))
                    mobile_returns.append(total_rewards)
                    mobile_timesteps.append(h)
                    avg_returns.append(np.mean(mobile_returns))
                    avg_timesteps.append(np.mean(mobile_timesteps))
                    s = env.reset()
                    current_ts += h + 1
                    break
                h += 1
                s = np.copy(sprime)
                if len(ag.memory) > batch_size:
                    ag.replay(batch_size)
            i_ep += 1
            ag.update_target_model()
            if i_ep % 10 == 0:
                if ag.epsilon > ag.epsilon_min:
                    ag.epsilon *= ag.epsilon_decay
            #if i_ep % 500 == 0:
                #ag.save_weights("Data/dqn_weight")
        df[i] = avg_returns
    df.to_csv("Data/multi_run_dqn.csv", sep=";", index=False)

    """

    ag.save_weights("Data/dqn_weight")
    np.savetxt("Data/average_returns_dqn.txt", avg_returns, fmt='%.3f')
    np.savetxt("Data/average_timesteps_dqn.txt", avg_timesteps, fmt='%.3f')

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
    #plt.ylim(-250, 250)
    plt.show()
    """
