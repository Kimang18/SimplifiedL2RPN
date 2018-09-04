# SimplifiedL2RPN
This project is just simplified version of the competition Learning to Run the Power Network. The objective is to implement Reinforcement Learning algorithms to control the power network.
In this version, there are 4 generators in green square, 2 consumers in yellow square, 2 batteries in blue square, 5 substations in purple circle, and 6 transmission lines.

Given the injections(production and consumption), the power flow in each line is calculated by DC approximation using Kirchoff's laws. We assume that at each time step, the power flow reaches its steady state.
With these flows, the transmission line is overflowed when the flow is higher than its thermal limit. The power grid controler needs to cure the existing overflow and avoid the future overflow if possible.
The possible action in this project is the line disconnection or reconnection. There are 22 possible in the action space for RL framework.

There are 4 RL agents implemented in this project: Dataset Aggregation (DAgger), Advantages Actor-Critic (A2C), Deep Q-learning (DQN), and Double Deep Q-learning (DDQN).

In order to use the code, first of all, clone this repository. Then install the required packages by typing 
`pip install -r requirements.txt` in the Terminal (you might want to create a virtual environment of python and install these packages inside it). If you do not have `pip`, please take a look in this link https://pip.pypa.io/en/stable/installing/.

To create and train another RL agent (also visualizing the power network), please take a look in train_a2c.py, train_dqn.py, and train_ddqn.py.
