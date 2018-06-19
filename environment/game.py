__author__ = 'kimangkhun'
from utility.utils import compute_flows
import numpy as np
import pandas as pd
from utility.visualize import visualize
import matplotlib.pyplot as plt

np.random.seed(1) # reproducibility

class Environment:
    def __init__(self):
        # TODO: cascading failure
        # TODO: action_sample, observation_space

        # Public:
        self.action_space_size = 64
        self.amt_lines = 6

        # Private:
        # Create electrical network
        F = np.zeros((14, 2), dtype=int)
        F[0] = np.array([1, 5])     # Generator
        F[1] = np.array([2, 6])     # Generator
        F[2] = np.array([3, 7])     # Generator
        F[3] = np.array([4, 8])     # Generator
        F[4] = np.array([5, 6])
        F[5] = np.array([5, 8])
        F[6] = np.array([5, 9])
        F[7] = np.array([6, 7])
        F[8] = np.array([7, 8])
        F[9] = np.array([7, 11])    # External system
        F[10] = np.array([7, 12])   # Consumer
        F[11] = np.array([8, 9])
        F[12] = np.array([8, 13])   # Consumer
        F[13] = np.array([9, 10])   # External system
        self._F = F

        # Injections index, it is fixed
        self._idx_injections = np.array([0, 1, 2, 3, 10, 12])

        # Mapping line id to flows id
        self._map_line = np.zeros(6, dtype=int) # there are 6 lines in our case
        self._map_line[0] = 4
        self._map_line[1] = 5
        self._map_line[2] = 6
        self._map_line[3] = 7
        self._map_line[4] = 8
        self._map_line[5] = 11

        # Load the chronics
        self._chronics = pd.read_csv('Data/_injections.csv', sep=";")
        # chronics indicator to tell which time step we use in chronics
        self._chronics_ind = 0 # start by using time step 0 of chronics
        self._set_injections()

        # Thermal limits
        """
        self._thermal_limits = np.zeros(6, dtype=float)
        for i in range(self.amt_lines):
            self._thermal_limits[i] = max(np.abs(self._chronics[self._chronics.columns[self._map_line[i]]].quantile(0.95)),
                                          np.abs(self._chronics[self._chronics.columns[self._map_line[i]]].quantile(0.05)))
        """
        self._thermal_limits = np.array([263.2638898, 161.4654948, 302.82674594,
                                         332.94697347, 117.03188555, 199.62076482])

        # line's usage percentage
        self._usage_percentage = np.zeros(self.amt_lines)

        # number of lines overflown
        self._flows, conv = compute_flows(self._F, self._injections, self._idx_injections)
        self._check_overflow(self._flows)

        # visualization
        self.render(show=False)

        # solved
        self._solved = False

    def render(self, show=True):
        if not show:
            plt.ion()
            fig, ax = plt.subplots(figsize=(7,7))
            self.vis = visualize(ax)
        else:
            if self._solved:
                solved_injections = np.copy(self._solved_flows[0])
                solved_injections = np.append(solved_injections, [self._solved_flows[1], self._solved_flows[2], self._solved_flows[3],
                                            self._solved_flows[10], self._solved_flows[12],
                                            self._solved_flows[9], self._solved_flows[13]])
                solved_injections = np.around(solved_injections, decimals=2)
                self.vis.draw_lines(self._solved_action, self._solved_lines_overflown,
                                    self._solved_usage_percentage, solved_injections)
                self._solved = False
            injections = np.copy(self._flows[0])
            injections = np.append(injections,
                                          [self._flows[1], self._flows[2], self._flows[3],
                                           self._flows[10], self._flows[12],
                                           self._flows[9], self._flows[13]])
            injections = np.around(injections, decimals=2)
            self.vis.draw_lines(self._last_action, self._lines_overflown, self._usage_percentage, injections)

    def _set_injections(self):
        self._injections = np.array([[self._chronics["generator1"][self._chronics_ind]],
                                   [self._chronics["generator2"][self._chronics_ind]],
                                   [self._chronics["generator3"][self._chronics_ind]],
                                   [self._chronics["generator4"][self._chronics_ind]],
                                   [self._chronics["consumer1"][self._chronics_ind]],
                                   [self._chronics["consumer2"][self._chronics_ind]]])

    def _check_overflow(self, flows):
        count = 0
        self._lines_overflown = np.zeros(self.amt_lines, dtype=int)
        for i in range(self.amt_lines):
            if abs(flows[self._map_line[i]]) > 1.05 * self._thermal_limits[i]:
                count += 1
                self._lines_overflown[i] = 1
            self._usage_percentage[i] = np.around(flows[self._map_line[i]] / self._thermal_limits[i], decimals=2)
        return count

    def _validate_action(self, action):
        if len(action) != 6:
            print("Invalid action : expect array of size 6, but get {}".format(len(action)))
            raise ValueError
        nb_cut = 6 - len(np.nonzero(action)[0])
        if nb_cut > 3:
            info = "Disconnect too many lines"
            return False, info
        # Three-line disconnections
        if action[0] == 0 and action[1] == 0 and action[2] == 0:
            info = "Disconnect generator 1"
            return False, info
        if action[0] == 0 and action[1] == 0 and action[5] == 0:
            info = "There are two grids"

        if action[1] == 0 and action[2] == 0 and action[3] == 0:
            info = "Disconnect generator [1-2]"
            return False, info
        if action[1] == 0 and action[2] == 0 and action[4] == 0:
            info = "There are two grids"
            return True, info
        if action[1] == 0 and action[3] == 0 and action[5] == 0:
            info = "There are two grids"
            return True, info
        if action[1] == 0 and action[4] == 0 and action[5] == 0:
            info = "Isolate generator-consumer without external"
            return False, info

        # Two-line disconnections
        if action[0] == 0 and action[3] == 0:
            info = "Disconnect generators 2"
            return False, info
        if action[0] == 0 and action[4] == 0:
            info = "There are two grids"
            return True, info
        if action[2] == 0 and action[5] == 0:
            info = "Disconnect external 2"
            return False, info
        if action[3] == 0 and action[4] == 0:
            info = "There are two grids"
            return True, info

        return True, None

    def seed(self, nb):
        np.random.seed(nb)

    def step(self, action):

        valid, info = self._validate_action(action)
        if not valid:
            return None, -1, True, info

        self._last_action = np.copy(action)

        connect = np.nonzero(action)

        if len(connect[0]) == self.amt_lines: # no line disconnection
            flows, conv = compute_flows(self._F, self._injections, self._idx_injections)
            nb_overflow = self._check_overflow(flows)
        else:
            disconnect = np.setdiff1d(np.arange(self.amt_lines), connect)

            disconnect = self._map_line[disconnect[:]]

            # add the line injection index
            idx_injections = np.append(self._idx_injections, disconnect)

            # force injection in the lines to be zero
            injections = np.copy(self._injections)
            for i in range(len(disconnect)):
                injections = np.vstack((injections, [[0]]))

            self._flows, conv = compute_flows(self._F, injections, idx_injections)
            nb_overflow = self._check_overflow(self._flows)

        if nb_overflow == 0:
            # overflow is solved
            self._solved = True
            self._solved_flows = np.copy(self._flows)
            self._solved_action = np.copy(self._last_action)
            self._solved_usage_percentage = np.copy(self._usage_percentage)
            self._solved_lines_overflown = np.copy(self._lines_overflown)

            # load new case
            if self._chronics_ind == 8227:      # the end of the chronics
                self._chronics_ind = 0
            else:
                self._chronics_ind += 1

            self._set_injections()              # update injections
            self._flows = np.array(self._chronics.loc[self._chronics_ind][0:14])
            self._last_action = np.ones(self.amt_lines, dtype=int)
            self._check_overflow(self._flows)         # update overflown lines
            info = "No overflows"
            return self._flows.flatten(), 1, False, info

        elif nb_overflow < 3:
            state = self._flows.flatten()             # no need to update injections
            info = "There are lines overflown"
            return state, -nb_overflow * 0.1, False, info
        else:
            info = "Too many lines overflown"
            return None, -1, True, info


    def reset(self):
        self._solved = False
        self._chronics_ind = np.random.randint(0, 8228)
        self._set_injections()
        self._flows = np.array(self._chronics.loc[self._chronics_ind][0:14])    # Because last element is the convergence flag
        self._check_overflow(self._flows)
        self._last_action = np.ones(self.amt_lines, dtype=int)

        return self._flows.flatten()

    def simulate(self, action):
        valid, info = self._validate_action(action)
        if not valid:
            return None, -1, True, info

        connect = np.nonzero(action)

        if len(connect[0]) == self.amt_lines:  # no line disconnection
            flows, conv = compute_flows(self._F, self._injections, self._idx_injections)
            nb_overflow = self._check_overflow(flows)
        else:
            disconnect = np.setdiff1d(np.arange(self.amt_lines), connect)

            # the line injection index start from 2
            disconnect = self._map_line[disconnect[:]]

            # add the line injection index
            idx_injections = np.append(self._idx_injections, disconnect)

            # force injection in the lines to be zero
            injections = np.copy(self._injections)
            for i in range(len(disconnect)):
                injections = np.vstack((injections, [[0]]))

            flows, conv = compute_flows(self._F, injections, idx_injections)
            nb_overflow = self._check_overflow(flows)

        if nb_overflow == 0:
            state = flows.flatten()
            return state, 1, False, info

        elif nb_overflow <= 3:
            state = flows.flatten()  # no need to update injections
            info = "There are lines overflown"
            return state, -nb_overflow * 0.1, False, info
        else:
            info = "Too many lines overflown"
            return None, -1, True, info

    def action_sample(self):
        return np.random.random_integers(0, 1, size=self.amt_lines)

if __name__ == "__main__":
    env = Environment()
    total_reward = 0.0
    for i in range(1000):
        action = env.action_sample()
        _, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            env.reset()
    print(total_reward)