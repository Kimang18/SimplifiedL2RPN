__author__ = 'kimangkhun'
from utility.utils import compute_flows
import numpy as np
import pandas as pd
from utility.visualize import visualize
import matplotlib.pyplot as plt
from collections import deque

np.random.seed(1) # reproducibility

class Graph(object):

    def __init__(self, graph_dict=None):
        """
        Initializes a graph object
        If no dictionary or None is given,
        an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the edges of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """
        If the vertex "vertex" is not in
        self.__graph_dict, a key "vertex" with an empty
        list as a value is added to the dictionary.
        Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """
        assumes that edge is of type set, tuple or list;
        between two vertices can be multiple edges!
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]
        if vertex2 in self.__graph_dict:
            self.__graph_dict[vertex2].append(vertex1)
        else:
            self.__graph_dict[vertex2] = [vertex1]

    def __generate_edges(self):
        """
        A static method generating the edges of the graph "graph".
        Edges are represented as sets with one (a loop back to the vertex)
        or two vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbor in self.__graph_dict[vertex]:
                if {neighbor, vertex} not in edges:
                    edges.append({vertex, neighbor})
        return edges

    def is_connected(self,
                     vertices_visited=None,
                     start_vertex=None):
        """ determines if the graph is connected """
        if vertices_visited is None:
            vertices_visited = set()
        gdict = self.__graph_dict
        vertices = list(gdict.keys())
        if not start_vertex:
            # choose a vertex from graph as a starting point
            start_vertex = vertices[0]
        vertices_visited.add(start_vertex)
        if len(vertices_visited) != len(vertices):
            for vertex in gdict[start_vertex]:
                if vertex not in vertices_visited:
                    if self.is_connected(vertices_visited, vertex):
                        return True
        else:
            return True
        return False

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

class Environment:
    class Action_Space:
        def __init__(self):
            self.action_size = 6
            # Create all possible action
            self._actions = []
            self._str_actions = []
            n_action = 0
            np.random.seed(1)
            while (n_action < 64):
                a = np.random.random_integers(0, 1, size=self.action_size)
                if str(a) in self._str_actions:
                    continue
                else:
                    self._actions.append(a)
                    self._str_actions.append(str(a))
                    n_action += 1

            # Help agent to reduce action space by deleting useless action
            toRemove = []
            for i in range(len(self._actions)):
                if len(np.nonzero(self._actions[i])[0]) < 4:
                    toRemove.append(i)
            for i in sorted(toRemove, reverse=True):
                del self._actions[i]
                del self._str_actions[i]
            self.n = len(self._actions)

        def __repr__(self):
            print(self._actions)

        def get(self, i):
            return self._actions[i]

        def get_str(self, i):
            return self._str_actions[i]

        def get_index(self, action):
            return self._str_actions.index(str(action))

    def __init__(self):
        # TODO: cascading failure

        # Public:
        self.amt_lines = 6
        self.action_space = self.Action_Space()

        # Private:
        # self._F is the power grid topology
        self._F = np.zeros((14, 2), dtype=float)

        # Fixed objects
        self._F[0] = np.array([1, 5.0])  # Generator1 fixed to bus 0
        self._F[1] = np.array([2, 6])  # Generator2
        self._F[10] = np.array([7.0, 11])  # Consumer11 fixed to bus 0
        self._F[12] = np.array([8.0, 12])  # Consumer12 fixed to bus 0
        self._F[13] = np.array([9, 13])  # External system13
        self.__default_vertex = np.arange(1, 14, dtype=float)

        # Default configuration
        self._F[2] = np.array([3, 7.0])  # Generator3
        self._F[3] = np.array([4, 8.0])  # Generator4
        self._F[9] = np.array([7.0, 10])  # External system10
        self._F[4] = np.array([5.0, 6])  # Line L1
        self._F[6] = np.array([5.0, 9])  # Line L3
        self._F[7] = np.array([6, 7.0])  # Line L4
        self._F[11] = np.array([8.0, 9])  # Line L6
        self._F[5] = np.array([5.0, 8.0])  # Line L2
        self._F[8] = np.array([7.0, 8.0])  # Line L5

        # Set of vertex in default case
        self.__graph = Graph()
        for i in range(13):
            self.__graph.add_vertex(self.__default_vertex[i])

        for i in range(14):
            self.__graph.add_edge(self._F[i])

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
        self._thermal_limits = np.array([237.50225641352628, 499.71778031480625, 255.05779338562377,
                                         389.01455317609054, 450.797754748493, 530.2488324717878])

        # line's usage percentage
        self._usage_percentage = np.zeros(self.amt_lines)

        # number of lines overflown
        self._flows, conv = compute_flows(self._F, self._injections, self._idx_injections)
        self._check_overflow(self._flows)

        # visualization
        self.render(show=False)

        # solved
        self._solved = False

        # time series
        self.flows_series = []
        for i in range(self.amt_lines):
            self.flows_series.append(deque(maxlen=100))
            self.flows_series[i].append(self._flows[self._map_line[i]])

        # reward signal
        #self.rew_signal = deque(maxlen=100)

    def render(self, show=True, q_value=None, str_action=None):
        if not show:
            plt.ion()
            self.vis = visualize(self._thermal_limits)
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
            self.vis.draw_lines(self._last_action, self._lines_overflown,
                                self._usage_percentage, injections,
                                flow_series=None, rew_signal=None, q_value=q_value, str_actions=str_action)#, self.flows_series, self.rew_signal)

    def _set_injections(self):
        self._injections = np.array([[self._chronics["generator1"][self._chronics_ind]],
                                   [self._chronics["generator2"][self._chronics_ind]],
                                   [self._chronics["generator3"][self._chronics_ind]],
                                   [self._chronics["generator4"][self._chronics_ind]],
                                   [self._chronics["consumer1"][self._chronics_ind]],
                                   [self._chronics["consumer2"][self._chronics_ind]]])

    def _check_overflow(self, flows, simulate=False):
        count = 0
        self._lines_overflown = np.zeros(self.amt_lines, dtype=int)
        for i in range(self.amt_lines):
            if simulate:
                if abs(flows[self._map_line[i]]) > 1.05 * self._thermal_limits[i]:
                    count += 1
            else:
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
            return False, info

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

    def _validate_topology(self):
        for i in range(14):
            if len(np.nonzero(self._F[i])[0]) != 0:
                self.__graph.add_edge(self._F[i])

        return self.__graph.is_connected()

    def seed(self, nb):
        np.random.seed(nb)

    def step(self, action_tilde):
        action = action_tilde[11:]
        connect = np.nonzero(action)
        disconnect = np.setdiff1d(np.arange(self.amt_lines), connect)
        disconnect = self._map_line[disconnect[:]]

        a = action_tilde[:11]

        self._F = np.zeros((14, 2), dtype=float)
        # Fixed objects
        self._F[0] = np.array([1, 5.0])  # Generator1 fixed to bus 0
        self._F[1] = np.array([2, 6])  # Generator2
        self._F[10] = np.array([7.0, 11])  # Consumer11 fixed to bus 0
        self._F[12] = np.array([8.0, 12])  # Consumer12 fixed to bus 0
        self._F[13] = np.array([9, 13])  # External system13
        # Node splitting by a
        graph = Graph()
        for i in range(13):
            graph.add_vertex(self.__default_vertex[i])
        if a[0]:  # Generator3
            self._F[2] = np.array([3, 7.1])
            graph.add_vertex(7.1)
        else:
            self._F[2] = np.array([3, 7.0])
        if a[1]:  # Generator4
            self._F[3] = np.array([4, 8.1])
            graph.add_vertex(8.1)
        else:
            self._F[3] = np.array([4, 8.0])
        if a[2]:  # External system10
            self._F[9] = np.array([7.1, 10])
            graph.add_vertex(7.1)
        else:
            self._F[9] = np.array([7.0, 10])
        if 4 not in disconnect:
            if a[3]:  # Origin of line L1
                self._F[4] = np.array([5.1, 6])
                graph.add_vertex(5.1)
            else:
                self._F[4] = np.array([5.0, 6])
        if 6 not in disconnect:
            if a[4]:  # Origin of line L3
                self._F[6] = np.array([5.1, 9])
                graph.add_vertex(5.1)
            else:
                self._F[6] = np.array([5.0, 9])
        if 7 not in disconnect:
            if a[5]:  # Extremity of line L4
                self._F[7] = np.array([6, 7.1])
                graph.add_vertex(7.1)
            else:
                self._F[7] = np.array([6, 7.0])
        if 11 not in disconnect:
            if a[6]:  # Origin of line L6
                self._F[11] = np.array([8.1, 9])
                graph.add_vertex(8.1)
            else:
                self._F[11] = np.array([8.0, 9])
        if 5 not in disconnect:
            if a[7]:  # Origin of line L2
                if a[8]:  # Extremity of line L2
                    self._F[5] = np.array([5.1, 8.1])
                    graph.add_vertex(5.1)
                    graph.add_vertex(8.1)
                else:
                    self._F[5] = np.array([5.1, 8.0])
                    graph.add_vertex(5.1)
            else:
                if a[8]:  # Extremity of line L2
                    self._F[5] = np.array([5.0, 8.1])
                    graph.add_vertex(8.1)
                else:
                    self._F[5] = np.array([5.0, 8.0])
        if 8 not in disconnect:
            if a[9]:  # Origin of line L5
                if a[10]:  # Extremity of line L5
                    self._F[8] = np.array([7.1, 8.1])
                    graph.add_vertex(7.1)
                    graph.add_vertex(8.1)
                else:
                    self._F[8] = np.array([7.1, 8.0])
                    graph.add_vertex(7.1)
            else:
                if a[10]:  # Extremity of line L5
                    self._F[8] = np.array([7.0, 8.1])
                    graph.add_vertex(8.1)
                else:
                    self._F[8] = np.array([7.0, 8.0])
        self.__graph = graph

        # check the configuration
        if not self._validate_topology():
            info = "Graph separation"
            state_0 = np.zeros((1, 15))
            state_1 = np.zeros((1, 15))
            #state = np.vstack((state_0, state_1))
            state = np.hstack((state_0.transpose(), state_1.transpose()))
            return state, -2, True, info
        flows, conv = compute_flows(self._F, self._injections, self._idx_injections)
        nb_overflow_1 = self._check_overflow(flows)
        nb_overflow_0 = self._nb_overflows

        if nb_overflow_0 > 0 and nb_overflow_1 > 0:
            info = "Cascading failure"
            reward = -1
            state_0 = np.zeros((1, 15))
            state_1 = np.zeros((1, 15))
            #state = np.vstack((state_0, state_1))
            state = np.hstack((state_0.transpose(), state_1.transpose()))
            return state, reward, True, info

        self._last_action = np.copy(action)

        # overflow is solved
        self._solved = True
        self._solved_flows = np.copy(flows)
        self._solved_action = np.copy(self._last_action)
        self._solved_usage_percentage = np.copy(self._usage_percentage)
        self._solved_lines_overflown = np.copy(self._lines_overflown)
        state_0 = np.append(flows.flatten(), [nb_overflow_1])
        state_0 = np.reshape(state_0, [1, len(state_0)])

        # load new case
        if self._chronics_ind == 7949:  # the end of the chronics
            self._chronics_ind = 0
        else:
            self._chronics_ind += 1

        self._set_injections()  # update injections

        if len(disconnect) == 0:
            # update flow directly
            self._flows = np.array(self._chronics.loc[self._chronics_ind][0:14])
        else:
            # recompute new flows
            self._flows, _ = compute_flows(self._F, self._injections, self._idx_injections)

        self._nb_overflows = self._check_overflow(self._flows)  # update overflown lines
        state_1 = np.append(self._flows.flatten(), [self._nb_overflows])
        state_1 = np.reshape(state_1, [1, len(state_1)])
        #state = np.vstack((state_0, state_1))
        state = np.hstack((state_0.transpose(), state_1.transpose()))


        if nb_overflow_0 > 0 and nb_overflow_1 == 0:
            if self._nb_overflows == 0:
                reward = 0.6
                info = "curative and preventive"
            else:
                reward = 0.1
                info = "curative"
        elif nb_overflow_0 == 0 and nb_overflow_1 > 0:
            if self._nb_overflows == 0:
                reward = -0.5
                info = "preventive"
            else:
                reward = -1.0
                info = "overflow causal"
        else:
            if self._nb_overflows == 0:
                reward = 0.1
                info = "continue"
            else:
                reward = -0.5
                info = "not preventive"

        return state, reward, False, info

    def reset(self):
        # Reset the topology
        # Fixed objects
        self._F[0] = np.array([1, 5.0])  # Generator1 fixed to bus 0
        self._F[1] = np.array([2, 6])  # Generator2
        self._F[10] = np.array([7.0, 11])  # Consumer11 fixed to bus 0
        self._F[12] = np.array([8.0, 12])  # Consumer12 fixed to bus 0
        self._F[13] = np.array([9, 13])  # External system13
        self.__default_vertex = np.arange(1, 14, dtype=float)

        # Default configuration
        self._F[2] = np.array([3, 7.0])  # Generator3
        self._F[3] = np.array([4, 8.0])  # Generator4
        self._F[9] = np.array([7.0, 10])  # External system10
        self._F[4] = np.array([5.0, 6])  # Line L1
        self._F[6] = np.array([5.0, 9])  # Line L3
        self._F[7] = np.array([6, 7.0])  # Line L4
        self._F[11] = np.array([8.0, 9])  # Line L6
        self._F[5] = np.array([5.0, 8.0])  # Line L2
        self._F[8] = np.array([7.0, 8.0])  # Line L5

        self._solved = False
        self._chronics_ind = np.random.randint(0, 7949)
        self._set_injections()
        self._flows = np.array(self._chronics.loc[self._chronics_ind][0:14])    # Because last element is the convergence flag
        self._nb_overflows = self._check_overflow(self._flows)
        self._last_action = np.ones(self.amt_lines, dtype=int)
        nb_overflow_0 = self._nb_overflows
        state_0 = np.append(self._flows.flatten(), [nb_overflow_0])
        state_0 = np.reshape(state_0, [1, len(state_0)])

        self._chronics_ind += 1
        self._set_injections()
        self._flows = np.array(self._chronics.loc[self._chronics_ind][0:14])  # Because last element is the convergence flag
        self._nb_overflows = self._check_overflow(self._flows)
        self._last_action = np.ones(self.amt_lines, dtype=int)
        nb_overflow_1 = self._nb_overflows
        state_1 = np.append(self._flows.flatten(), [nb_overflow_1])
        state_1 = np.reshape(state_1, [1, len(state_1)])
        state = np.hstack((state_0.transpose(), state_1.transpose()))

        return state

    def simulate(self, action_tilde):
        action = action_tilde[11:]
        connect = np.nonzero(action)
        disconnect = np.setdiff1d(np.arange(self.amt_lines), connect)
        disconnect = self._map_line[disconnect[:]]

        a = action_tilde[:11]

        F = np.zeros((14, 2), dtype=float)
        # Fixed objects
        F[0] = np.array([1, 5.0])  # Generator1 fixed to bus 0
        F[1] = np.array([2, 6])  # Generator2
        F[10] = np.array([7.0, 11])  # Consumer11 fixed to bus 0
        F[12] = np.array([8.0, 12])  # Consumer12 fixed to bus 0
        F[13] = np.array([9, 13])  # External system13
        # Node splitting by a
        graph = Graph()
        for i in range(13):
            graph.add_vertex(self.__default_vertex[i])
        if a[0]:  # Generator3
            F[2] = np.array([3, 7.1])
            graph.add_vertex(7.1)
        else:
            F[2] = np.array([3, 7.0])
        if a[1]:  # Generator4
            F[3] = np.array([4, 8.1])
            graph.add_vertex(8.1)
        else:
            F[3] = np.array([4, 8.0])
        if a[2]:  # External system10
            F[9] = np.array([7.1, 10])
            graph.add_vertex(7.1)
        else:
            F[9] = np.array([7.0, 10])
        if 4 not in disconnect:
            if a[3]:  # Origin of line L1
                F[4] = np.array([5.1, 6])
                graph.add_vertex(5.1)
            else:
                F[4] = np.array([5.0, 6])
        if 6 not in disconnect:
            if a[4]:  # Origin of line L3
                F[6] = np.array([5.1, 9])
                graph.add_vertex(5.1)
            else:
                F[6] = np.array([5.0, 9])
        if 7 not in disconnect:
            if a[5]:  # Extremity of line L4
                F[7] = np.array([6, 7.1])
                graph.add_vertex(7.1)
            else:
                F[7] = np.array([6, 7.0])
        if 11 not in disconnect:
            if a[6]:  # Origin of line L6
                F[11] = np.array([8.1, 9])
                graph.add_vertex(8.1)
            else:
                F[11] = np.array([8.0, 9])
        if 5 not in disconnect:
            if a[7]:  # Origin of line L2
                if a[8]:  # Extremity of line L2
                    F[5] = np.array([5.1, 8.1])
                    graph.add_vertex(5.1)
                    graph.add_vertex(8.1)
                else:
                    F[5] = np.array([5.1, 8.0])
                    graph.add_vertex(5.1)
            else:
                if a[8]:  # Extremity of line L2
                    F[5] = np.array([5.0, 8.1])
                    graph.add_vertex(8.1)
                else:
                    F[5] = np.array([5.0, 8.0])
        if 8 not in disconnect:
            if a[9]:  # Origin of line L5
                if a[10]:  # Extremity of line L5
                    F[8] = np.array([7.1, 8.1])
                    graph.add_vertex(7.1)
                    graph.add_vertex(8.1)
                else:
                    F[8] = np.array([7.1, 8.0])
                    graph.add_vertex(7.1)
            else:
                if a[10]:  # Extremity of line L5
                    F[8] = np.array([7.0, 8.1])
                    graph.add_vertex(8.1)
                else:
                    F[8] = np.array([7.0, 8.0])

        for i in range(14):
            if len(np.nonzero(F[i])[0]) != 0:
                graph.add_edge(F[i])
        valid_action = graph.is_connected()

        # check the configuration
        if not valid_action:
            info = "Graph separation"
            return np.zeros_like(self._flows.flatten()), -2, True, info
        flows, conv = compute_flows(F, self._injections, self._idx_injections)
        nb_overflow_1 = self._check_overflow(flows, simulate=True)
        nb_overflow_0 = self._nb_overflows

        curr_chronics_ind = self._chronics_ind
        # load new case
        if self._chronics_ind == 7949:  # the end of the chronics
            self._chronics_ind = 0
        else:
            self._chronics_ind += 1
        self._set_injections()  # update injections
        # recompute new flows
        flows, _ = compute_flows(F, self._injections, self._idx_injections)
        nb_overflow_2 = self._check_overflow(flows, simulate=True)  # update overflown lines

        # Revert back to current index
        self._chronics_ind = curr_chronics_ind
        self._set_injections()

        state_0 = np.zeros((1, 15))
        state_1 = np.zeros((1, 15))
        # state = np.vstack((state_0, state_1))
        state = np.hstack((state_0.transpose(), state_1.transpose()))

        if nb_overflow_0 > 0 and nb_overflow_1 > 0:
            info = "cascading failure"
            reward = -1.0

        elif nb_overflow_0 > 0 and nb_overflow_1 == 0:
            if nb_overflow_2 == 0:
                reward = 0.6
                info = "curative and preventive"
            else:
                reward = 0.1
                info = "curative"
        elif nb_overflow_0 == 0 and nb_overflow_1 > 0:
            if nb_overflow_2 > 0:
                reward = -1.0
                info = "overflow causal"
            else:
                reward = -0.5
                info = "preventive"
        else:
            if nb_overflow_2 > 0:
                reward = -0.5
                info = "not preventive"
            else:
                reward = 0.1
                info = "continue"

        return state, reward, False, info

    def action_sample(self):
        return np.random.random_integers(0, 1, size=17)

if __name__ == "__main__":

    env = Environment()
    s = env.reset()
    print(s[0][1])
    i = 0
    act = np.zeros(17)
    act[11:] = np.ones(6)
    sprime, reward, done, info = env.step(act)
    print(sprime[0][0])
    print(reward, done)
    """
    while i < 100000:
        act = env.action_sample()
        _, reward_sim, done_sim, info_sim = env.simulate(act)
        sprime, reward, done, info = env.step(act)
        print(reward == reward_sim, done == done_sim, info == info_sim)
        i += 1
        if done:
            env.reset()

    
    F = np.zeros((14, 2), dtype=int)
    F[0] = np.array([1, 5])  # Generator
    F[1] = np.array([2, 6])  # Generator
    F[2] = np.array([3, 7])  # Generator
    F[3] = np.array([4, 8])  # Generator
    F[4] = np.array([5, 6])
    F[5] = np.array([5, 8])
    F[6] = np.array([5, 9])
    F[7] = np.array([6, 7])
    F[8] = np.array([7, 8])
    F[9] = np.array([7, 10])  # External system
    F[10] = np.array([7, 11])  # Consumer
    F[11] = np.array([8, 9])
    F[12] = np.array([8, 12])  # Consumer
    F[13] = np.array([9, 13])  # External system

    graph = Graph()
    for i in range(14):
        graph.add_edge(F[i])

    print(graph)
    print(graph.is_connected())
    """