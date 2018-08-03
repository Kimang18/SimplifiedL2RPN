import numpy as np
from matplotlib.pyplot import *

WIDTH = 5
HEIGHT = 5

generators = np.array([[2.5,11], [6,12], [9.5,11], [10.5,5]])
externals = np.array([[3.5,3], [10.5, 8]])
consumers = np.array([[9,3], [10.5, 10]])
substations = np.array([[3,9], [6,10], [9,9], [4,5], [9,5]])

line_width = 2

ticks = np.arange(-0.5, 12.5, 1)
colors = ['b', 'r', 'g', 'y', 'c', 'b', 'r', 'g', 'y', 'c', 'b', 'r', 'g', 'y', 'c',
          'b', 'r', 'g', 'y', 'c', 'b', 'r', 'g', 'y', 'c', 'b', 'r', 'g', 'y', 'c',
          'b', 'r', 'g', 'y', 'c', 'b', 'r', 'g', 'y', 'c', 'b' , 'r', 'g', 'y', 'c',
          'b', 'r', 'g', 'y', 'c', 'b', 'r', 'g', 'y', 'c', 'b', 'r', 'g', 'y', 'c',
          'b', 'r', 'g', 'y']

class visualize:
    def __init__(self, thermal_limits):
        self.colors = ['black', 'red']
        # for quality function
        fig, self.ax_q = subplots(figsize=(7.5, 7))

        # for network
        fig, self.ax = subplots(figsize=(7, 7))
        self.thermal_limits = thermal_limits
        # draw lines for electric components
        self.ax.plot([2.5, 3], [11, 9], linestyle='dashed', color='black', linewidth=line_width)    # generator1
        self.ax.plot([6, 6], [10, 12], linestyle='dashed', color='black', linewidth=line_width)     # generator2
        self.ax.plot([9, 9.5], [9, 11], linestyle='dashed', color='black', linewidth=line_width)    # generator3
        self.ax.plot([9, 10.5], [9, 10], linestyle='dashed', color='black', linewidth=line_width)   # consumer1
        self.ax.plot([9, 10.5], [9, 8], linestyle='dashed', color='skyblue', linewidth=line_width)    # external1
        self.ax.plot([9, 10.5], [5, 5], linestyle='dashed', color='black', linewidth=line_width)    # generator4
        self.ax.plot([9, 9], [5, 3], linestyle='dashed', color='black', linewidth=line_width)       # consumer2
        self.ax.plot([4, 3.5], [5, 3], linestyle='dashed', color='skyblue', linewidth=line_width)     # external2

        # draw electric components
        self.ax.plot(generators[:, 0], generators[:, 1], "s", color="lightgreen", markersize=25, alpha=1.0)
        self.ax.plot(externals[:, 0], externals[:, 1], "s", color="skyblue", markersize=25, alpha=1.0)
        self.ax.plot(consumers[:, 0], consumers[:, 1], "s", color="khaki" , markersize=25, alpha=1.0)
        self.ax.plot(substations[:, 0], substations[:, 1], "o", color="thistle", markersize=25, alpha=1.0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # lines for substations
        self.lines = [np.array([[3, 6], [9, 10]]),
                      np.array([[3, 9], [9, 5]]),
                      np.array([[4, 3], [5, 9]]),
                      np.array([[6, 9],[10, 9]]),
                      np.array([[9, 9],[9, 5]]),
                      np.array([[9, 4],[5, 5]])]
        # texts for substations
        self.texts = [np.array([4, 9.6]), np.array([6, 7.2]), np.array([3.6, 7]),
                      np.array([7.4, 9.6]), np.array([9.2, 7]), np.array([6, 4.6])]

        # text for electric components
        self.compo_texts = [np.array([2.8, 10]), np.array([6.1, 11]), np.array([9.2, 11.4]), np.array([9.5, 5.2]), # generators
                            np.array([10, 9.4]), np.array([9.1, 3.8]), # consumers
                            np.array([10, 8.5]), np.array([4, 3.8])] # externals

        # text for line name
        self.ax.text(4.2, 9.1, "L1", fontweight='bold')
        self.ax.text(5.7, 7.5, "L2", fontweight='bold')
        self.ax.text(3.7, 6.5, "L3", fontweight='bold')
        self.ax.text(7.1, 9.3, "L4", fontweight='bold')
        self.ax.text(9.2, 6.7, "L5", fontweight='bold')
        self.ax.text(6, 5.1, "L6", fontweight='bold')

        # for line flows
        self.axs = []
        fig = figure(figsize=(7, 8))
        self.axs.append(subplot(711))
        self.axs.append(subplot(712))
        self.axs.append(subplot(713))
        self.axs.append(subplot(714))
        self.axs.append(subplot(715))
        self.axs.append(subplot(716))
        self.axs.append(subplot(717))

    def draw_lines(self, action, lines_color, usage_percentage, injections,
                   flow_series=None, rew_signal=None, q_value=None, str_actions=None):

        lines = []
        texts = []
        compo_texts = []
        for i in np.nonzero(action)[0]:
            lines.append(self.ax.plot(self.lines[i][0, :], self.lines[i][1,:], color=self.colors[lines_color[i]], linewidth=3))
            texts.append(self.ax.text(self.texts[i][0], self.texts[i][1], str(usage_percentage[i])))
        for i in range(len(injections)):
            compo_texts.append(self.ax.text(self.compo_texts[i][0], self.compo_texts[i][1], str(injections[i])))
        if flow_series is not None:
            for i in range(len(flow_series)):
                self.axs[i].cla()
                self.axs[i].axhline(self.thermal_limits[i], xmin=0, xmax=100, color='blue')
                self.axs[i].axhline(-self.thermal_limits[i], xmin=0, xmax=100, color='blue')
                self.axs[i].plot(range(len(flow_series[i])), flow_series[i], color='black')
                self.axs[i].set_ylabel("L{}".format(i+1))
                self.axs[i].set_xlim(0, 100)
                self.axs[i].set_xticklabels([])
            self.axs[6].cla()                       # reward signal
            self.axs[6].plot(range(len(rew_signal)), rew_signal)
            self.axs[6].set_ylabel("R")
            self.axs[6].set_xlim(0, 100)
            self.axs[6].set_ylim(-1.1, 1.1)
        if q_value is not None:
            self.ax_q.cla()
            for i in range(len(q_value)):
                self.ax_q.axhline(i, linestyle=':', color='black')
            self.ax_q.barh(range(len(q_value)), q_value, height=0.9, left=np.min(q_value), edgecolor='black',
                           align='center', color=colors, ecolor='black', tick_label=str_actions)
            self.ax_q.set_ylim(-1, 10)
        show()
        pause(0.0001)
        for i in range(len(lines)):
            l = lines[i].pop(0)
            l.remove()
            texts[i].set_visible(False)
        for i in range(len(injections)):
            compo_texts[i].set_visible(False)

