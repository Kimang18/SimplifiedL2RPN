import numpy as np
from matplotlib.pyplot import *

WIDTH = 5
HEIGHT = 5

generators = np.array([[2.5,11], [6,12], [9.5,11], [10.5,5]])
externals = np.array([[3.5,3], [10.5, 8]])
consumers = np.array([[9,3], [10.5, 10]])
substations = np.array([[3,9], [6,10], [9,9], [4,5], [9,5]])

line_width = 3

"""
fig = figure()
ax = fig.add_subplot(1, 1, 1)
"""
ticks = np.arange(-0.5, 12.5, 1)

class visualize:
    def __init__(self, ax):
        self.colors = ['green', 'red']
        self.ax = ax
        # draw lines for electric components
        self.ax.plot([2.5, 3], [11, 9], color='green', linewidth=line_width)    # generator1
        self.ax.plot([6, 6], [10, 12], color='green', linewidth=line_width)     # generator2
        self.ax.plot([9, 9.5], [9, 11], color='green', linewidth=line_width)    # generator3
        self.ax.plot([9, 10.5], [9, 10], color='red', linewidth=line_width)   # consumer1
        self.ax.plot([9, 10.5], [9, 8], color='blue', linewidth=line_width)    # external1
        self.ax.plot([9, 10.5], [5, 5], color='green', linewidth=line_width)    # generator4
        self.ax.plot([9, 9], [5, 3], color='red', linewidth=line_width)       # consumer2
        self.ax.plot([4, 3.5], [5, 3], color='blue', linewidth=line_width)     # external2

        # draw electric components
        self.ax.plot(generators[:, 0], generators[:, 1], "g8", markersize=25, alpha=1.0)
        self.ax.plot(externals[:, 0], externals[:, 1], "bo", markersize=25, alpha=1.0)
        self.ax.plot(consumers[:, 0], consumers[:, 1], "ro", markersize=25, alpha=1.0)
        self.ax.plot(substations[:, 0], substations[:, 1], "ys", markersize=25, alpha=1.0)
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
    def draw_lines(self, action, lines_color, usage_percentage, injections):

        lines = []
        texts = []
        compo_texts = []
        for i in np.nonzero(action)[0]:
            lines.append(self.ax.plot(self.lines[i][0, :], self.lines[i][1,:], color=self.colors[lines_color[i]], linewidth=4))
            texts.append(text(self.texts[i][0], self.texts[i][1], str(usage_percentage[i])))
        for i in range(len(injections)):
            compo_texts.append(text(self.compo_texts[i][0], self.compo_texts[i][1], str(injections[i])))
        show()
        pause(0.7)
        for i in range(len(lines)):
            l = lines[i].pop(0)
            l.remove()
            texts[i].set_visible(False)
        for i in range(len(injections)):
            compo_texts[i].set_visible(False)

    def draw_injections(self, injections):
        texts = []
