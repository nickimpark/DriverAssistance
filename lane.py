import numpy as np
from copy import copy


class Lane(object):
    def __init__(self):
        # was the line detected in the last frame or not
        self.detected = False
        # x values for detected line pixels
        self.cur_fitx = None
        # y values for detected line pixels
        self.cur_fity = None
        # x values of the last N fits of the line
        self.prev_fitx = []
        # Mean value of last fits of the line
        self.mean_fitx = None
        # polynomial coefficients for the most recent fit
        self.current_poly = [np.array([False])]
        # best polynomial coefficients for the last iteration
        self.prev_poly = [np.array([False])]
        # buffer previous N lines
        self.N = 4

    def average_pre_lanes(self):
        tmp = copy(self.prev_fitx)
        tmp.append(self.cur_fitx)
        self.mean_fitx = np.mean(tmp, axis=0)

    def append_fitx(self):
        if len(self.prev_fitx) == self.N:
            self.prev_fitx.pop(0)
        self.prev_fitx.append(self.mean_fitx)

    def process(self, ploty):
        self.cur_fity = ploty
        self.average_pre_lanes()
        self.append_fitx()
        self.prev_poly = self.current_poly
