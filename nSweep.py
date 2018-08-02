import IPython

from IPython.display import HTML
from IPython.display import Markdown
from IPython.display import display
from IPython.display import Image
import pandas as pd
from pandas import rolling_median
import numpy as np
import scipy as sp
import plotly.plotly as py
import plotly.graph_objs as go
from scipy.interpolate import spline, UnivariateSpline, Akima1DInterpolator,\
    PchipInterpolator
from scipy.signal import blackman, hamming, hanning
import math
import matplotlib.pyplot as plt
import os


class Sweep(object):
    Ctg = 0  # Si Units (F/sq. m)
    Cbg = 0  # Si Units (F/sq. m)
    L1 = 0  # um
    W1 = 0  # um
    L2 = 0  # um
    W2 = 0  # um
    Vtg0 = 0
    Vbg0 = 0
    Vbg_CNP = 0
    Vbg_CNP = 0
    slope = 0  # slope of charge neutrality VBG as function of VTG

    def __init__(self, filepath, row, headertf):
        self.data = np.genfromtxt(filepath, autostrip=True, skip_header=row,
                                  dtype=float, names=headertf, delimiter='\t')

        if headertf is True:
            self.Ixx = self.data['Ixx_']
            self.TG = self.data['TG_']
            self.BG = self.data['BG_']
            self.SiG = self.data['SiG_']
            self.V1 = self.data['V1_']
            self.V2 = self.data['V2_']
            self.V3 = self.data['V3_']
            self.V4 = self.data['V4_']
            self.B = self.data['B_']
            self.TA = self.data['TA_']
            self.TB = self.data['TB_']
            self.R1 = self.V1 / self.Ixx
            self.R2 = self.V2 / self.Ixx
            self.R3 = self.V3 / self.Ixx
            self.R4 = self.V4 / self.Ixx
        # None case for MegaSweeps
        elif headertf is None:
            self.Ixx = self.data[:, 2]
            self.TG = self.data[:, 1]
            self.BG = self.data[:, 0]
            self.V1 = self.data[:, 3]
            self.V2 = self.data[:, 4]
            self.V3 = self.data[:, 5]
            self.V4 = self.data[:, 6]
            self.R1 = self.V1 / self.Ixx
            self.R2 = self.V2 / self.Ixx
            self.R3 = self.V3 / self.Ixx
            self.R4 = self.V4 / self.Ixx

    def calcDV(self):
        self.D = 0.5 * 1e-6 * (self.Ctg * (self.TG - self.Vtg_CNP)/8.854e-12 -
                               self.Cbg * (self.BG - self.Vbg_CNP)/8.854e-12)
        self.Veff =( -self.slope * \
            self.BG + (self.TG - self.Vtg0)) / np.sqrt(1 + self.slope ** 2)

# Calculate Conductivity with length and Width
