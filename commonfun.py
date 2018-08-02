import IPython
import plotly
import json
import numpy as np
import math
import scipy as sp
from scipy import stats
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import blackman, hamming, hanning
from detect_peaks import detect_peaks  # Need detect_peaks for findgap


def sigmaxx(Rxx, Rxy, Width, Length):
    sxx = 25812.81 * Rxx / (Rxx ** 2+Rxy ** 2) * Length / Width
    return sxx


def plotize(data, layout=None):  # From example in nteract notebook,plotly
    # Plot with Plotly.js using the Plotly JSON Chart Schema

    #  http://help.plot.ly/json-chart-schema/

    if layout is None:
        layout = {}

    redata = json.loads(json.dumps(data,
                                   cls=plotly.utils.PlotlyJSONEncoder))
    relayout = json.loads(json.dumps(layout,
                                     cls=plotly.utils.PlotlyJSONEncoder))

    bundle = {}
    bundle['application/vnd.plotly.v1+json'] = {
                                                'data': redata,
                                                'layout': relayout,
     }
    IPython.display.display(bundle, raw=True)


def movmean(x, N):  # Moving mean with window N
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def findmax(x, y, xmin, xmax):
    maxidx = np.where(y == np.max(y[(x > xmin) & (x < xmax)]))[0][0]
    maxvalx = x[maxidx]
    maxvaly = y[maxidx]
    return maxvalx, maxvaly, maxidx


def findmin(x, y, xmin, xmax):
    minidx = np.where(y == np.min(y[(x > xmin) & (x < xmax)]))[0][0]
    minvalx = x[minidx]
    minvaly = y[minidx]
    return minvalx, minvaly, minidx


# find the minimum y value between two x values in a x y=f(x) array
# useful for sweeps where you go back and forth while lowering the temperature
# Need detect_peaks
def findgap(x, y, lowershoulder, highershoulder):
    index = ((x > lowershoulder) & (x < highershoulder))

    irange = detect_peaks(index, edge='both')

    A = irange[::2]
    B = irange[1::2]
    if A[1] > B[1]:
        rside = A
        lside = B
    else:
        rside = B
        lside = A
    peaksx = np.empty(A.size)
    peaksy = np.empty(A.size)
    locs = np.empty(A.size)
    for idx in range(len(lside)):
        peaksx[idx], peaksy[idx], locs[idx] = \
            findmin(x[lside[idx]:rside[idx]],
                    y[lside[idx]:rside[idx]],
                    min(x[lside[idx]], x[rside[idx]]),
                    max(x[lside[idx]], x[rside[idx]]))
        locs[idx] = lside[idx]+locs[idx]
    return peaksx, peaksy, locs


# Fit a linear line to a 1/T curve with given begin and end, gives gap
# calculation as per exp(-Eg/kbT). invmin and invmax are given in 1/x
def gapfit(x, y, invmin, invmax):
    datax = 1 / x
    idx = np.where((datax > invmin) & (datax < invmax))
    datax = datax[idx]
    datay = np.log(y[idx])
    slope, intercept, r_val, p_val, std_err = stats.linregress(datax, datay)
    gap = -slope*8.6173*10**-5  # returns gap in eV
    return slope, intercept, r_val**2, gap


# Density Calculation for V sweeps, with 3 input sweeps at different B fields
# B values are given in an array with [B1, B2, B3]
def nfitVsweep(x1, x2, x3, y1, y2, y3, B):
    power = math.ceil(math.log(len(x1[1:]), 2))
    x_smooth = np.linspace(min(x1[1:]), max(x1[1:]), 2**(power+1))
    e = 1.60217662*10**-19  # Electron charge in Coulombs
    print(np.any(np.diff(x1[1:]) > 0))
    b1 = Akima1DInterpolator((x1[1:]), (y1[1:]))
    # x Must be strictly ascending, so need to feed in data backwards....
    b2 = Akima1DInterpolator((x2[1:]), (y2[1:]))
    b3 = Akima1DInterpolator((x3[1:]), (y3[1:]))
    y1_smooth = b1(x_smooth)
    y2_smooth = b2(x_smooth)
    y3_smooth = b3(x_smooth)

    slopes = np.empty(x_smooth.size)
    densities = np.empty(x_smooth.size)
    r2vals = np.empty(x_smooth.size)

    for idx in range(len(x_smooth)):
        yfit = [y1_smooth[idx], y2_smooth[idx], y3_smooth[idx]]
        slopes[idx], _, r_val, _, _ = stats.linregress(B, yfit)
        r2vals[idx] = r_val ** 2
        densities[idx] = 1 / (slopes[idx] * e)

    return densities, slopes, r2vals, x_smooth, y1_smooth, y2_smooth, y3_smooth


def reshapeidx(data):
    # gives back the idx needed to reshape the data. Pass a gate value...
    size1 = data.shape[0]
    pks = detect_peaks(data, edge='both')
    size2 = pks.shape[0]
    if (size1/size2).is_integer() is False:
        size2 += 1
    return size2, size1/size2


# The below code takes in SdH data, applies a moving average, then corrects
# for the offset by either the max or min (minmax) in a designated range.
# The code then cuts off the B sweep at a specific threshold and interpolates
# in order to average the positive B and negative B data.
def sdhRvB(data, R, ws1, Bthreshold, range, minmax):
    # Takes data in and does a moving mean with window size ws1
    # -1xBthreshold and B threshold

    B = data.B
    # B, R = rollm(B, R)
    Rmm = movmean(R, ws1)

    x1 = np.where(((B[ws1-1:]) > range[0]) & ((B[ws1-1:]) < range[1]))
    x2 = np.where((B[ws1-1:] < -range[0]) & (B[ws1-1:] > -range[1]))
    if minmax.lower() == 'min':
        y1 = np.argmin(Rmm[x1])
        y2 = np.argmin(Rmm[x2])
    elif minmax.lower() == 'max':
        y1 = np.argmax(Rmm[x1])
        y2 = np.argmax(Rmm[x2])
    else:
        print('Need to input min or max!')
        sys.exit(1)
    print(B[ws1 - 1 + y1 + x1[0][0]])
    print(B[ws1 - 1 + y2 + x2[0][0]])
    Boffset = (B[ws1 - 1 + y1 + x1[0][0]] + B[ws1 - 1 + y2 + x2[0][0]]) / 2

    B2 = B[ws1-1:]-Boffset

    power = math.ceil(math.log(len(Rmm), 2))
    B_smooth1 = np.linspace(0, Bthreshold, 2**(power+1))
    B_smooth2 = np.linspace(0, -Bthreshold, 2**(power+1))
    Bidx = np.where(abs(B2) < Bthreshold)
    B2 = B2[Bidx]
    Rmm = Rmm[Bidx]

    print(B2[-1], B2[-2], B2[0], B2[1])
    # Akima Spline requires values to be finite, real and ascending
    if (B2[-1] > B2[0]):
        Bdiff = np.diff(B2)
        diffidx = np.where(Bdiff < 0)
        #Bdiff = np.append(Bdiff, 0)
        B2 = np.delete(B2, diffidx[0]+1)
        Rmm = np.delete(Rmm, diffidx[0]+1)
        b1 = Akima1DInterpolator(B2, Rmm)
    else:

        Bdiff = np.diff(B2)
        diffidx = np.where(Bdiff > 0)
        #Bdiff = np.append(Bdiff, 0)
        B2 = np.delete(B2, diffidx[0]+1)
        Rmm = np.delete(Rmm, diffidx[0]+1)
        b1 = Akima1DInterpolator(-B2, Rmm)
    R_smooth1 = b1(B_smooth1)
    R_smooth2 = b1(B_smooth2)
    R_out = (R_smooth1+R_smooth2)/2
    return B_smooth1, R_out, B2, Rmm, Boffset


# The below code takes in a positive B range and corresponding R values
# and caclulates the fft in 1/B vs dR. Allows for a min & max 1/V threshold and
# Moving mean of window ws2
def sdhfft(B, R, ws2, mininvB, maxinvB):

    nanidx = np.isnan(R)
    B = B[~nanidx]
    R = R[~nanidx]
    # print(B)
    # print(R)
    invB = (np.divide(1, B))

    # Sorts 1/B and R together by 1/B
    invB_sorted, R_sorted = zip(*sorted(zip(invB, R)))

    x = np.array(invB_sorted)
    y = np.array(R_sorted)

    dx = np.diff(x)
    dy = np.diff(y)

    dydx = dy/dx

    x_data = x[ws2:]
    y_data = movmean(dydx[:], ws2)

    power = math.ceil(math.log(len(x_data), 2))
    x_data_smooth = np.linspace(mininvB, maxinvB, 2**(power+1))

    bi = Akima1DInterpolator(x_data, y_data)

    y_data_smooth = bi(x_data_smooth)
    # x_data_smooth = np.pad(x_data_smooth[~np.isnan(y_data_smooth)],
                          # [0, 2**(power + 1)], 'constant')
    # y_data_smooth = np.pad(y_data_smooth[~np.isnan(y_data_smooth)],
                          # [0, 2**(power + 1)], 'constant')
    # Zero Pad for higher resolution in FFT

    # FFT with different windows
    Rfft1 = sp.fft(y_data_smooth[:])
    Rfft2 = sp.fft(y_data_smooth[:]*blackman(len(y_data_smooth)))
    Rfft3 = sp.fft(y_data_smooth[:]*hamming(len(y_data_smooth)))
    Rfft4 = sp.fft(y_data_smooth[:]*hanning(len(y_data_smooth)))
    fB = (2 ** (power + 1)) / (maxinvB - mininvB)
    # fB is the 'sampling frequency' in B
    # print(Rfft.shape)
    xfft = np.linspace(x_data_smooth[0], len(x_data_smooth),
                       len(x_data_smooth)) * fB / len(x_data_smooth)
    # print(xfft.shape)
    # xfft2 =np.linspace(0,len(x_data_smooth),
    #                    len(x_data_smooth))*fB/len(x_data_smooth)

    return x_data, y_data, x_data_smooth, y_data_smooth,\
        xfft, Rfft1, Rfft2, Rfft3, Rfft4


def sdhn(B):  # units of 1e10 /sq. cm
    return B*4*1.60217662e-19/6.62607004e-34/1e14


def Rxyfit(B, Rxy, Bmin, Bmax):
    Rxyidx = np.where((B > Bmin) & (B < Bmax))
    Rxyfit = Rxy[Rxyidx]
    Bfit = B[Rxyidx]
    slope, intercept, r_val, _, _, = sp.stats.linregress(Bfit, Rxyfit)
    # plt.plot(Bfit, Rxyfit)
    rsq = r_val**2
    n = 1e-14/slope/1.60217662e-19  # return density in 1e10
    return slope, intercept, rsq, n


def nD(data, VTG, VBG):
    # Rough Calculation
    CBG = data.Cbg  # Si Units (F/sq. m)

    CTG = data.Ctg  # Si Units (F/sq. m)
    Vtg0_calc = data.Vtg0
    Vbg0_calc = data.Vbg0

    VTG_CNP = data.Vtg_CNP
    VBG_CNP = data.Vbg_CNP
    n = 1e-4 * (((CTG * (VTG - Vtg0_calc) / (1.60217662e-19))) +
                (CBG * (VBG - Vbg0_calc) / (1.60217662e-19)))
    D = 0.5 * 1e-6 * ((CTG * (VTG - VTG_CNP) / 8.854e-12) -
                      (CBG * (VBG - VBG_CNP) / 8.854e-12))
    # D is in V/nm

    return n/1e10, D
