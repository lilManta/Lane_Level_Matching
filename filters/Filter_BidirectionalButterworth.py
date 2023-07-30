from scipy.signal import butter, lfilter
import pickle

#####################################################
## This script implements a Butterworth bandpass,
## lowpass or highpass filter described in functions
#####################################################
## Concept: Christopher Bennett
## Edited: Frederic Brenner
#####################################################
## Date: 06.2020
#####################################################


def butter_bandpass(lowcut, highcut, fs, order=5, filterType='lowpass'):

    """ Find a and b values for butterworth filter """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if filterType == 'lowpass':
        b, a = butter(order, high, btype='lowpass')
    elif filterType == 'bandpass':
        b, a = butter(order, [low, high], btype='bandpass')
    else:
        b, a = butter(order, low, btype='highpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, filterType='low'):

    """ Filter the data using a,b determined by butterworth """

    b, a = butter_bandpass(lowcut, highcut, fs, order=order, filterType=filterType)
    y = lfilter(b, a, data)
    return y


def new_timescale_for_video(data, videoStart='', nsamples=0):

    """
        Finds the necessary new timescale to map to video within BBPF and returns this
    """

    from datetime import datetime, timedelta
    import numpy as np

    # determine when the video starts
    temp_string = videoStart.rstrip()
    videoStart_datetime = np.datetime64(datetime.strptime(temp_string, '%Y:%m:%d,%H:%M.%S'))

    # determine first and last timestamp
    first_timestamp = data.time.values[0]
    last_timestamp = data.time.values[-1]

    # determine the average sampletime
    totaltime = (last_timestamp - first_timestamp).astype('float') * 10 ** -9
    sampletime = totaltime / nsamples

    # Make new timescale to find in video
    _start_zero = datetime(2000, 1, 1, 0, 0, 0, 0)
    t = []
    counter = 0
    for x in range(0, nsamples):
        if data.time.values[x] > videoStart_datetime:
            counter += 1
            t.append(_start_zero + timedelta(seconds=counter * sampletime))
        else:
            t.append(_start_zero)

    return t


def BBPF(data, signal='X', sampleRate=100, lowCut=0.0, highCut=0.0, order=4):

    """ calls the butterworth bandpass filter (with cascade filter, fitlers in both directions)

        data            =   the dataframe to be filtered
        signal          =   the column in data which is to be filtered

        sampleRate      =   the sample rate of the data
        lowcut          =   the lower frequency limit of the BPF
        highcut         =   the upper frequency limit of the BPF

        order           =   the order of the BPF (Order ^ => sharper cutoff)
        plotResults     =   should the frequency response and filtered data be plotted

        viewTime        =   String of time wanted to inspect    => '%Y:%m:%d,%H:%M'         Ex: viewTime='2018:04:09,11:14'
        videoStart      =   String the start time of Video      => '%Y:%m:%d,%H:%M.%S'      Ex: videoStart='2018:04:09,10:51.14'

    """

    # Which signal is to be filtered
    if signal == 'X':
        noisyData = data.X.values
    elif signal == 'Y':
        noisyData = data.Y.values
    else:
        noisyData = data.Z.values

    # Bandpass filter data
    if highCut != 0.0 and lowCut != 0.0:    # Both passed           => Bandwidth filter
        filterType = 'bandpass'
    elif lowCut == 0.0:                     # only highcut passed   => lowpass filter
        filterType = 'lowpass'
    else:                                   # only lowcut passed    => highpass filter
        filterType = 'highpass'

    # to avoid offset of filter use cacade Bypassfilter (once in each direction)
    y = cascadeFilter(noisyData, lowCut, highCut, sampleRate, order=order, filterType=filterType)

    return y


def cascadeFilter(noisyData, lowCut, highCut, sampleRate, order=4, filterType='lowpass'):

    """ This uses the method of two butterworth filters (forwards and backwards) to get rid of offset"""

    import numpy as np

    y = butter_bandpass_filter(noisyData, lowCut, highCut, sampleRate, order=order, filterType=filterType)
    y_backwards = np.flip(y)
    _y = butter_bandpass_filter(y_backwards, lowCut, highCut, sampleRate, order=order, filterType=filterType)
    y = np.flip(_y)

    return y


def colorManouver(time, data, start, stop):

    """ To see manouver in plot """
    import bisect
    import numpy as np
    from datetime import datetime

    _start = bisect.bisect_left(time, start)
    _stop = bisect.bisect_right(time, stop)
    print(_start)
    manouver = data.copy()
    manouver[:_start] = np.nan
    manouver[_stop:] = np.nan

    return manouver


def testBandWidthFilter():

    """
        To view what the Budderworthfilter is producing
    """
    # Import saved Acc, Gps, Gyro and Orient Data
    import pickle
    with open('Measurements/Audi_A3/2018_04_09/pandas/datetimeFormat/Acc_datetime.pkl', 'rb') as package:
        Acc = pickle.load(package)
    with open('Measurements/Audi_A3/2018_04_09/pandas/datetimeFormat/Gyro_datetime.pkl', 'rb') as package:
        Gyro = pickle.load(package)
    with open('Measurements/Audi_A3/2018_04_09/pandas/datetimeFormat/Orient_datetime.pkl', 'rb') as package:
        Orient = pickle.load(package)

    # filter data and plot
    test = BBPF(Acc, signal='X', sampleRate=100, highCut=0.15, order=4)


if __name__ == '__main__':

    with open('Measurements/Audi_A3/2018_04_09/pandas/datetimeFormat/Acc_datetime.pkl', 'rb') as package:
        Acc = pickle.load(package)

    ######## Butterworth Bandpassfilter ###########################
    FilteredSignal = BBPF(Acc, signal='X', sampleRate=100, highCut=0.35, order=4)
    ##############################################################
