import numpy as np
from skbio.diversity import beta_diversity
from skbio.stats import distance
import scipy
from scipy import stats


def data_dist(data):
    return np.mean(data, axis=1)


# to plot use:
# _ = plt.hist(stat_utils.data_dist(data), color='green', bins=20)

def class_dist(classL):
    (unique, counts) = np.unique(classL, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    x = frequencies[:, 1]
    y = frequencies[:, 0]
    return x, y


# to plot use:
# _ = plt.pie(x, labels=y, autopct='%1.1f%%', shadow=True, startangle=90)

def data_diversity(data):
    ids = list((np.linspace(1, data.shape[0], num=data.shape[0])).astype('int'))
    bc_dm = beta_diversity("braycurtis", np.absolute(data), ids)
    q = bc_dm.to_data_frame().to_numpy()
    t = np.mean(q, axis=0)
    avg_div_score = np.mean(t)
    return t, avg_div_score


# to plot use:
# _ = plt.hist(t, color='red', bins=40,label='Diversity')
# use avg_div_score as the overall score (it's just a single number)


def data_novelty(data, order=40, number=4):
    flist = []
    for i in range(data.shape[0]):
        spectrum = np.fft.fft(data[i, :])
        indexes = np.squeeze(np.array(scipy.signal.argrelextrema(abs(spectrum), comparator=np.greater, order=order)))
        freq = np.fft.fftfreq(len(spectrum))
        y = abs(spectrum) / max(abs(spectrum))
        y_n = np.squeeze(y[indexes])
        freq_n = np.squeeze(freq[indexes])
        f = freq_n[(freq_n > 0)]
        indices = np.delete(((-y_n[(freq_n > 0)]).argsort()[:number]), 0)
        flist.append(f[indices])
    flist = np.array(flist)
    flattened_f = flist.flatten()
    avg_f_per_sig = np.mean(flist, axis=1)
    glob_nov = np.mean(avg_f_per_sig)
    return flist, flattened_f, avg_f_per_sig, glob_nov


# to plot use:
# _ =  plt.hist(flattened_f, color='blue', bins=20,label='Novelty')
# use glob_nov as the overall score (it's just a single number)

def feat_RMSE(orig_data, fake_data):
    minf = []
    maxf = []
    meanf = []
    variancef = []
    skewnessf = []
    kurtosisf = []
    for i in range(orig_data.shape[0]):
        k = np.array(stats.describe(orig_data[i, :]))
        minf.append(k[1][0])
        maxf.append(k[1][1])
        meanf.append(k[2])
        variancef.append(k[3])
        skewnessf.append(k[4])
        kurtosisf.append(k[5])
    minf = np.mean(minf)
    maxf = np.mean(maxf)
    meanf = np.mean(meanf)
    variancef = np.mean(variancef)
    skewnessf = np.mean(skewnessf)
    kurtosisf = np.mean(kurtosisf)

    minff = []
    maxff = []
    meanff = []
    varianceff = []
    skewnessff = []
    kurtosisff = []
    for j in range(fake_data.shape[0]):
        l = np.array(stats.describe(fake_data[j, :]))
        minff.append(l[1][0])
        maxff.append(l[1][1])
        meanff.append(l[2])
        varianceff.append(l[3])
        skewnessff.append(l[4])
        kurtosisff.append(l[5])

    minff = np.mean(minff)
    maxff = np.mean(maxff)
    meanff = np.mean(meanff)
    varianceff = np.mean(varianceff)
    skewnessff = np.mean(skewnessff)
    kurtosisff = np.mean(kurtosisff)

    RMSE = np.sqrt((np.square(minff - minf) + np.square(maxff - maxf) + np.square(meanff - meanf) + np.square(
        varianceff - variancef) + np.square(skewnessff - skewnessf) + np.square(kurtosisff - kurtosisf)) / 6)
    return RMSE

# to use: st.feat_RMSE(data,fakedata)