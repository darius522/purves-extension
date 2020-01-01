import sys, os
import shutil
from shutil import copyfile
import csv
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess #standard for offline processing
import essentia
import re
import glob
from fractions import Fraction
import matplotlib.patches as mpatches
from scipy.signal import hanning, resample

shrutis = np.array([16/15,10/9,32/27,6/5,27/20,45/32,8/5,27/16,16/9,9/5])
shrutis_dict = {
"16/15" : 0,
"10/9" : 1,
"32/27" : 2,
"6/5" : 3,
"27/20" : 4,
"45/32" : 5,
"8/5" : 6,
"27/16" : 7,
"16/9" : 8,
"9/5" : 9
}


if len(sys.argv) > 1:
    folderpath = sys.argv[1]
    newDir = "./tmpDir"
    if not os.path.isdir(newDir):
    	os.mkdir(newDir)

#generic variables
num_files = len(glob.glob(folderpath+"/*.wav",recursive=False))
count = 0
M = 8192
N = 8192
H = 512
fs = 48000
num_peaks = 16
total_freqs = []
total_mags = []
overal_mag_mean = []
overal_freq_mean = []

#Init algos.
spectrum = ess.Spectrum(size = N)
window = ess.Windowing(size = M, type = "hann")
peaks = ess.SpectralPeaks(sampleRate = fs, maxPeaks = num_peaks, minFrequency = 150, maxFrequency = 450, magnitudeThreshold=-90)

for root, dirs, files in os.walk(folderpath):
	for file in files:

		if os.path.splitext(file)[1] == '.wav':

			currentfilePath = os.path.abspath(('{}/{}'.format(root,file)))
			x = ess.MonoLoader(filename = currentfilePath, sampleRate = fs)()
			num_frames = int(np.floor(len(x)/H))
			freqs = []
			mags = []

			for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): 

				mX = spectrum(window(frame))
				f, m = peaks(mX)

				if len(f) == num_peaks:
					freqs.append(f)
					mags.append(m)

			if len(freqs)>0:
				mean_freq_file = np.mean(freqs, axis=0, dtype=np.float64)
				mean_mag_file = np.mean(mags, axis=0, dtype=np.float64)
				total_freqs.append(mean_freq_file)
				total_mags.append(mean_mag_file)

			count = count+1
			print(count)

print(np.shape(total_freqs))
overal_freq_mean = np.mean(total_freqs, axis=0, dtype=np.float64)
overal_mag_mean = np.mean(total_mags, axis=0, dtype=np.float64)

idx_fund = overal_mag_mean.argmax()
fund_shruti_ratios = overal_freq_mean[idx_fund]*shrutis

plt.stem(fund_shruti_ratios[:9],overal_mag_mean[idx_fund+1:idx_fund+10], linefmt='r:',markerfmt='ro')
plt.stem(overal_freq_mean,overal_mag_mean, 'b')

font1 = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }

font2 = {'family': 'arial',
        'color':  'darkred',
        'weight': 'bold',
        'size': 8,
        }

font3 = {'family': 'arial',
        'color':  'blue',
        'weight': 'bold',
        'size': 8,
        }

# Only freq
for idx in range(int(len(overal_freq_mean))):
	freq = overal_freq_mean[idx]
	mag = overal_mag_mean[idx]
	ratio = Fraction(int(freq),int(overal_freq_mean[0])).limit_denominator(260)
	plt.text(freq, 0.16*np.max(overal_mag_mean) + mag, 
		str(int(freq))+'Hz', 
		horizontalalignment='center',
		fontdict=font1)

# # Compute ratio based off fundamental
# for idx in range(int(len(overal_freq_mean))):
# 	freq = overal_freq_mean[idx]
# 	mag = overal_mag_mean[idx]
# 	ratio = Fraction(int(freq),int(overal_freq_mean[idx_fund])).limit_denominator(260)
# 	plt.text(freq, 0.13*np.max(overal_mag_mean) + mag, 
# 		str(ratio), 
# 		horizontalalignment='center',
# 		fontdict=font2)

# # Compute closest shruti
# for idx in range(int(len(overal_freq_mean))):
# 	freq = overal_freq_mean[idx]
# 	mag = overal_mag_mean[idx]
# 	ratio = int(freq) / int(overal_freq_mean[idx_fund])
# 	idx_s = (np.abs(shrutis - ratio)).argmin()
# 	shruti = list(shrutis_dict.keys())[list(shrutis_dict.values()).index(idx_s)]
# 	plt.text(freq, 0.99*np.max(overal_mag_mean) + mag, 
# 		str(shruti), 
# 		horizontalalignment='center',
# 		fontdict=font3)

peaks_stem = mpatches.Patch(color='blue', label='Speech Spectral Peaks')
shruti_stem = mpatches.Patch(color='red', label='Shruti based off Speech F0')
plt.legend(handles=[peaks_stem,shruti_stem])

plt.xlim(0,overal_freq_mean[-1]+100)
plt.ylim(0,np.max(overal_mag_mean)*1.5)
plt.xlabel('Frequency in Hz')
plt.ylabel('Peaks Amplitude')
plt.title('Comparison of Female Telugu Speaker Spectral Profile with Tanpura Shrutis')
plt.show()