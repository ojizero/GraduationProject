[0:5]
smooth = 10
cons   = 5
thresh = 75
inc    = 1.001
dec    = 0.999

=======

for i in range(6):
	plt.figure(i)
	plt.plot(splicers[i].smooth_intensities(splicers[i].measure_intensity(splicers[i].data_stream), 10))
	[plt.axvline(x=j, color='k', linestyle="--") for t in slices[i] for j in t]




import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def _euclidean_magnitude (data_point):
	return sqrt(sum(map(lambda d: d**2, data_point)))

def intensity_calculator (data_array, gyro_ratio=1.0, accl_ratio=1.0, magnitude=_euclidean_magnitude):
	gyro_data = np.array(list(map(magnitude, data_array[:,0:3])))
	accl_data = np.array(list(map(magnitude, data_array[:,3:6])))
	return ((gyro_ratio * gyro_data) + (accl_ratio * accl_data))

def smooth_intensities (data_array, averaging_window=11):
	def _averager (index):
		return np.nan_to_num(np.average((data_array[(index - averaging_window//2):(index + averaging_window//2)])))
	if False:
		pass
	else:
		slicer_index = 0  # return whole array of data
	return np.array(list(map(_averager, range(len(data_array))))[slicer_index:])

def silence_segments (intensities, number_of_samples=5, threshold=10, increase_factor=1.0001, decrease_factor=0.9999):
	def _silence_finder (index):
		nonlocal threshold
		silent = all(np.nan_to_num(intensities[index-number_of_samples:index]) < threshold)
		if silent:
			threshold *= decrease_factor
		else:
			threshold *= increase_factor
		return silent
	# indices = map(_silence_finder, range(len(intensities)))
	indices = [index for index in range(len(intensities)) if _silence_finder(index)]
	# assuming ennu el lists would be of equal length
	# which they should
	return list ( zip (
		[xi for i, xi in enumerate(indices) if i     == 0            or xi - indices[i-1] != 1],
		[xi for i, xi in enumerate(indices) if i + 1 == len(indices) or indices[i+1] - xi != 1]
	))

def splice_data (data, silence_segments):
	segments = [(silence_segments[i-1][1], silence_segments[i][0]) for i in range(1, len(silence_segments))]
	return [data[start:end,:] for start, end in segments]


data = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/Proper/alef.ba.alef.ba', delimiter=',')
both = intensity_calculator(data)
gyro = smooth_intensities(intensity_calculator(data, accl_ratio=0), 50)
silence = silence_segments(gyro)

plt.plot(gyro)
[plt.axvline(x=i, color='k', linestyle="--") for t in silence_segments for i in t]


both_smooth_10 = smooth_intensities(both, 10)
both_smooth_20 = smooth_intensities(both, 20)
both_smooth_30 = smooth_intensities(both, 30)
plt.figure(1)
plt.plot(both)
plt.plot(both_smooth_10)
plt.figure(2)
plt.plot(both)
plt.plot(both_smooth_20)
plt.figure(3)
plt.plot(both)
plt.plot(both_smooth_30)
plt.show()
