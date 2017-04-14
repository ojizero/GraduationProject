import numpy as np
from math import sqrt


def _euclidean_magnitude (data_point):
	return sqrt(sum(map(lambda d: d**2, data_point)))

## gyro data more relevant for segmentation ?
# data = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/echoed', delimiter=',')
# data_no_motion = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/echoed_no_motion', delimiter=',')

# intensities = intensity_calculator(data)
# gryo_only = intensity_calculator(data, accl_ratio=0.0)
# accl_only = intensity_calculator(data, gyro_ratio=0.0)

# intensities_no_motion = intensity_calculator(data_no_motion)
# gryo_only_no_motion = intensity_calculator(data_no_motion, accl_ratio=0.0)
# accl_only_no_motion = intensity_calculator(data_no_motion, gyro_ratio=0.0)

# plt.figure(1)
# plt.plot(intensities)
# plt.plot(intensities_no_motion)

# plt.figure(2)
# plt.plot(gryo_only)
# plt.plot(gryo_only_no_motion)

# plt.figure(3)
# plt.plot(accl_only)
# plt.plot(accl_only_no_motion)

# plt.show()

class Splicer:
	def __init__ (
		self, smoothing_window=50,
		threshold=0.0, increase_factor=1.0001, decrease_factor=0.9999, samples_cutoff=50,
		accl_ratio=1.0, gyro_ratio=1.0, magnitude_method=_euclidean_magnitude
	):
		self.threshold = threshold

		self.increase_factor = increase_factor
		self.decrease_factor = decrease_factor
		self.samples_cutoff  = samples_cutoff

		self.accl_ratio = accl_ratio
		self.gyro_ratio = gyro_ratio

		self.smoothing_window  = tuple([0 for _ in range(smoothing_window)])

	# def __init__ (
	# 	self, data_array, smoothing_window=5,
	# 	startup_threshold=0.0, start_threshold_change_factor=1.0,
	# 	end_threshold=0.0, end_threshold_change_factor=1.0
	# ):
	# 	self.intensities = self.measure_intensity(data_array)
	# 	self.end_threshold     = end_threshold
	# 	self.startup_threshold = startup_threshold

	# on calling this for a single record send an array of a single record
	def measure_intensity (self, data_array=self.data_array, magnitude=self.magnitude_method):
		gyro_data = np.array(list(map(magnitude, data_array[:,0:3])))
		accl_data = np.array(list(map(magnitude, data_array[:,3:6])))
		return ((self.gyro_ratio * gyro_data) + (self.accl_ratio * accl_data))

	# on calling this for a single record send an array of a single record
	def smooth_intensities (self, data_array=self.data_array, averaging_window=self.smoothing_window):
		def _averager (index):
			return np.nan_to_num(np.average((data_array[(index - averaging_window//2):(index + averaging_window//2)])))

		if type(averaging_window) == tuple:
			# update smoothing window in instance
			self.smoothing_window = averaging_window[1:] + (data_array[0],)
			# update input data
			data_array = self.smoothing_window
			# change averaging window to int
			averaging_window = len(averaging_window)

			# slicer index is used to determine whether to
			# 	return whole array <ready data case>
			# or
			# 	only last element <real time case>
			slicer_index = -1 # return last element only
		else:
			slicer_index = 0  # return whole array of data

		return np.array(list(map(_averager, range(len(data_array))))[slicer_index:])

	def silence_segments (self, intensities, number_of_samples=self.samples_cutoff, threshold=self.threshold, increase_factor=self.increase_factor, decrease_factor=self.decrease_factor):
		def _silence_finder (index):
			nonlocal threshold
			silent = all(np.nan_to_num(intensities[index-number_of_samples:index]) < threshold)

			if silent:
				threshold *= decrease_factor
			else:
				threshold *= increase_factor

			return silent

		indices = [index for index in range(len(intensities)) if _silence_finder(index)]

		# assuming ennu el lists would be of equal length
		# which they should
		return list(zip (
			[xi for i, xi in enumerate(indices) if i     == 0            or xi - indices[i-1] != 1],
			[xi for i, xi in enumerate(indices) if i + 1 == len(indices) or indices[i+1] - xi != 1]
		))

	def splice_data (self, data, silence_segments):
		segments = [(silence_segments[i-1][1], silence_segments[i][0]) for i in range(1, len(silence_segments))]
		return [data[start:end] for start, end in segments]


if __name__ == '__main__':
	pass