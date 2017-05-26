import numpy as np
from math import sqrt

#move to utils
def _euclidean_magnitude (data_point):
	return sqrt(sum(map(lambda d: d**2, data_point)))


# refactor
class Splicer:
	def __init__ (
		self, data_stream=None, smoothing_window=10,
		threshold=75.0, increase_factor=1.001, decrease_factor=0.999, samples_cutoff=5,
		accl_ratio=0.0, gyro_ratio=1.0, magnitude_method=_euclidean_magnitude
	):
		self.data_stream = data_stream
		self.threshold   = threshold

		self.increase_factor = increase_factor
		self.decrease_factor = decrease_factor
		self.samples_cutoff  = samples_cutoff

		self.accl_ratio = accl_ratio
		self.gyro_ratio = gyro_ratio

		self.smoothing_window = tuple([0 for _ in range(smoothing_window)])
		self.magnitude_method = magnitude_method



	# on calling this for a single record send an array of a single record
	def measure_intensity (self, data_array=None, magnitude=None):
		if data_array is None:
			data_array = self.data_stream
		if magnitude is None:
			magnitude = self.magnitude_method

		gyro_data = np.array(list(map(magnitude, data_array[:,0:3])))
		accl_data = np.array(list(map(magnitude, data_array[:,3:6])))
		return ((self.gyro_ratio * gyro_data) + (self.accl_ratio * accl_data))

	# refactor me please :'(
	# on calling this for a single record send an array of a single record
	def smooth_intensities (self, data_array=None, averaging_window=None):
		# if data_array is None:
		# 	data_array = self.data_stream
		# if averaging_window is None:
		# 	averaging_window = self.smoothing_window

		def _averager (index):
			return np.nan_to_num(np.average(data_array[(index - len(averaging_window)//2):(index + len(averaging_window)//2)]))

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

		if averaging_window is None:
			averaging_window = self.smoothing_window

		return np.array(list(map(_averager, range(len(data_array))))[slicer_index:])

	def silence_segments (self, intensities=None, number_of_samples=None, threshold=None, increase_factor=None, decrease_factor=None):
		if intensities is None:
			intensities = self.measure_intensity(self.data_stream)
		if number_of_samples is None:
			number_of_samples = self.samples_cutoff
		if threshold is None:
			threshold = self.threshold
		if increase_factor is None:
			increase_factor = self.increase_factor
		if decrease_factor is None:
			decrease_factor = self.decrease_factor

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
		return [data[...,start:end,:] for start, end in segments]


if __name__ == '__main__':
	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/segmentation.sample.report.csv', delimiter=',')
	# data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/ameer/6a/6a.4_29_15_22_50.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	# ### 3 ways
	# ## 1. sum all data, perform one splicer
	# streams_sum = np.sum(data_streams, 0)
	# s = Splicer(streams_sum)
	# silence_segments = s.silence_segments()

	## 2. sum all magnitudes, perform one splicer
	s = Splicer()
	intensities_sum = sum([s.measure_intensity(stream) for stream in data_streams])

	intensities_smoothed = s.smooth_intensities(intensities_sum)

	silence_segments = s.silence_segments(intensities_sum) ## use for spliting

	# ## 3. perfrom many splicers, get intersections
	# splicers = [Splicer(data_stream) for data_stream in data_streams]
	# silence_segments = list(zip(*[
	# 		splicers[index].silence_segments() for index in range(len(data_streams))
	# 	]))
	# silence_intersects = [
	# 	(max([start for start, _ in intersection]), min([end for _, end in intersection]))
	# 		for intersection in silence_segments
	# ] ## use for spliting
	sensors_map = {
		'rng': {
			'_index'    : 0,
			'_mux_index': 2
		},
		'idx': {
			'_index'    : 1,
			'_mux_index': 3
		},
		'tmb': {
			'_index'    : 2,
			'_mux_index': 4
		},
		'mdl': {
			'_index'    : 3,
			'_mux_index': 5
		},
		'pnk': {
			'_index'    : 4,
			'_mux_index': 6
		},
		'ref': {
			'_index'    : 5,
			'_mux_index': 7
		}
	}
