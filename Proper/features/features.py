import numpy as np
import scipy.stats as sp

class Extractor:
	def __init__ (self, window_size=20):
		pass

	def claculate_features (self, data):
		pass

	# windowed_data = [data[index-(window_size//2):index+(window_size//2)] for index in range(window_size//2,len(data)-(window_size//2))]
	def _autocorrelate (self, data_windowed):
		return [np.correlate(data_window, data_window, mode='full') for data_window in data_windowed]

	def _mean (self, data_windowed):
		return [np.average(data_window) for data_window in data_windowed]

	def _variance (self, data_windowed):
		return [np.var(data_window) for data_window in data_windowed]

	def _skeyness (self, data_windowed):
		return [sp.skew(data_window) for data_window in data_windowed]

	def _kurtoises (self, data_windowed):
		return [sp.kurtosis(data_window) for data_window in data_windowed]

	def _signal_magnitude_area (self, data_windowed):
		# return []
		pass

	def extract_features (self, data):
		feature_set = {(feature, eval('self.feature')) for feature in self.__dir__() if feature.endswith('_feature')}
		features = {}

		for feature_name, feature_function in feature_set:
			features[feature_name] = feature_function(data)

		return features

	def format_features (self, feature_set):
		pass