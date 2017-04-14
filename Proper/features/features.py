import numpy as np
import scipy.stats as st

class Extractor:
	def __init___feature (self, window_size=20):
		self.window_size = window_size

	def claculate_features (self, data, window_size=self.window_size):
		# window the data
		data_windowed = [data[index-window_size//2:index+window_size//2] for index in range(window_size//2,len(data)-window_size//2)]
		#call extract_features
		return self.extract_features(data_windowed)

	def extract_features (self, data):
		return {feature: eval('self.feature')(data) for feature in self.__dir__() if feature.endswith('_feature')}

	## Features to be used
	def _autocorrelate_feature (self, data_windowed):
		return np.array(np.nan_to_num([np.correlate(data_window, data_window, mode='full') for data_window in data_windowed]))

	def _mean_feature (self, data_windowed):
		return np.array(np.nan_to_num([np.average(data_window) for data_window in data_windowed]))

	def _variance_feature (self, data_windowed):
		return np.array(np.nan_to_num([np.var(data_window) for data_window in data_windowed]))

	def _skeyness_feature (self, data_windowed):
		return np.array(np.nan_to_num([st.skew(data_window) for data_window in data_windowed]))

	def _kurtoises_feature (self, data_windowed):
		return np.array(np.nan_to_num([st.kurtosis(data_window) for data_window in data_windowed]))

	def _signal_magnitude_area_feature (self, data_windowed):
		# return []
		pass