import numpy as np
import scipy.stats as st

class Extractor:
	def __init___feature (self, window_size=20):
		self.window_size = window_size

	def claculate_features (self, data, window_size=None):
		if window_size is None:
			window_size = self.window_size
		# window the data
		data_windowed = [data[index-window_size//2:index+window_size//2] for index in range(window_size//2,len(data)-window_size//2)]
		#call extract_features
		# heek 3ala kul el columns one by one
		return [self.extract_features(data_column) for data_column in data_windowed[:,:]]

	def extract_features (self, data_column):
		return {feature: eval('self.%s'%feature)(data_column) for feature in self.__dir__() if feature.endswith('_feature')}

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

	def _dft_feature (self, data_windowed):
		return np.array(np.nan_to_num([np.fft(data_window) for data_window in data_windowed]))

	def _entropy_feature (self, data_windowed):
		return np.array(np.nan_to_num([st.entropy(data_window) for data_window in data_windowed]))

	# def _crosscorrelation_feature (self, data_windowed):
	# 	correlations_array = [
	# 		np.correlate(data_window, np.concatenate(([0] * n, data_window[n:])), mode='full')
	# 			for n in range(len(data_window))
	# 	]

	# 	pass

	def _power_spectum_density_feature (self, data_windowed):
		return self._dft_feature(data_windowed) ** 2

	def _dc_component_feature (self, data_windowed):
		return self._power_spectum_density_feature(data_windowed)[0]

	def _signal_magnitude_area_feature (self, data_windowed):
		return [sum(np.absolute(data_window)) for data_window in data_windowed]

	def _integration_feature (self, data_windowed):
		# integration := np.trapz
		return np.trapz(data_windowed)

	def _rms_feature (self, data_windowed):
		return [np.sqrt(sum(data_window ** 2))/(len(data_window)) for data_window in data_windowed]