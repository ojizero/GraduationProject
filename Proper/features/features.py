import numpy as np
import scipy.stats as st

class Extractor:
	window_size = 10

	@staticmethod
	def extract (data, window_size=None, multi=True):
		data        = np.array([data])      if not multi
		window_size = Extractor.window_size if window_size is None

		# window the data
		data_windowed = [data[:,index-window_size//2:index+window_size//2,...] for index in range(window_size//2,len(data)-window_size//2)]

		# returns an R -> R[sensor][window][reading]['feature_method_name']
		return [Extractor._extract_features(data_windowed[...,col]) for col in range(data_windowed.shape[-1])]

	@classmethod
	def _extract_features (cls, data_column):
		return {feature: eval('%s.%s' % (cls.__name__, feature))(data_column) for feature in dir(cls) if feature.endswith('_feature')}

	## Features to be used

	@staticmethod
	def _autocorrelate_feature (data_windowed):
		return np.array(np.nan_to_num([
			np.correlate(data_window, data_window, mode='full') for data_window in data_windowed
		]))

	@staticmethod
	def _mean_feature (data_windowed):
		return np.array(np.nan_to_num([np.average(data_window) for data_window in data_windowed]))

	@staticmethod
	def _variance_feature (data_windowed):
		return np.array(np.nan_to_num([np.var(data_window) for data_window in data_windowed]))

	@staticmethod
	def _skewness_feature (data_windowed):
		return np.array(np.nan_to_num([st.skew(data_window) for data_window in data_windowed]))

	@staticmethod
	def _kurtoises_feature (data_windowed):
		return np.array(np.nan_to_num([st.kurtosis(data_window) for data_window in data_windowed]))

	@staticmethod
	def _dft_feature (data_windowed):
		return np.array(np.nan_to_num([np.fft(data_window) for data_window in data_windowed]))

	@staticmethod
	def _entropy_feature (data_windowed):
		return np.array(np.nan_to_num([st.entropy(data_window) for data_window in data_windowed]))

	# # highly dependant on fourier
	# def _power_spectum_density_feature (self, data_windowed):
	# 	return self._dft_feature(data_windowed) ** 2 ## odd ??

	@staticmethod
	def _dc_component_feature (data_windowed):
		return self._power_spectum_density_feature(data_windowed)[0]

	@staticmethod
	def _signal_magnitude_area_feature (data_windowed):
		return [sum(np.absolute(data_window)) for data_window in data_windowed]

	@staticmethod
	def _integration_feature (data_windowed):
		# integration := np.trapz
		return np.trapz(data_windowed)

	@staticmethod
	def _rms_feature (data_windowed):
		return [np.sqrt(sum(data_window ** 2))/(len(data_window)) for data_window in data_windowed]