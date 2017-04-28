import numpy as np
import scipy.stats as st

## TODO move later to some other place in project
import functools
class classinstancemethod:
	'''
	A describtor for having a method that is both a class method and an instance method

	Courtesy of: Mike Axiak, from StackOverflow, cheers and thanks for saving us time and effort !
	https://stackoverflow.com/questions/2589690/creating-a-method-that-is-simultaneously-an-instance-and-class-method
	'''
	def __init__(self, method):
		self.method = method

	def __get__(self, obj=None, typ=None):
		@functools.wraps(self.method)
		def _wrapper(*args, **kwargs):
			if obj is not None:
				# as instance method
				return self.method(obj, *args, **kwargs)
			else:
				# as class method
				return self.method(typ, *args, **kwargs)

		return _wrapper


class Extractor:
	_WINDOW_SIZE = 10

	@staticmethod
	def extract (data, window_size=None, multi=True):
		if not multi:
			data = np.array([data])
		if window_size is None:
			window_size = Extractor._WINDOW_SIZE

		# window the data
		data_windowed = np.array([data[:,pivot-window_size//2:pivot+window_size//2,...] for pivot in range(window_size//2, data.shape[1]-window_size//2, window_size)])

		# returns an R -> R[sensor][window][reading]['feature_method_name']
		return np.array([Extractor._extract_features(data_windowed[...,col]) for col in range(data_windowed.shape[-1])])

	@classinstancemethod
	def _extract_features (obj, data_column):
		# this condition is used to manipulate the introspective part when calculating features
		if isinstance(obj, type):
			obj = obj.__name__
		else:
			obj = 'self'

		# perform each method ending with '_feature' from given class or instance on given data
		return {feature: eval('%s.%s' % (obj, feature))(data_column) for feature in dir(eval(obj)) if feature.endswith('_feature')}

	## Features to be used

	@staticmethod
	def _autocorrelate_feature (data_streams):
		return np.array(np.nan_to_num([
			[np.correlate(window, window, mode='full') for window in stream]
				for stream in data_streams
		]))

	@staticmethod
	def _mean_feature (data_streams):
		return np.array(np.nan_to_num([
			[np.average(window) for window in stream]
				for stream in data_streams
		]))

	@staticmethod
	def _variance_feature (data_streams):
		return np.array(np.nan_to_num([
			[np.var(window) for window in stream]
				for stream in data_streams
		]))

	@staticmethod
	def _skewness_feature (data_streams):
		return np.array(np.nan_to_num([
			[st.skew(window) for window in stream]
				for stream in data_streams
		]))

	@staticmethod
	def _kurtoises_feature (data_streams):
		return np.array(np.nan_to_num([
			[st.kurtosis(window) for window in stream]
				for stream in data_streams
		]))

	@staticmethod
	def _dft_feature (data_streams):
		return np.array(np.nan_to_num([
			[np.fft.fft(window) for window in stream]
				for stream in data_streams
		]))

	@staticmethod
	def _entropy_feature (data_streams):
		return np.array(np.nan_to_num([
			[st.entropy(window) for window in stream]
				for stream in data_streams
		]))

	# # highly dependant on fourier
	# def _power_spectum_density_feature (self, data_windowed):
	# 	return self._dft_feature(data_windowed) ** 2 ## odd ??

	@staticmethod
	def _dc_component_feature (data_streams):
		return np.array(np.nan_to_num([
			[window_dft[0] for window_dft in stream]
				for stream in Extractor._dft_feature(data_streams)
		]))

	@staticmethod
	def _signal_magnitude_area_feature (data_streams):
		return np.array(np.nan_to_num([
			[sum(np.absolute(window)) for window in stream]
				for stream in data_streams
		]))

	@staticmethod
	def _integration_feature (data_streams):
		# integration := np.trapz
		return np.array(np.nan_to_num([
			[np.trapz(window) for window in stream]
				for stream in data_streams
		]))

	@staticmethod
	def _rms_feature (data_streams):
		return np.array(np.nan_to_num([
			[np.sqrt(sum(window**2))/(len(window)) for window in stream]
				for stream in data_streams
		]))


if __name__ == '__main__':
	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/Proper/data/ameer/7a/ha.4_22_14_50_54.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	featured = Extractor.extract(data_streams)

	with open('features_dump', 'w') as f:
		f.write(str(featured))
