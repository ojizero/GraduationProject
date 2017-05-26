import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import numpy as np
from re import match
import scipy.stats as st

from features.extractor import Extractor
from utils.decorators import staticmethod
from utils.helpers import array_length_normalizer


class FeaturesExtractor (Extractor):
	_EXTRACT_ON  = lambda string: match('^[^_].*_feature(?:s?_?set)?$', string)

	@staticmethod
	def arg_max (data):
		return FeaturesExtractor._generic_loop(data, np.argmax)

	@staticmethod
	def arg_min (data):
		return FeaturesExtractor._generic_loop(data, np.argmin)

	@staticmethod
	def arg_avg (data):
		_func = lambda w: np.argmin(np.absolute(w - np.average(w)))
		return FeaturesExtractor._generic_loop(data, _func)

	@staticmethod
	def autocorrelate_feature_set (**kwargs):
		length_normalizer = kwargs.pop('length_normalizer', array_length_normalizer)
		_autocorrelation  = lambda w: np.correlate(w, w, mode='full')

		# calculate main feature
		autocorrelation = FeaturesExtractor._generic_loop(kwargs['data_column'], _autocorrelation)

		# reduce feature to a controlled feature set
		normalized_autocorrelation = FeaturesExtractor._generic_loop(autocorrelation, length_normalizer)
		arg_max_autocorrelation    = FeaturesExtractor.arg_max(autocorrelation)
		arg_min_autocorrelation    = FeaturesExtractor.arg_min(autocorrelation)
		avg_autocorrelation        = FeaturesExtractor.mean_feature(autocorrelation)
		arg_avg_autocorrelation    = FeaturesExtractor.arg_avg(autocorrelation)

		return (
			( {
				'normalized_autocorrelation': normalized_autocorrelation[s_index,w_index],
				'arg_max_autocorrelation'   : arg_max_autocorrelation[s_index,w_index],
				'arg_min_autocorrelation'   : arg_min_autocorrelation[s_index,w_index],
				'avg_autocorrelation'       : avg_autocorrelation[s_index,w_index],
				'arg_avg_autocorrelation'   : arg_avg_autocorrelation[s_index,w_index]
			} for w_index, _ in enumerate(stream) )
				for s_index, stream in enumerate(kwargs['data_column'])
		)

	@staticmethod
	def frequency_feature_set (**kwargs):
		length_normalizer = kwargs.pop('length_normalizer', array_length_normalizer)
		# companion feature
		_dc_bias          = lambda dft: sum(dft)/len(dft)

		# calculate main feature
		dft = FeaturesExtractor._generic_loop(kwargs['data_column'], np.fft.fft)

		# reduce feature to a controlled feature set
		normalized_dft = FeaturesExtractor._generic_loop(dft, length_normalizer)
		arg_max_dft    = FeaturesExtractor.arg_max(dft)
		arg_min_dft    = FeaturesExtractor.arg_min(dft)
		avg_dft        = FeaturesExtractor.mean_feature(dft)
		arg_avg_dft    = FeaturesExtractor.arg_avg(dft)
		dc_bias        = FeaturesExtractor._generic_loop(dft, _dc_bias)

		return (
			( {
				'normalized_dft': normalized_dft[s_index,w_index],
				'arg_max_dft'   : arg_max_dft[s_index,w_index],
				'arg_min_dft'   : arg_min_dft[s_index,w_index],
				'avg_dft'       : avg_dft[s_index,w_index],
				'arg_avg_dft'   : arg_avg_dft[s_index,w_index],
				'dc_bias'       : dc_bias[s_index, w_index]
			} for w_index, _ in enumerate(stream) )
				for s_index, stream in enumerate(kwargs['data_column'])
		)

	@staticmethod
	def mean_feature (data_column, **kwargs):
		return FeaturesExtractor._generic_loop(data_column, np.average)

	@staticmethod
	def variance_feature (data_column, **kwargs):
		return FeaturesExtractor._generic_loop(data_column, np.var)

	@staticmethod
	def skewness_feature (data_column, **kwargs):
		return FeaturesExtractor._generic_loop(data_column, st.skew)

	@staticmethod
	def kurtoises_feature (data_column, **kwargs):
		return FeaturesExtractor._generic_loop(data_column, st.kurtosis)

	@staticmethod
	def entropy_feature (data_column, **kwargs):
		return FeaturesExtractor._generic_loop(data_column, st.entropy)

	@staticmethod
	def signal_magnitude_area_feature (data_column, **kwargs):
		_feature = lambda w: sum(np.absolute(w))
		return FeaturesExtractor._generic_loop(data_column, _feature)

	@staticmethod
	def integration_feature (data_column, **kwargs):
		# integration := np.trapz
		return FeaturesExtractor._generic_loop(data_column, np.trapz)

	@staticmethod
	def rms_feature (data_column, **kwargs):
		_feature = lambda w: np.sqrt(sum(w**2))/len(w)
		return FeaturesExtractor._generic_loop(data_column, _feature)


if __name__ == '__main__':
	pass
	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/ameer/7a/ha.4_22_14_50_54.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	features = FeaturesExtractor.extract(data=data_streams)

	with open('features_dump', 'w') as f:
		f.write(str(features))
