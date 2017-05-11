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
	_EXTRACT_ON  = lambda string: match('^[^_].*_feature$', string)

	@staticmethod
	def autocorrelate_feature (**kwargs):
		length_normalizer = kwargs.pop('length_normalizer', array_length_normalizer)
		_autocorrelation  = lambda w: np.correlate(w, w, mode='full')
		autocorrelation   = FeaturesExtractor._generic_loop(kwargs['data_column'], _autocorrelation)

		normalized_autocorrelation = \
			FeaturesExtractor._generic_loop(autocorrelation, length_normalizer)

		arg_max_autocorrelation    = \
			FeaturesExtractor._generic_loop(autocorrelation, np.argmax)

		arg_min_autocorrelation    = \
			FeaturesExtractor._generic_loop(autocorrelation, np.argmin)

		avg_autocorrelation        = \
			FeaturesExtractor._generic_loop(autocorrelation, np.average)

		arg_avg_autocorrelation    = \
			FeaturesExtractor._generic_loop(autocorrelation, lambda a: np.argmin(np.absolute(a - np.average(a))))

		return (
			( {
				'normalized_autocorrelation': normalized_autocorrelation[s_index,w_index],
				'arg_max_autocorrelation': arg_max_autocorrelation[s_index,w_index],
				'arg_min_autocorrelation': arg_min_autocorrelation[s_index,w_index],
				'avg_autocorrelation': avg_autocorrelation[s_index,w_index],
				'arg_avg_autocorrelation': arg_avg_autocorrelation[s_index,w_index]
			} for w_index, _ in enumerate(stream) )
				for s_index, stream in enumerate(kwargs['data_column'])
		)

	@staticmethod
	def _dft_feature (data_column, **kwargs):
		return FeaturesExtractor._generic_loop(data_column, np.fft.fft)

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

	# @staticmethod
	# def dc_component_feature (data_column, **kwargs):
	# 	_feature = lambda dft: dft[0]
	# 	return FeaturesExtractor._generic_loop(FeaturesExtractor._dft_feature(data_column), _feature)

	# @staticmethod
	# def _dft_argmax_feature (data_column, **kwargs):
	# 	_feature  = lambda dft: np.argmax(dft)
	# 	return FeaturesExtractor._generic_loop(FeaturesExtractor._dft_feature(data_column), _feature)

	# @staticmethod
	# def _dft_argmin_feature (data_column, **kwargs):
	# 	_feature  = lambda dft: np.argmin(dft)
	# 	return FeaturesExtractor._generic_loop(FeaturesExtractor._dft_feature(data_column), _feature)

	# @staticmethod
	# def _dft_average_feature (data_column, **kwargs):
	# 	_feature = lambda dft: np.average(dft)
	# 	return FeaturesExtractor._generic_loop(FeaturesExtractor._dft_feature(data_column), _feature)

	# @staticmethod
	# def _dft_argaverage_feature (data_column, **kwargs):
	# 	_feature = lambda dft: dft[dft == np.average(dft)]
	# 	return FeaturesExtractor._generic_loop(FeaturesExtractor._dft_feature(data_column), _feature)



if __name__ == '__main__':
	pass
	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/ameer/7a/ha.4_22_14_50_54.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	features = FeaturesExtractor.extract(data=data_streams)

	with open('features_dump', 'w') as f:
		f.write(str(features))
