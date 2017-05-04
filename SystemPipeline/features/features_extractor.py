import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import numpy as np
import scipy.stats as st

from features.extractor import Extractor
from utils.decorators import staticmethod


class FeaturesExtractor (Extractor):
	@staticmethod
	def autocorrelate_feature (data_column, **kwargs):
		_feature = lambda w: np.correlate(w, w, mode='full')
		return FeaturesExtractor._generic_loop(data_column, _feature)

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
	def dft_feature (data_column, **kwargs):
		return FeaturesExtractor._generic_loop(data_column, np.fft.fft)

	@staticmethod
	def entropy_feature (data_column, **kwargs):
		return FeaturesExtractor._generic_loop(data_column, st.entropy)

	@staticmethod
	def dc_component_feature (data_column, **kwargs):
		_feature = lambda dft: dft[0]
		return FeaturesExtractor._generic_loop(FeaturesExtractor.dft_feature(data_column), _feature)

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
