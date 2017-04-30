import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


from features.extractor import Extractor


class FeaturesExtractor (Extractor):
	@staticmethod
	def autocorrelate_feature (data_streams):
		_feature = lambda w: np.correlate(w, w, mode='full')
		return Extractor._generic_feature_applier(data_streams, _feature)

	@staticmethod
	def mean_feature (data_streams):
		return Extractor._generic_feature_applier(data_streams, np.average)

	@staticmethod
	def variance_feature (data_streams):
		return Extractor._generic_feature_applier(data_streams, np.var)

	@staticmethod
	def skewness_feature (data_streams):
		return Extractor._generic_feature_applier(data_streams, st.skew)

	@staticmethod
	def kurtoises_feature (data_streams):
		return Extractor._generic_feature_applier(data_streams, st.kurtosis)

	@staticmethod
	def dft_feature (data_streams):
		return Extractor._generic_feature_applier(data_streams, np.fft.fft)

	@staticmethod
	def entropy_feature (data_streams):
		return Extractor._generic_feature_applier(data_streams, st.entropy)

	@staticmethod
	def dc_component_feature (data_streams):
		_feature = lambda dft: dft[0]
		return Extractor._generic_feature_applier(Extractor.dft_feature(data_streams), _feature)

	@staticmethod
	def signal_magnitude_area_feature (data_streams):
		_feature = lambda w: sum(np.absolute(w))
		return Extractor._generic_feature_applier(data_streams, _feature)

	@staticmethod
	def integration_feature (data_streams):
		# integration := np.trapz
		return Extractor._generic_feature_applier(data_streams, np.trapz)

	@staticmethod
	def rms_feature (data_streams):
		_feature = lambda w: np.sqrt(sum(w**2))/len(w)
		return Extractor._generic_feature_applier(data_streams, _feature)


if __name__ == '__main__':
	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/ameer/7a/ha.4_22_14_50_54.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	featured = FeaturesExtractor.extract(data_streams)

	with open('features_dump', 'w') as f:
		f.write(str(featured))
