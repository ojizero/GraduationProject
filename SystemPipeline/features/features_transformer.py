import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import numpy as np

from features.features_extractor import FeaturesExtractor
from utils.decorators import classinstancemethod

class FeaturesTransformer:
	'''
	produces a feature vector from the result of feature extraction
	'''
	_extractor = FeaturesExtractor


	def __init__ (self, *args, **kwargs):
		self._extractor = kwargs.pop('extractor', FeaturesTransformer._extractor)

	# label => [feature1_window1_stream1, feature1_window2_stream1, ..., featureN_windowN_streamN]

	@classinstancemethod
	def _transform (obj, **kwargs):
		features = kwargs.get('features')
		if features is None:
			features = obj._extractor.extract(**kwargs)

		result = []

		for data_column, streams in features:
			for s_index in range(len(streams)):
				stream = streams[s_index]
				for w_index in range(len(stream)):
					window = stream[w_index]

					feature_tuple = ('%s_stream%s_window%s' % (data_column, s_index, w_index), window)

					result.append(feature_tuple)

		return dict(result)



if __name__ == '__main__':
	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/ameer/7a/ha.4_22_14_50_54.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	FeaturesTransformer.transform(data=data_streams)
