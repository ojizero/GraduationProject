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
	def transform (obj, **kwargs):
		transformed_dict = obj._transform(**kwargs)

		features, values = zip(*transformed_dict.items())

		return features, values

	@classinstancemethod
	def _transform (obj, **kwargs):
		extracted_feature = kwargs.get('extracted_feature')
		if extracted_feature is None:
			extracted_feature = obj._extractor.extract(**kwargs)

		# uses R -> R[column/reading]['feature_name'][sensor][window]
		return {
			'%s_stream%s_window%s' % (feature_name, s_index, w_index): feature_value
				for reading_features in extracted_feature
					for feature_name, feature_value in reading_features.items()
						for s_index, stream in enumerate(feature_value)
							for w_index, window in enumerate(feature_value)
		}



if __name__ == '__main__':
	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/ameer/7a/ha.4_22_14_50_54.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	transformed = FeaturesTransformer.transform(data=data_streams)

	with open('transformer_dump', 'w') as f:
		f.write(str(transformed))
