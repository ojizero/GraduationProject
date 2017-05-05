import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


from features.features_extractor import FeaturesExtractor
from utils.decorators import classinstancemethod


class FeaturesTransformer:
	'''
	produces a feature vector from the result of feature extraction
	'''
	_extractor     = FeaturesExtractor
	_windows_count = 5 # number of windows, -1 for letting extractor decide

	def __init__ (self, *args, **kwargs):
		self._extractor = kwargs.pop('extractor', FeaturesTransformer._extractor)

	@classinstancemethod
	def transform (obj, **kwargs):
		extracted_feature = kwargs.get('extracted_feature', None)
		if extracted_feature is None:
			windows_count = kwargs.get('windows_count', obj._windows_count)
			if windows_count > 0:
				kwargs['window_size'] = kwargs['data'].shape[1] // windows_count

			kwargs['extracted_feature'] = obj._extractor.extract(**kwargs)

		transformed_dict = obj._transform(**kwargs)

		features, values = zip(*transformed_dict.items())

		return features, values

	@staticmethod
	def _transform (extracted_feature):
		# uses R -> R[column/reading]['feature_name'][sensor][window]
		return {
			'%s_reading%s_stream%s_window%s' % (feature_name, r_index, s_index, w_index): feature_value
				for r_index, reading_features in enumerate(extracted_feature)
					for feature_name, feature_value in reading_features.items()
						for s_index, stream in enumerate(feature_value)
							for w_index, window in enumerate(stream)
		}



if __name__ == '__main__':
	import numpy as np


	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/ameer/7a/ha.4_22_14_50_54.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	transformed = FeaturesTransformer.transform(data=data_streams)

	with open('transformer_dump', 'w') as f:
		f.write(str(transformed))
