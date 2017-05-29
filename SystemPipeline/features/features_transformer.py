import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


from features.features_extractor import FeaturesExtractor
from utils.decorators import classinstancemethod
from utils.helpers import flatten_key_val_vector, complex_flattener

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
		extracted_feature = kwargs.pop('extracted_feature', None)
		if extracted_feature is None:
			windows_count = kwargs.pop('windows_count', obj._windows_count)
			if windows_count > 0:
				kwargs['window_size'] = kwargs['data'].shape[1] // windows_count

			extracted_feature = obj._extractor.extract(**kwargs)

		transformed_dict = obj._transform(extracted_feature)

		flattened_vector = flatten_key_val_vector(vector=transformed_dict.items(), data_modifier=complex_flattener)

		features, values = zip(*flattened_vector)

		return features, values

	@staticmethod
	def _transform (extracted_feature):
		# uses R -> R[column/reading]['feature_name'][sensor][window]
		return {
			'%s_reading%s_stream%s_window%s' % (feature_name, r_index, s_index, w_index): window
				for r_index, reading_features in enumerate(extracted_feature)
					for feature_name, feature_values in reading_features.items()
						for s_index, stream in enumerate(feature_values)
							for w_index, window in enumerate(stream)
		}



if __name__ == '__main__':
	import numpy as np
	from collections import Iterable

	data_whole = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/our_data/ameer/6a/6a.4_29_15_22_50.csv', delimiter=',')
	data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

	transformed = FeaturesTransformer.transform(data=data_streams)

	# assert success
	for v in transformed[1]:
		assert not isinstance(v, np.complex), 'failed'

	with open('transformer_dump', 'w') as f:
		f.write(str(transformed))
